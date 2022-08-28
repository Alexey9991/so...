import torch
from torch import nn
from torch.nn import functional as F

from pytorch_ranger import Ranger

class ResidualLSTM(nn.Module):

    def __init__(self, d_model):
        super(ResidualLSTM, self).__init__()
        self.LSTM=nn.LSTM(d_model, d_model, num_layers=1, bidirectional=True)
        self.linear1=nn.Linear(d_model*2, d_model*4)
        self.linear2=nn.Linear(d_model*4, d_model)


    def forward(self, x):
        res=x
        x, _ = self.LSTM(x)
        x=F.relu(self.linear1(x))
        x=self.linear2(x)
        x=res+x
        return x
    
class SAKTModel(nn.Module):
    def __init__(self, n_skill, n_cat, nout, max_seq=100, embed_dim=128, pos_encode='LSTM', nlayers=2, rnnlayers=3,
    dropout=0.1, nheads=8):
        super(SAKTModel, self).__init__()
        self.n_skill = n_skill
        self.embed_dim = embed_dim
        #self.pos_encoder = PositionalEncoding(embed_dim, dropout)
        if pos_encode=='LSTM':
            self.pos_encoder = nn.ModuleList([ResidualLSTM(embed_dim) for i in range(rnnlayers)])
        elif pos_encode=='GRU':
            self.pos_encoder = nn.ModuleList([ResidualGRU(embed_dim) for i in range(rnnlayers)])
        elif pos_encode=='GRU2':
            self.pos_encoder = nn.GRU(embed_dim,embed_dim, num_layers=2,dropout=dropout)
        elif pos_encode=='RNN':
            self.pos_encoder = nn.RNN(embed_dim,embed_dim,num_layers=2,dropout=dropout)
        self.pos_encoder_dropout = nn.Dropout(dropout)
        self.embedding = nn.Linear(n_skill, embed_dim)
        self.cat_embedding = nn.Embedding(n_cat, embed_dim, padding_idx=0)
        self.layer_normal = nn.LayerNorm(embed_dim)
        encoder_layers = [nn.TransformerEncoderLayer(embed_dim, nheads, embed_dim*4, dropout) for i in range(nlayers)]
        conv_layers = [nn.Conv1d(embed_dim,embed_dim,(nlayers-i)*2-1,stride=1,padding=0) for i in range(nlayers)]
        deconv_layers = [nn.ConvTranspose1d(embed_dim,embed_dim,(nlayers-i)*2-1,stride=1,padding=0) for i in range(nlayers)]
        layer_norm_layers = [nn.LayerNorm(embed_dim) for i in range(nlayers)]
        layer_norm_layers2 = [nn.LayerNorm(embed_dim) for i in range(nlayers)]
        self.transformer_encoder = nn.ModuleList(encoder_layers)
        self.conv_layers = nn.ModuleList(conv_layers)
        self.layer_norm_layers = nn.ModuleList(layer_norm_layers)
        self.layer_norm_layers2 = nn.ModuleList(layer_norm_layers2)
        self.deconv_layers = nn.ModuleList(deconv_layers)
        self.nheads = nheads
        self.pred = nn.Linear(embed_dim, nout)
        self.downsample = nn.Linear(embed_dim*2,embed_dim)

    def forward(self, numerical_features, categorical_features=None):
        device = numerical_features.device
        numerical_features=self.embedding(numerical_features)
        x = numerical_features#+categorical_features
        x = x.permute(1, 0, 2)
        for lstm in self.pos_encoder:
            lstm.LSTM.flatten_parameters()
            x=lstm(x)

        x = self.pos_encoder_dropout(x)
        x = self.layer_normal(x)



        for conv, transformer_layer, layer_norm1, layer_norm2, deconv in zip(self.conv_layers,
                                                               self.transformer_encoder,
                                                               self.layer_norm_layers,
                                                               self.layer_norm_layers2,
                                                               self.deconv_layers):
            #LXBXC to BXCXL
            res=x
            x=F.relu(conv(x.permute(1,2,0)).permute(2,0,1))
            x=layer_norm1(x)
            x=transformer_layer(x)
            x=F.relu(deconv(x.permute(1,2,0)).permute(2,0,1))
            x=layer_norm2(x)
            x=res+x

        x = x.permute(1, 0, 2)

        output = self.pred(x)

        return output.squeeze(-1)
    
model = SAKTModel(train.shape[-1], 10, 1, embed_dim=256, pos_encode='LSTM',
                  max_seq=None, nlayers=3, rnnlayers=3,
                  dropout=0,nheads=16).cuda()

optimizer = Ranger(model.parameters(), lr=8e-4)
criterion = nn.L1Loss(reduction='none')

epochs=150
val_metric = 100
best_metric = 100
cos_epoch=int(epochs*0.75)
scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,(epochs-cos_epoch)*len(train_dataloader))
steps_per_epoch=len(train_dataloader)
val_steps=len(val_dataloader)

for epoch in range(epochs):
    model.train()
    train_loss=0
    t=time.time()
    for step,batch in enumerate(train_dataloader):
        #series=batch.to(device)#.float()
        features,targets,mask=batch
        features=features.cuda()
        targets=targets.cuda()
        mask=mask.cuda()
        #exit()

        optimizer.zero_grad()
        output=model(features,None)
        #exit()
        #exit()

        loss=criterion(output,targets)#*loss_weight_vector
        loss=torch.masked_select(loss,mask)
        loss=loss.mean()
        loss.backward()
        # with amp.scale_loss(loss, optimizer) as scaled_loss:
        #     scaled_loss.backward()
        optimizer.step()

        train_loss+=loss.item()
        #scheduler.step()
        print ("Step [{}/{}] Loss: {:.3f} Time: {:.1f}"
                           .format(step+1, steps_per_epoch, train_loss/(step+1), time.time()-t),end='\r',flush=True)
        if epoch > cos_epoch:
            scheduler.step()
        #break
    print('')
    train_loss/=(step+1)

    #exit()
    model.eval()
    val_metric=[]
    val_loss=0
    t=time.time()
    preds=[]
    truths=[]
    masks=[]
    for step,batch in enumerate(val_dataloader):
        features,targets,mask=batch
        features=features.cuda()
        targets=targets.cuda()
        mask=mask.cuda()
        with torch.no_grad():
            output=model(features,None)

            loss=criterion(output,targets)
            loss=torch.masked_select(loss,mask)
            loss=loss.mean()
            val_loss+=loss.item()
            #val_metric.append(MCMAE(output.reshape(-1,4),labels.reshape(-1,4),stds[-4:]))
            preds.append(output.cpu())
            truths.append(targets.cpu())
            masks.append(mask.cpu())
        print ("Validation Step [{}/{}] Loss: {:.3f} Time: {:.1f}"
                           .format(step+1, val_steps, val_loss/(step+1), time.time()-t),end='\r',flush=True)

    preds=torch.cat(preds).numpy()
    truths=torch.cat(truths).numpy()
    masks=torch.cat(masks).numpy()
    val_metric=(np.abs(truths-preds)*masks).sum()/masks.sum()#*stds['pressure']
    #exit()
    print('')
    #val_metric=torch.stack(val_metric).mean().cpu().numpy()
    val_loss/=(step+1)


    if val_metric < best_metric:
        best_metric=val_metric
        torch.save(model.state_dict(),f'model{fold}.pth')