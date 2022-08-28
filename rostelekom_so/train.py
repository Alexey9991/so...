from matplotlib import colors, pyplot as plt
%matplotlib inline

import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split

#------------------- Разделим выборку

X_train, X_val, y_train, y_val = train_test_split(train.drop(columns=['label']), train['label'], random_state=1, test_size=0.10, shuffle=True, stratify=train['label'])
X_train.shape, X_val.shape, y_train.shape, y_val.shape

#------------------- Импортируем CatBoostClassifier и сделаем предсказание

from sklearn.pipeline import make_pipeline, Pipeline
from catboost import CatBoostClassifier

model = Pipeline(steps=[('regressor', CatBoostClassifier(class_weights=[0.3, 0.7], iterations=500, learning_rate=0.1, depth=5, random_state=1))])
model.fit(X_train, y_train)

y_pred = model.predict_proba(test.drop(columns=['label']))

submission = pd.read_csv('C:/Users/user/damn-dataset/Название_команды.csv', sep=";")
submission['label'] = y_pred

submission.to_csv('C:/Users/user/damn-dataset/so...csv', index=False)

#-------------------Выведем Feature Importance

feature_importance = model.named_steps['regressor'].feature_importances_

sorted_idx = np.argsort(feature_importance)
fig = plt.figure(figsize=(12, 6))
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), np.array(test.drop(columns=['label']).columns)[sorted_idx])
plt.title('Feature Importance')
