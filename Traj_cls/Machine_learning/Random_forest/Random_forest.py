# -*- coding: utf-8 -*-
# @Time    : 2023/10/10 20:25
# @Author  : Dongliang

import os

import joblib
import pandas as pd
import scipy
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

import AAISPT.Traj_cls.utils.config as c
from AAISPT.Traj_cls.utils.Metric_evaluation import MetricEvaluation

# Load data
path = r'D:\TrajSeg-Cls\TrajSEG-CLS_V3\CLS\Endocytosis_NEW\5D\Feature_based V1'
savepath = os.path.join(path, 'RandomForest')
if not os.path.exists(savepath):
    os.mkdir(savepath)

mode = 'Exp'
num_class, label_name = c.config(mode=mode, path=path)

# Load data
train = scipy.io.loadmat(os.path.join(path, 'stand_train_feature.mat'))
train_x, train_y = train['data'], train['label'].T.ravel()

val = scipy.io.loadmat(os.path.join(path, 'stand_val_feature.mat'))
val_x, val_y = val['data'], val['label'].T.ravel()

# Load the best parameters
params = pd.read_csv(os.path.join(savepath, 'bayes_search_results.csv'), sep=',')
index = params['rank_test_score'].idxmin()
n_estimators = params.loc[index, 'param_n_estimators']
# n_estimators = 131

# Create RandomForestClassifier
# The larger the n_estimators, the better the model.
model = RandomForestClassifier(
    n_estimators=n_estimators,
    random_state=37
)

# Train the model using the training sets and check score
model.fit(train_x, train_y)
r_score = model.score(val_x, val_y)
print("Random Forest: {}".format(r_score))

# Save model
joblib.dump(model, os.path.join(savepath, 'model.pkl'))

# Predict Output
predicted = model.predict(val_x)
probs = model.predict_proba(val_x)

# Save predicted label
scipy.io.savemat(os.path.join(savepath, 'cls_pre.mat'),
                 {
                     'clspre': predicted,
                     'clsgt': val_y,
                     'probs': probs
                 }
                 )

# Evaluation model
confusionmat = confusion_matrix(val_y, predicted)
evaluator = MetricEvaluation(num_classes=num_class, label_names=label_name, save_path=savepath)
evaluator.plot_confusion_matrix(confusionmat=confusionmat, normalize=True)
metrics = evaluator.calculate_metrics(gt=val_y, predicted=predicted)
evaluator.write_metric_log(confusion_mat=confusionmat, metrics=metrics)
print('Done!')
