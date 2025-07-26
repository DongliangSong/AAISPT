# -*- coding: utf-8 -*-
# @Time    : 2023/10/10 20:25
# @Author  : Dongliang

import os

import joblib
import pandas as pd
import scipy
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix

import AAISPT.Traj_cls.utils.config as c
from AAISPT.Traj_cls.utils.Metric_evaluation import MetricEvaluation

path = r'D:\TrajSeg-Cls\TrajSEG-CLS_V3\CLS\Endocytosis_NEW\5D\Feature_based V1'
savepath = os.path.join(path, 'GBDT')
if not os.path.exists(savepath):
    os.mkdir(savepath)

mode = 'Exp'
num_class, label_name = c.config(mode=mode, path=path)

# Load data
train = scipy.io.loadmat(os.path.join(path, 'stand_train_feature.mat'))
train_x, train_y = train['data'], train['label'].T.ravel()
val = scipy.io.loadmat(os.path.join(path, 'stand_val_feature.mat'))
val_x, val_y = val['data'], val['label'].T.ravel()

# Load best parameters
params = pd.read_csv(os.path.join(savepath, 'bayes_search_results.csv'), sep=',')
index = params['rank_test_score'].idxmin()

n_estimators = params.loc[index, 'param_n_estimators']
learning_rate = params.loc[index, 'param_learning_rate']
max_depth = params.loc[index, 'param_max_depth']
min_samples_split = params.loc[index, 'param_min_samples_split']
min_samples_leaf = params.loc[index, 'param_min_samples_leaf']
# max_features = params.loc[index, 'param_max_features']
subsample = params.loc[index, 'param_subsample']

# Create GradientBoostingClassifier
# n_estimators : The number of weak learners(trees)
# learning_rate : controls over-fitting
# tree depth(max_depth) or the number of leaf nodes (max_leaf_nodes) :  control the size of each tree
# By reducing the step size (learning_rate) and increasing the maximum number of iterations (n_estimators),
# the generalization ability of the model is increased.
gbdt = GradientBoostingClassifier(
    n_estimators=n_estimators,
    learning_rate=learning_rate,
    max_depth=max_depth,
    min_samples_split=min_samples_split,
    min_samples_leaf=min_samples_leaf,
    max_features=7,
    subsample=subsample,
    random_state=0
)

gbdt.fit(train_x, train_y)
score_g = gbdt.score(val_x, val_y)
print("gbdt:{} \n".format(score_g))

# Save Models
joblib.dump(gbdt, os.path.join(savepath, 'model.pkl'))

# Predict Output
predicted = gbdt.predict(val_x)
probs = gbdt.predict_proba(val_x)

# Save predicted label
scipy.io.savemat(os.path.join(savepath, 'cls_pre.mat'),
                 {'clspre': predicted,
                  'clsgt': val_y,
                  'probs': probs
                  }
                 )

# Evaluation model
confusionmat = confusion_matrix(val_y, predicted)
evaluator = MetricEvaluation(num_classes=num_class, label_names=label_name, save_path=savepath)
evaluator.plot_confusion_matrix(confusionmat=confusionmat, normalize=True)

# Calculate Metric
metrics = evaluator.calculate_metrics(gt=val_y, predicted=predicted)
evaluator.write_metric_log(confusion_mat=confusionmat, metrics=metrics)
print('Done!')
