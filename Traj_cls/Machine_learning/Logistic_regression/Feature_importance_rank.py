# -*- coding: utf-8 -*-
# @Time    : 2023/7/6 15:42
# @Author  : Dongliang

import os

import joblib
import scipy
from sklearn.inspection import permutation_importance

import AAISPT.Traj_cls.utils.config as c
from AAISPT.Traj_cls.utils.Metric_evaluation import MetricEvaluation

path = r'D:\TrajSeg-Cls\TrajSEG-CLS_V3\CLS\Endocytosis_NEW\5D\Feature_based'
model_path = os.path.join(path, 'LR')
savepath = model_path
if not os.path.exists(savepath):
    os.mkdir(savepath)

mode = 'Exp'
num_class, label_name = c.config(mode=mode, path=path)
dim, feature_names = c.config_feature(path=path)

# Load Models and dataset
model = joblib.load(os.path.join(model_path, 'model.pkl'))

test = scipy.io.loadmat(os.path.join(path, 'stand_test_feature.mat'))
test_x, test_y = test['data'], test['label'].T

# Calculate feature importance using permutation importance
scoring = ['r2', 'neg_mean_absolute_percentage_error', 'neg_mean_squared_error']
r = permutation_importance(
    model, test_x, test_y,
    n_repeats=30,
    random_state=0
)

for i in r.importances_mean.argsort()[::-1]:
    if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
        print(f"{feature_names[i]:<12}:"
              f"{r.importances_mean[i]:.3f}"
              f" +/- {r.importances_std[i]:.3f}")
print('*' * 50)

# Plot barh
evaluator = MetricEvaluation(num_classes=num_class, label_names=label_name, save_path=savepath, model_path=model_path)
evaluator.plot_features(feature_names=feature_names, feature_importance=r.importances_mean)

scipy.io.savemat(os.path.join(savepath, 'feature_importance.mat'),
                 {
                     'feature_names': feature_names,
                     'importance_mean': r.importances_mean,
                     'importance_std': r.importances_std
                 }
                 )
print('Done!')
