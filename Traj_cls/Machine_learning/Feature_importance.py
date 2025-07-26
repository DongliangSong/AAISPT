# -*- coding: utf-8 -*-
# @Time    : 2025/4/1 11:27
# @Author  : Dongliang

import os

import joblib
import numpy as np
import scipy
from sklearn.base import is_classifier, is_regressor
from sklearn.inspection import permutation_importance

import AAISPT.Traj_cls.utils.config as c
from AAISPT.Traj_cls.utils.Metric_evaluation import MetricEvaluation

path = r'D:\TrajSeg-Cls\TrajSEG-CLS_V3\CLS\YanYu_NEW\Feature_based V1'

# Load feature name
dim, feature_names = c.config_feature(path=path)
num_class, label_name = c.config(mode='Exp', path=path)

# Load test data
test = scipy.io.loadmat(os.path.join(path, 'stand_test_feature.mat'))
test_x, test_y = test['data'], test['label'].T

print('test_x shape"', test_x.shape)
print('test_y shape:', test_y.shape)
if test_y.ndim > 1:
    test_y = test_y.ravel()

if np.any(np.isnan(test_x)) or np.any(np.isnan(test_y)):
    raise ValueError('There are missing values in the data (NaN)')
if np.any(np.isinf(test_x)) or np.any(np.isinf(test_y)):
    raise ValueError('There are infinite values in the data (inf)')

# Define all models
models = [
    'Decision_tree',
    'RandomForest',
    'GBDT',
    'LR',
    'KNN',
    'SVM_linear',
    'SVM_poly',
    'SVM_rbf'
]

# Iterate through all models to calculate feature importance
for model_name in models:
    print(f'\n=== Processing: {model_name} ===')
    model_path = os.path.join(path, model_name)
    savepath = model_path

    # Load the model
    model = joblib.load(os.path.join(model_path, 'model.pkl'))

    # Set scoring
    if is_classifier(model):
        scoring = ['f1_macro'] if len(np.unique(test_y)) > 2 else ['f1']
    elif is_regressor(model):
        scoring = ['r2', 'neg_mean_absolute_percentage_error', 'neg_mean_squared_error']
    else:
        raise ValueError('The model type is not supported: it must be a classifier or regressor.')

    # Use permutation_importance to calculate the feature importance
    r_multi = {}
    for metric in scoring:
        r = permutation_importance(
            model, test_x, test_y,
            scoring=metric,
            n_repeats=10,
            random_state=0,
            n_jobs=-1
        )
        r_multi[metric] = r

    # Print feature importance
    for metric, r in r_multi.items():
        print(f'\nFeature importance ({model_name}, {metric}):')
        for i in r.importances_mean.argsort()[::-1]:
            if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
                print(f'{feature_names[i]:<12}: {r.importances_mean[i]:.3f} +/- {r.importances_std[i]:.3f}')

    # Visual feature importance
    evaluator = MetricEvaluation(num_classes=num_class, label_names=label_name, save_path=savepath,
                                 model_path=model_path)
    evaluator.plot_features(
        feature_names=feature_names,
        feature_importance=list(r_multi.values())[0].importances_mean
    )

    # Save feature importance
    save_dict = {'feature_names': feature_names}
    for metric, r in r_multi.items():
        save_dict[f'importance_mean_{metric}'] = r.importances_mean
        save_dict[f'importance_std_{metric}'] = r.importances_std

    scipy.io.savemat(os.path.join(savepath, 'feature_importance.mat'), save_dict)
    print(f"Feature importance has been saved to: {savepath}")

print('Done!')
