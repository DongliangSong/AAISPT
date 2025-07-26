# -*- coding: utf-8 -*-
# @Time    : 2025/4/1 14:05
# @Author  : Dongliang

import os

import joblib
import numpy as np
import pandas as pd
import scipy
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import AAISPT.Traj_cls.utils.config as c
from AAISPT.Traj_cls.utils.Metric_evaluation import MetricEvaluation

path = r'D:\TrajSeg-Cls\TrajSEG-CLS_V3\CLS\YanYu_NEW\Feature_based'

# Load configuration
mode = 'Exp'
num_class, label_name = c.config(mode=mode, path=path)

# Load data
train = scipy.io.loadmat(os.path.join(path, 'stand_train_feature.mat'))
train_x, train_y = train['data'], train['label'].T
val = scipy.io.loadmat(os.path.join(path, 'stand_val_feature.mat'))
val_x, val_y = val['data'], val['label'].T

print('train_x shape:', train_x.shape)
print('train_y shape:', train_y.shape)
print('val_x shape:', val_x.shape)
print('val_y shape:', val_y.shape)

if train_y.ndim > 1:
    train_y = train_y.ravel()
if val_y.ndim > 1:
    val_y = val_y.ravel()

if np.any(np.isnan(train_x)) or np.any(np.isnan(train_y)) or np.any(np.isnan(val_x)) or np.any(np.isnan(val_y)):
    raise ValueError('There are missing values in the data (NaN)')
if np.any(np.isinf(train_x)) or np.any(np.isinf(train_y)) or np.any(np.isinf(val_x)) or np.any(np.isinf(val_y)):
    raise ValueError('There are infinite values in the data (inf)')

# Define all models and their parameters
models = {
    'Decision_tree': {
        'model_class': DecisionTreeClassifier,
        'params': ['max_depth', 'min_samples_leaf', 'min_samples_split'],
        'fixed_params': {'criterion': 'gini'}
    },
    'GBDT': {
        'model_class': GradientBoostingClassifier,
        'params': ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'subsample', 'learning_rate'],
        'fixed_params': {'random_state': 0}
    },
    'KNN': {
        'model_class': KNeighborsClassifier,
        'params': ['n_neighbors'],
        'fixed_params': {}
    },
    'LR': {
        'model_class': LogisticRegression,
        'params': [],
        'fixed_params': {'n_jobs': -1, 'penalty': 'l1', 'solver': 'saga'},
    },
    'RandomForest': {
        'model_class': RandomForestClassifier,
        'params': ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf'],
        'fixed_params': {'random_state': 37}
    },
    'SVM_linear': {
        'model_class': SVC,
        'params': ['kernel', 'C'],
        'fixed_params': {'decision_function_shape': 'ovr', 'probability': True}
    },
    'SVM_rbf': {
        'model_class': SVC,
        'params': ['kernel', 'C', 'gamma'],
        'fixed_params': {'decision_function_shape': 'ovr', 'probability': True}
    },
    'SVM_poly': {
        'model_class': SVC,
        'params': ['kernel', 'C', 'degree'],
        'fixed_params': {'decision_function_shape': 'ovr', 'probability': True}
    }
}

for model_name in models:
    print(f'\n=== Processing: {model_name} ===')
    savepath = os.path.join(path, model_name)
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    # Load best parameters
    params = pd.read_csv(os.path.join(savepath, 'bayes_search_results.csv'), sep=',')
    index = params['rank_test_score'].idxmin()

    if isinstance(models[model_name]['params'], list):
        model_params = {param: params.loc[index, f'param_{param}'] for param in models[model_name]['params']}
    else:
        model_params = models[model_name]['params'](params, index)
        model_params = {k: v for k, v in model_params.items() if v is not None}

    # Combine parameters
    model_params.update(models[model_name]['fixed_params'])

    model = models[model_name]['model_class'](**model_params)

    # Train the model using the training sets and check score
    model.fit(train_x, train_y)
    score = model.score(val_x, val_y)
    print(f"{model_name} score: {score:.3f}")

    # Save the model
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
    evaluator = MetricEvaluation(num_classes=num_class, label_names=label_name, save_path=savepath, model_path=savepath)
    evaluator.plot_confusion_matrix(confusionmat=confusionmat, normalize=True)

    # Calculate Metric
    metrics = evaluator.calculate_metrics(val_y, predicted)
    evaluator.write_metric_log(confusion_mat=confusionmat, metrics=metrics)

    if model_name == 'LR':
        scipy.io.savemat(os.path.join(savepath, 'coefficient.mat'), {'coefficient': model.coef_})
        print(f'LR Coefficient: \n{model.coef_}')
        print(f'LR Intercept: \n{model.intercept_}')

print('Done!')
