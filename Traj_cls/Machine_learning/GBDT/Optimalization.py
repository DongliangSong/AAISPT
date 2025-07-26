# -*- coding: utf-8 -*-
# @Time    : 2023/10/10 22:32
# @Author  : Dongliang


import os

import numpy as np
import pandas as pd
import scipy
from sklearn.ensemble import GradientBoostingClassifier
from skopt import BayesSearchCV

import AAISPT.Traj_cls.utils.config as c

path = c.path
savepath = os.path.join(path, 'GBDT')
if not os.path.exists(savepath):
    os.makedirs(savepath)

# Load data
train = scipy.io.loadmat(os.path.join(path, 'stand_train_feature.mat'))
train_x, train_y = train['data'], train['label'].T

print("train_x shape:", train_x.shape)
print("train_y shape:", train_y.shape)
if train_y.ndim > 1:
    train_y = train_y.ravel()

# Bayes method for parameter tuning, the order of parameter tuning is:
# n_estimators
# max_depth、min_samples_split
# min_samples_leaf、min_samples_split
# max_features、subsample
# learning_rate
params = {
    'n_estimators': np.arange(10, 60, 5),
    'max_depth': np.arange(1, 6),
    'min_samples_split': np.arange(2, 10),
    'min_samples_leaf': np.arange(1, 5),
    'max_features': np.arange(1, 11),
    'subsample': np.arange(0.5, 1, 0.1),
    'learning_rate': np.arange(0.01, 0.5, 0.01)
}

model = GradientBoostingClassifier(random_state=25)

opt = BayesSearchCV(
    estimator=model,
    search_spaces=params,
    n_iter=32,
    scoring='f1_macro',
    cv=10,
    random_state=25,
    n_jobs=-1
)

# executes bayesian optimization
opt.fit(train_x, train_y)

best_params_ = opt.best_params_
print('Best hyperparameter: ', opt.best_params_)
print('Best F1 SCORE: ', opt.best_score_)

results = pd.DataFrame(opt.cv_results_)
results.to_csv(os.path.join(savepath, 'bayes_search_results.csv'), index=False)
print(results)
print('Done!')
