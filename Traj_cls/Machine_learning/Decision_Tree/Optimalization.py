# -*- coding: utf-8 -*-
# @Time    : 2023/10/10 20:45
# @Author  : Dongliang


import os

import pandas as pd
import scipy
from sklearn.tree import DecisionTreeClassifier
from skopt import BayesSearchCV
from skopt.space import Integer

import AAISPT.Traj_cls.utils.config as c

path = c.path
savepath = os.path.join(path, 'Decision_tree')
if not os.path.exists(savepath):
    os.mkdir(savepath)

# Load data
train = scipy.io.loadmat(os.path.join(path, 'stand_train_feature.mat'))
train_x, train_y = train['data'], train['label'].T

# Check the dataset shape
print('train_x shape"', train_x.shape)
print('train_y shape:', train_y.shape)
if train_y.ndim > 1:
    train_y = train_y.ravel()

# Define the Hyperparameter Search Space
params = {
    'max_depth': Integer(1, 10),
    'min_samples_leaf': Integer(1, 5),
    'min_samples_split': Integer(2, 10)
}

# Define the model
model = DecisionTreeClassifier(random_state=25)

opt = BayesSearchCV(
    estimator=model,
    search_spaces=params,
    n_iter=32,
    scoring='f1_macro',
    cv=10,
    n_jobs=-1,
    random_state=25
)

opt.fit(train_x, train_y)

best_params_ = opt.best_params_
print('Best hyperparameter: ', opt.best_params_)
print('Best F1 SCORE: ', opt.best_score_)

# Save results
results = pd.DataFrame(opt.cv_results_)
results.to_csv(os.path.join(savepath, 'bayes_search_results.csv'), index=False)
print(results)
print('Done!')
