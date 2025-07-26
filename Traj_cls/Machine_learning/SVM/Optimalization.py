# -*- coding: utf-8 -*-
# @Time    : 2023/10/12 21:09
# @Author  : Dongliang

import os

import pandas as pd
import scipy
from sklearn.svm import SVC
from skopt import BayesSearchCV

import AAISPT.Traj_cls.utils.config as c

path = c.path
savepath = os.path.join(path, 'SVM')
if not os.path.exists(savepath):
    os.mkdir(savepath)

# Load data
train = scipy.io.loadmat(os.path.join(path, 'stand_train_feature.mat'))
train_x, train_y = train['data'], train['label'].T

# Check the dataset shape
print('train_x shape:', train_x.shape)
print('train_y shape:', train_y.shape)
if train_y.ndim > 1:
    train_y = train_y.ravel()

# Set parameters
params = [
    {'kernel': ['rbf'], 'C': [0.001, 0.01, 0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1, 10, 100]},
    {'kernel': ['linear'], 'C': [0.001, 0.01, 0.1, 1, 10, 100]},
    {'kernel': ['poly'], 'C': [0.001, 0.01, 0.1, 1, 10, 100], 'degree': [2, 3, 4, 5, 6, 7, 8, 9]}
]

for i in params:
    search = BayesSearchCV(
        estimator=SVC(decision_function_shape='ovr'),
        search_spaces=i,
        scoring='f1_macro',
        cv=10,
        n_jobs=-1,
        random_state=25
    )

    search.fit(train_x, train_y)
    search.score(train_x, train_y)
    print('Best hyperparameter: ', search.best_params_)
    print('Best F1 SCORE: ', search.best_score_)

    results = pd.DataFrame(search.cv_results_)
    p = i['kernel']
    if 'rbf' == i['kernel'][0]:
        file = os.path.join(savepath, 'RBF')

    elif 'poly' == i['kernel'][0]:
        file = os.path.join(savepath, 'Poly')

    elif 'linear' == i['kernel'][0]:
        file = os.path.join(savepath, 'Linear')

    if not os.path.exists(file):
        os.mkdir(file)

    print('=' * 50)
    filename = 'bayes_search_results.csv'
    results.to_csv(os.path.join(file, filename), index=False)

print('Done!')
