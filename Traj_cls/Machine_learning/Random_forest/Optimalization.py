# -*- coding: utf-8 -*-
# @Time    : 2023/10/12 20:14
# @Author  : Dongliang

import os

import numpy as np
import pandas as pd
import scipy
from sklearn.ensemble import RandomForestClassifier
from skopt import BayesSearchCV

import AAISPT.Traj_cls.utils.config as c

path = c.path
savepath = os.path.join(path, 'RandomForest')
if not os.path.exists(savepath):
    os.mkdir(savepath)

# Load data
train = scipy.io.loadmat(os.path.join(path, 'stand_train_feature.mat'))
train_x, train_y = train['data'], train['label'].T

# Check the dataset shape
print("train_x shape:", train_x.shape)
print("train_y shape:", train_y.shape)
if train_y.ndim > 1:
    train_y = train_y.ravel()

# Optimize the number of estimators
params = {'n_estimators': np.arange(1, 201, 10)}

model = RandomForestClassifier(random_state=25)
opt = BayesSearchCV(
    estimator=model,
    search_spaces=params,
    n_iter=32,
    scoring='f1_macro',
    cv=10,
    random_state=25,
    n_jobs=-1
)

opt.fit(train_x, train_y)

best_params_ = opt.best_params_
print(opt.best_params_)
print(opt.best_score_)

results = pd.DataFrame(opt.cv_results_)
results.to_csv(os.path.join(savepath, 'bayes_search_results.csv'), index=False)
print(results)

# scores = []
# for i in range(1, 201, 10):
#     model = RandomForestClassifier(n_estimators=i, random_state=25)
#     score = cross_val_score(model, train_x, train_y, cv=10).mean()
#     scores.append(score)
#
#     with open(os.path.join(savepath, 'Para_Optimal.txt'), 'a') as f:
#         f.write(str(i) + " " + str(score) + "\n")
#
# plt.plot(np.linspace(1, 201, 10), scores, color="red", label="n_estimators")
# plt.legend()
# plt.title("n_estimators vs CV score")
# plt.savefig(os.path.join(savepath, 'Para_Optimal.png'), dpi=600)
#
# print("max_scores : {}, index is {}".format(max(scores), scores.index(max(scores))))
print('Done!')
