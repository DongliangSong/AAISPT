# -*- coding: utf-8 -*-
# @Time    : 2023/10/11 13:17
# @Author  : Dongliang

import os

import pandas as pd
import scipy
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from skopt import BayesSearchCV

# Load data
path = r'D:\TrajSeg-Cls\TrajSEG-CLS_V3\CLS\Endocytosis_NEW\5D\Feature_based'
savepath = os.path.join(path, 'KNN')
if not os.path.exists(savepath):
    os.mkdir(savepath)

train = scipy.io.loadmat(os.path.join(path, 'stand_train_feature.mat'))
train_x, train_y = train['data'], train['label'].T

# Check the dataset shape
print("train_x shape:", train_x.shape)
print("train_y shape:", train_y.shape)
if train_y.ndim > 1:
    train_y = train_y.ravel()

# Optimize k
params = {'n_neighbors': [2, 4, 8, 16, 32, 64, 128]}

model = KNeighborsClassifier()
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

# k_range = [2, 4, 8, 16, 32, 64, 128]
# k_f1score = []
#
# for k in k_range:
#     knn = KNeighborsClassifier(n_neighbors=k)
#     scores = cross_val_score(knn, train_x, train_y, cv=10, scoring='f1_macro')
#     k_f1score.append(scores.mean())
#
#     with open(os.path.join(savepath, 'Para_Optimal.txt'), 'a') as f:
#         f.write(str(k) + ' : ' + str(scores) + '\n')
#
# plt.plot(k_range, k_f1score)
# plt.xlabel('Value of K')
# plt.ylabel('F1-SCORE')
# plt.title('Optimal K')
# plt.savefig(os.path.join(savepath, 'Para_Optimal.png'), dpi=600)
print('Done!')
