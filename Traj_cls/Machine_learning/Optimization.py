# -*- coding: utf-8 -*-
# @Time    : 2025/4/1 10:14
# @Author  : Dongliang

import os

import numpy as np
import pandas as pd
import scipy
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from skopt import BayesSearchCV
from skopt.space import Integer, Real, Categorical

import AAISPT.Traj_cls.utils.config as c

path = c.path

# Load data
train = scipy.io.loadmat(os.path.join(path, 'stand_train_feature.mat'))
train_x, train_y = train['data'], train['label'].T

# Check the dataset shape
print('train_x shape" ', train_x.shape)
print('train_y shape: ', train_y.shape)
if train_y.ndim > 1:
    train_y = train_y.ravel()

if np.any(np.isnan(train_x)) or np.any(np.isnan(train_y)):
    raise ValueError("The data contains missing values (NaN)")
if np.any(np.isinf(train_x)) or np.any(np.isinf(train_y)):
    raise ValueError("The data contains infinite values (inf)")

# Set scoring
scoring = 'f1' if len(np.unique(train_y)) == 2 else 'f1_macro'

# Define the Hyperparameter Search Space
models_params = {
    'RandomForest': {
        'model': RandomForestClassifier(random_state=25),
        'params': {
            'n_estimators': Integer(50, 200),
            'max_depth': Integer(3, 20),
            'min_samples_split': Integer(2, 10),
            'min_samples_leaf': Integer(1, 5),
        }
    },
    'GBDT': {
        'model': GradientBoostingClassifier(random_state=25),
        'params': {
            'n_estimators': Integer(10, 200),
            'max_depth': Integer(1, 6),
            'min_samples_split': Integer(2, 10),
            'min_samples_leaf': Integer(1, 5),
            'subsample': Real(0.5, 1.0, prior='uniform'),
            'learning_rate': Real(0.01, 0.5, prior='log-uniform')
        }
    },
    'DecisionTree': {
        'model': DecisionTreeClassifier(random_state=25),
        'params': {
            'max_depth': Integer(1, 20),
            'min_samples_leaf': Integer(1, 5),
            'min_samples_split': Integer(2, 10)
        }
    },
    'KNN': {
        'model': KNeighborsClassifier(),
        'params': {'n_neighbors': Integer(2, 64, prior='log-uniform', base=2)}
    },
    'SVM': {
        'model': SVC(decision_function_shape='ovr', probability=True),
        'params': [
            {'kernel': Categorical(['rbf']), 'C': Real(0.001, 100, prior='log-uniform'),
             'gamma': Real(0.001, 10, prior='log-uniform')},
            {'kernel': Categorical(['linear']), 'C': Real(0.001, 100, prior='log-uniform')},
            {'kernel': Categorical(['poly']), 'C': Real(0.001, 100, prior='log-uniform'),
             'degree': Integer(2, 9)}
        ]
    }
}

# Iterate through all models for hyperparameter tuning
for model_name, config in models_params.items():
    print(f"\n=== Processing Model: {model_name} ===")
    savepath = os.path.join(path, model_name)
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    # Hyperparameter tuning
    params = config['params']
    if model_name == 'SVM':
        # SVM requires separate handling for different kernel functions.
        for param in params:
            kernel = param['kernel'].categories[0]

            kernel_savepath = os.path.join(path, f"{model_name}_{kernel.title()}")
            if not os.path.exists(kernel_savepath):
                os.makedirs(kernel_savepath)

            opt = BayesSearchCV(
                estimator=config['model'],
                search_spaces=param,
                n_iter=20,
                scoring=scoring,
                cv=10,
                n_jobs=-1,
                random_state=25
            )
            opt.fit(train_x, train_y)

            # Output
            print(f"Best hyperparameter ({kernel}):", opt.best_params_)
            print(f"Best F1 SCORE ({kernel}):", opt.best_score_)

            # Save
            results = pd.DataFrame(opt.cv_results_)
            try:
                results.to_csv(os.path.join(kernel_savepath, 'bayes_search_results.csv'), index=False)
                print(f"Saved to: {kernel_savepath}")
            except Exception as e:
                print(f"Save Failed: ({kernel}): {e}")

    else:
        opt = BayesSearchCV(
            estimator=config['model'],
            search_spaces=params,
            n_iter=20,
            scoring=scoring,
            cv=10,
            n_jobs=-1,
            random_state=25
        )
        opt.fit(train_x, train_y)

        print(f"Best hyperparameter:", opt.best_params_)
        print(f"Best F1 SCORE:", opt.best_score_)

        results = pd.DataFrame(opt.cv_results_)
        try:
            results.to_csv(os.path.join(savepath, 'bayes_search_results.csv'), index=False)
            print(f"Saved to: {savepath}")
        except Exception as e:
            print(f"Save Failed: {e}")

print('Done!')
