# -*- coding: utf-8 -*-
# @Time    : 2024/5/15 16:09
# @Author  : Dongliang

import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import AAISPT.Traj_cls.utils.config as c


# Perform k-fold cross-validation with a stratified shuffle split strategy on the dataset.
def get_model(model_name, params):
    if model_name == 'Logistic_Regression':
        model = LogisticRegression()
    elif model_name == 'SVM-Linear' or model_name == 'SVM-RBF' or model_name == 'SVM-Polynomial':
        model = SVC(**params)
    elif model_name == 'Decision_Tree':
        model = DecisionTreeClassifier(**params)
    elif model_name == 'KNN':
        model = KNeighborsClassifier(**params)
    elif model_name == 'Random_Forest':
        model = RandomForestClassifier(**params)
    elif model_name == 'GBDT':
        model = GradientBoostingClassifier(**params, random_state=0)
    return model


fontsize = 20

# Model name and hyper-parameters
model_name = {'Decision_Tree': {'criterion': 'gini',
                                'max_depth': 4,
                                'min_samples_split': 2,
                                'min_samples_leaf': 8
                                },

              'GBDT': {'n_estimators': 50,
                       'learning_rate': 0.34,
                       'max_depth': 4,
                       'min_samples_split': 7,
                       'min_samples_leaf': 1,
                       'max_features': 9,
                       'subsample': 0.9
                       },

              'KNN': {'n_neighbors': 32},

              'Logistic_Regression': {'n_jobs': -1,
                                      'random_state': 0,
                                      'penalty': 'l1',
                                      'solver': 'saga',
                                      'multi_class': 'ovr'
                                      },

              'Random_Forest': {'n_estimators': 1,
                                'random_state': 37
                                },

              'SVM-Linear': {'kernel': 'linear',
                             'C': 1.0,
                             'gamma': 'scale',
                             'decision_function_shape': 'ovr'
                             },

              'SVM-RBF': {'kernel': 'rbf',
                          'C': 1.0,
                          'gamma': 'scale',
                          'decision_function_shape': 'ovr'
                          },

              'SVM-Polynomial': {'kernel': 'poly',
                                 'C': 1.0,
                                 'gamma': 'scale',
                                 'degree': 3,
                                 'decision_function_shape': 'ovr'
                                 },
              }

if __name__ == '__main__':
    mode = 'Sim'

    num_class, label_name = c.config(mode=mode, path=path)
    kfold = 5

    path = r'D:\TrajSeg-Cls\TrajSEG-CLS_V2\SNR01\Feature_based'
    savepath = path

    ML_model_name = [
        'Logistic_Regression',
        'Random_Forest',
        'SVM-Linear',
        'SVM-RBF',
        'SVM-Polynomial',
        'Decision_Tree',
        'KNN',
        'GBDT'
    ]

    name = ML_model_name[0]  # 0 represents the index of the model.

    # Load and ShuffleSplit data
    # TODO:根据实际情况进行修改，FP1和FP2是第一
    FP_1 = 1
    FP_2 = 1
    FP_pred = np.vstack([FP_1, FP_2])

    y_before_after = np.hstack([np.zeros(FP_1.shape[0]), np.ones(FP_2.shape[0])])
    print(FP_pred.shape, y_before_after.shape)

    kf = StratifiedShuffleSplit(n_splits=kfold, random_state=0, test_size=0.1)
    kf.get_n_splits(FP_pred)

    # Standardize data
    scaler = StandardScaler()

    # Initialize model
    sensitivity, specificity, accuracy, F1, MacroF1 = [], [], [], [], []

    if name in model_name:
        params = model_name[name]
        conmats = np.zeros((kfold, num_class, num_class))
        i = 0
        for train_index, test_index in kf.split(FP_pred, y_before_after):
            X_train, X_test = FP_pred[train_index], FP_pred[test_index]
            y_train, y_test = y_before_after[train_index], y_before_after[test_index]

            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            X_train[np.isnan(X_train)] = 0
            X_test[np.isnan(X_test)] = 0

            # random oversampling
            from imblearn.over_sampling import RandomOverSampler

            ros = RandomOverSampler(random_state=0)
            X_train, y_train = ros.fit_resample(X_train, y_train)

            clf = get_model(name, params).fit(X_train, y_train)

            y_pred = clf.predict(X_test)

            sen, spec, each_accuracy, total_accuracy, F1Score, MacroF1Score = metric_calculate(y_test, y_pred)

            accuracy.append(total_accuracy)
            sensitivity.append(sen)
            specificity.append(spec)
            F1.append(F1Score)
            MacroF1.append(MacroF1Score)

            conmats[i, :, :] = confusion_matrix(y_test, y_pred, normalize='true')
            i += 1

        print('Accuracy: ', np.round(np.mean(accuracy), 3), '+/-', np.round(np.std(accuracy), 3))
        print('Sensitivity: ', np.round(np.mean(sensitivity), 3), '+/-', np.round(np.std(sensitivity), 3))
        print('Specificity: ', np.round(np.mean(specificity), 3), '+/-', np.round(np.std(specificity), 3))
        print('F1: ', np.round(np.mean(F1), 3), '+/-', np.round(np.std(F1), 3))

        # plot confusion matrix
        confusionmat = np.mean(conmats, axis=0)
        plot_confusion_matrix(confusionmat, label_name, savepath)
        plt.savefig(os.path.join(savepath, 'K-fold-CV-ConMat.pdf'), dpi=600)
        plt.show()
