# -*- coding: utf-8 -*-
# @Time    : 2024/10/22 10:14
# @Author  : Dongliang


import os

import numpy as np
import scipy
from sklearn.impute import SimpleImputer


def feature_denoise(X):
    """
    Outliers in the features were removed by quartile spacing and replaced using mean values.

    :param X: Input features.
    :return:
    """
    # Outlier handling: replaces the outlier with NaN.
    for j in range(X.shape[1]):
        Q1 = np.percentile(X[:, j], 25, axis=0)
        Q3 = np.percentile(X[:, j], 75, axis=0)
        IQR = Q3 - Q1
        lower_limit = Q1 - 1.5 * IQR
        upper_limit = Q3 + 1.5 * IQR

        # lower_limit, upper_limit = np.percentile(a=a[:, j], q=[10, 90])
        X[:, j] = np.where((X[:, j] < lower_limit) | (X[:, j] > upper_limit), np.nan, X[:, j])

    # Use the mean to populate the NaN value.
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    return X


if __name__ == '__main__':
    path = r'D:\TrajSeg-Cls\TrajSEG-CLS_V3\CLS\Andi\3D\Feature_based'
    savepath = path

    for mode in ['train', 'val', 'test']:
        data = scipy.io.loadmat(os.path.join(path, f'Roll_{mode}_feature.mat'))
        X, Y = data['data'].squeeze(), data['label']
        feature = feature_denoise(X)
        scipy.io.savemat(os.path.join(savepath, f'denoise_{mode}_feature.mat'), {'data': feature, 'label': Y})

    # data = scipy.io.loadmat(os.path.join(path, 'Rolling_feature_new.mat'))
    # X, Y = data['data'].squeeze(), data['label'].squeeze()
    # X = feature_denoise(X)
    # scipy.io.savemat(os.path.join(savepath, 'denoise_feature.mat'), {'data': X, 'label': Y})
