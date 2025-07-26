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

    """
    # Outlier handling: replaces the outlier with NaN.
    for i in range(X.shape[0]):
        a = X[i]
        for j in range(a.shape[1]):
            Q1 = np.percentile(a[:, j], 25, axis=0)
            Q3 = np.percentile(a[:, j], 75, axis=0)
            IQR = Q3 - Q1
            lower_limit = Q1 - 1.5 * IQR
            upper_limit = Q3 + 1.5 * IQR

            # lower_limit, upper_limit = np.percentile(a=a[:, j], q=[10, 90])
            a[:, j] = np.where((a[:, j] < lower_limit) | (a[:, j] > upper_limit), np.nan, a[:, j])

        # Use the mean to populate the NaN value.
        imputer = SimpleImputer(strategy='mean')
        a = imputer.fit_transform(a)
        X[i] = a
    return X


if __name__ == '__main__':
    paths = [
        r'.\data\CLS\Andi\2D',
        r'.\data\CLS\Andi\3D'
    ]

    for path in paths:
        for mode in ['train', 'val', 'test']:
            savename = os.path.join(path, 'Feature_based')
            # savepath = os.path.join(savename, mode)
            # if not os.path.exists(savepath):
            #     os.makedirs(savepath)

            data = scipy.io.loadmat(os.path.join(savename, f'Roll_{mode}_feature.mat'))
            X, Y = data['data'].squeeze(), data['label'].squeeze()
            X = feature_denoise(X)

            # Save features
            scipy.io.savemat(os.path.join(savename, f'denoise_{mode}_feature.mat'), {'data': X, 'label': Y})
