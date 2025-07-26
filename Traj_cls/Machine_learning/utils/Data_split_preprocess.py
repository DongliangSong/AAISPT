# -*- coding: utf-8 -*-
# @Time    : 2024/8/23 19:30
# @Author  : Dongliang

import json
import os

import numpy as np
import scipy
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

paths = [
    r'D:\TrajSeg-Cls\TrajSEG-CLS_V3\CLS\Endocytosis_NEW\5D\Feature_based V1',
    r'D:\TrajSeg-Cls\TrajSEG-CLS_V3\CLS\YanYu_NEW\Feature_based V1',
    r'D:\TrajSeg-Cls\TrajSEG-CLS_V3\CLS\QiPan_NEW\Feature_based V1'
]

for path in paths:
    # savepath = os.path.join(path, 'Feature_based')
    savepath = path
    if not os.path.exists(savepath):
        os.mkdir(savepath)

    dataset = scipy.io.loadmat(os.path.join(path, 'feature.mat'))
    data, label = dataset['features'].squeeze(), dataset['label'].squeeze()

    num = data.shape[1]
    for i in range(num):
        threshold = np.percentile(data[:, i], 95)
        ind = data[:, i] > threshold
        data[ind, i] = np.nan

    data[(data == np.inf) | (data == -np.inf)] = np.nan

    # Replaces the nan value with the mean value of each column.
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    data = imputer.fit_transform(data)

    scipy.io.savemat(os.path.join(savepath, 'feature_label.mat'), {'data': data, 'label': label})

    # Split dataset
    X_train_val, X_test, y_train_val, y_test = train_test_split(data, label, test_size=0.1, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=1 / 9, random_state=42)

    # Obtain the indices of the training set and test set
    train_idx = np.where(np.isin(data, X_train).all(axis=1))[0]
    val_idx = np.where(np.isin(data, X_val).all(axis=1))[0]
    test_idx = np.where(np.isin(data, X_test).all(axis=1))[0]

    scipy.io.savemat(os.path.join(savepath, 'sample_index.mat'), {'train_index': train_idx,
                                                                  'val_index': val_idx,
                                                                  'test_index': test_idx})
    scipy.io.savemat(os.path.join(savepath, 'train_feature.mat'), {'data': X_train, 'label': y_train})
    scipy.io.savemat(os.path.join(savepath, 'val_feature.mat'), {'data': X_val, 'label': y_val})
    scipy.io.savemat(os.path.join(savepath, 'test_feature.mat'), {'data': X_test, 'label': y_test})
    print('Split Finished!')

    # Data preprocessing
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    train_mean, train_std = scaler.mean_, np.sqrt(scaler.var_)

    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Save standardization dataset
    data_param = {'train_mean': train_mean.tolist(), 'train_std': train_std.tolist()}
    with open(os.path.join(savepath, 'train_param.json'), 'w') as f:
        json.dump(data_param, f, indent=4)

    # Save preprocessed features
    scipy.io.savemat(os.path.join(savepath, 'stand_train_feature.mat'), {'data': X_train, 'label': y_train})
    scipy.io.savemat(os.path.join(savepath, 'stand_val_feature.mat'), {'data': X_val, 'label': y_val})
    scipy.io.savemat(os.path.join(savepath, 'stand_test_feature.mat'), {'data': X_test, 'label': y_test})
    print('Standardization Finished!')
