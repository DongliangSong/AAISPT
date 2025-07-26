# -*- coding: utf-8 -*-
# @Time    : 2025/3/31 21:07
# @Author  : Dongliang

import os

import numpy as np
import scipy

path = r'D:\TrajSeg-Cls\TrajSEG-CLS_V3\CLS\Endocytosis_NEW\5D'

feature = scipy.io.loadmat(os.path.join(path, 'feature.mat'))
X = feature['features']

train = scipy.io.loadmat(os.path.join(path, 'Feature_based/train_feature.mat'))['data']
val = scipy.io.loadmat(os.path.join(path, 'Feature_based/val_feature.mat'))['data']
test = scipy.io.loadmat(os.path.join(path, 'Feature_based/test_feature.mat'))['data']

# 获取训练集和测试集的索引
train_idx = np.where(np.isin(X, train).all(axis=1))[0]
val_idx = np.where(np.isin(X, val).all(axis=1))[0]
test_idx = np.where(np.isin(X, test).all(axis=1))[0]

scipy.io.savemat(os.path.join(path, 'Feature_based/sample_index.mat'),
                 {'train_index': train_idx,
                  'val_index': val_idx,
                  'test_index': test_idx})
