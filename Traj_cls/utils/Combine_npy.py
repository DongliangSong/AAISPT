# -*- coding: utf-8 -*-
# @Time    : 2024/8/13 16:44
# @Author  : Dongliang

"""
This module converts the features extracted from the sliding window from.npy format to.mat.
"""
import glob
import os

import numpy as np
import scipy

num_class = 5
root = r'D:\TrajSeg-Cls\TrajSEG-CLS_V3\CLS\Var_L_200_new\SNR03\Rolling_feature'
path = os.path.join(root, 'test')

file = glob.glob(os.path.join(path, '*'))
dataset = []
for i in file:
    data = np.load(i)
    data = data.reshape(-1, 53).astype(np.float32)
    dataset.append(data)

for mode in ['train', 'val', 'test']:
    if mode in path:
        feature_name = os.path.join(root, f'{mode}_feature.mat')

num_traces = len(dataset)
num_perclass = num_traces // num_class
label = np.concatenate([i * np.ones((num_perclass, 1), dtype=np.int8) for i in range(num_class)])

scipy.io.savemat(feature_name, {'data': dataset, 'label': label})
