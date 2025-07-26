# -*- coding: utf-8 -*-
# @Time    : 2024/7/11 15:30
# @Author  : Dongliang

# Since YanYu and QiPan's datasets are predominantly 4-dimensional and 2-dimensional,
# their dimensionality is fixed to exclude a very small number of non-standard trajectories.

import os

import scipy

paths = [
    r'D:\TrajSeg-Cls\TrajSEG-CLS_V3\CLS\YanYu_NEW\small',
    r'D:\TrajSeg-Cls\Exp Data\QiPan\augdata\small',
]

for path in paths:
    x = scipy.io.loadmat(os.path.join(path, 'aug_data_resample.mat'))
    data, label = x['data'].squeeze(), x['label'].squeeze()

    if 'QiPan' in path:
        dim = 2
    elif 'YanYu' in path:
        dim = 4

    result = []
    labels = []
    for i in range(data.shape[0]):
        if data[i].shape[1] == dim:
            result.append(data[i])
            labels.append(label[i])

    scipy.io.savemat(os.path.join(path, 'aug_data_resample.mat'), {'data': result, 'label': labels})
