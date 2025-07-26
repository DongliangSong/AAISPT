# -*- coding: utf-8 -*-
# @Time    : 2024/7/11 18:46
# @Author  : Dongliang


import glob
import os

import numpy as np
import pandas as pd
import scipy

path = glob.glob('D:\TrajSeg-Cls\Exp Data\YanYu\Results\data\*')
savepath = r'D:\TrajSeg-Cls\Exp Data\YanYu\Results\augdata\small'
if not os.path.exists(savepath):
    os.mkdir(savepath)

label_mapping = {
    'Circling': 0,
    'confined_circling': 1,
    'rocking': 2
}

dataset = []
label = []

for i in path:
    dirname, file = os.path.split(i)
    if file not in label_mapping.keys():
        continue

    filename = glob.glob(os.path.join(i, '*.xlsx'))

    for key, value in label_mapping.items():
        if key in i:
            label.extend([value] * len(filename))

    for j in filename:
        data = pd.read_excel(j).values

        if 'Single_track' in j:
            trace = data[:, 1:3]
        else:
            if (data[:, 6] == 90).all() or (data[:, 8] == 90).all():
                trace = data[:, 1:3]
            else:
                xy = data[:, 1:3]
                ap = data[:, 7:9]
                trace = np.concatenate((xy, ap), axis=1)
        dataset.append(trace)

scipy.io.savemat(os.path.join(savepath, 'data.mat'), {'data': dataset, 'label': label})
