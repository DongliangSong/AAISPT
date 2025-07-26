# -*- coding: utf-8 -*-
# @Time    : 2024/7/11 19:04
# @Author  : Dongliang

import glob
import os

import pandas as pd
import scipy

path = glob.glob('D:\TrajSeg-Cls\Exp Data\QiPan\Fig2\*')

savepath = r'D:\TrajSeg-Cls\Exp Data\QiPan\augdata\small'
if not os.path.exists(savepath):
    os.mkdir(savepath)

label_mapping = {
    'S1_Tracks': 0,
    'S1-S2_tracks': 1,
    'S2_Tracks': 2
}

dataset = []
label = []

for i in path:
    dirname, file = os.path.split(i)
    if file not in label_mapping.keys():
        continue

    filename = glob.glob(os.path.join(i, 'Tracks/*.xlsx'))

    for key, value in label_mapping.items():
        if key in i:
            label.extend([value] * len(filename))

    for j in filename:
        data = pd.read_excel(j).values
        trace = data[:, 1:]
        dataset.append(trace)


scipy.io.savemat(os.path.join(savepath, 'QiPan_small.mat'), {'data': dataset, 'label': label})
