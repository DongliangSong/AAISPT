# -*- coding: utf-8 -*-
# @Time    : 2024/8/14 10:16
# @Author  : Dongliang

import os
import re

import numpy as np
import scipy

path = r'D:\TrajSeg-Cls\TrajSEG-CLS_V3\SEG\Third batch\2345扩增后'
file = os.path.join(path, 'train.log')
with open(file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

logs = []
for line in lines:
    if 'Train_loss' in line:
        epoch = re.search(r'Epoch\s*:\s*(\d+)', line)
        lr = re.search(r'lr\s*:\s*([0-9.]+)', line)
        train_loss = re.search(r'Train_loss\s*:\s*([0-9.]+)', line)
        val_loss = re.search(r'Val_loss\s*:\s*([0-9.]+)', line)

        epoch_value = int(epoch.group(1))
        lr_value = float(lr.group(1))
        train_loss_value = float(train_loss.group(1))
        val_loss_value = float(val_loss.group(1))

        line = np.stack((epoch_value, lr_value, train_loss_value, val_loss_value))
        logs.append(line)

scipy.io.savemat(os.path.join(path, 'logs.mat'), {'logs': logs})
