# -*- coding: utf-8 -*-
# @Time    : 2024/7/1 23:07
# @Author  : Dongliang

import os
import random

import numpy as np
import scipy

path = r'D:\TrajSeg-Cls\TrajSEG-CLS_V3\CLS\Var_L_500\SNR05'
savepath = path
num_class = 5

# Define a subset of possible dimensions
if num_class == 5:
    subsets = [
        [0, 1, 2, 3, 4],
        [0, 1, 3, 4],
    ]

elif num_class == 3:
    subsets = [
        [0, 1, 2],
        [0, 1]
    ]

for mode in ['train', 'val', 'test']:
    data = scipy.io.loadmat(os.path.join(path, f'addnoise_{mode}.mat'))['data'].squeeze()

    # Select a random subset for each sample
    reduced_data = []
    nums = data.shape[0]
    for i in range(nums):
        sample = data[i]
        subset = random.choice(subsets)
        reduced_sample = sample[:, subset]
        reduced_data.append(reduced_sample)

    each_class = nums // num_class
    label = np.vstack([i * np.ones((each_class, 1)) for i in range(num_class)])

    scipy.io.savemat(os.path.join(savepath, f'varLD_{mode}.mat'), {'data': reduced_data, 'label': label})
