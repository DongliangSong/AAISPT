# -*- coding: utf-8 -*-
# @Time    : 2025/1/11 17:40
# @Author  : Dongliang

from AAISPT.Feature_extraction.Rolling_extract_features import *

path = r'D:\TrajSeg-Cls\endoysis\Aug_SPT_FLU'
savepath = os.path.join(path, r'Roll_Feature\features')

step_angle = False
window_ratio = 0.1
dt = 0.02

data = scipy.io.loadmat(os.path.join(path, 'aug_data.mat'))
X, Y = data['data'].squeeze(), data['label'].squeeze()

nums = X.shape[0]
features = Parallel(n_jobs=-1)(
    delayed(process_trajectory)(i, X[i], dt, window_ratio, step_angle,savepath) for i in range(nums))
# TODO: Remember to modify the step_angle parameter according to the actual data.

