# -*- coding: utf-8 -*-
# @Time    : 2024/7/3 17:26
# @Author  : Dongliang


import os

import numpy as np
import scipy
from joblib import Parallel, delayed

from AAISPT.Feature_extraction.Fingerprint_feat_gen import GetFeatures


def exp_feature_extaction(idx, data, dt, step_angle, savepath):
    """
    Extract features from experimental data.

    :param idx: Index of the experimental data.
    :param data: Input experimental data.
    :param dt: Time interval between adjacent frames.
    :param step_angle: Check if the input data represents step angles; return true if it does, otherwise false.
    :param savepath: Path for saving error logs.
    :return: Experimental data feature list.
    """

    try:
        dim = data.shape[-1]
        x, y = data[:, 0], data[:, 1]

        if dim == 3 or dim == 5:
            z = data[:, 2]
        else:
            z = None

        if dim == 4:
            po, azi = data[:, 2], data[:, 3]
        elif dim == 5:
            po, azi = data[:, 3], data[:, 4]
        else:
            po, azi = None, None

        feature = GetFeatures(x=x, y=y, z=z, dt=dt, azimuth=azi, polar=po, step_angle=step_angle)
        np.save(os.path.join(savepath, f'{idx:06}.npy'), feature)
        return feature

    except Exception as e:
        with open(os.path.join(savepath, 'error_log.txt'), 'a') as f:
            f.write(f'Error in: {idx}\n')
        return None


if __name__ == '__main__':
    dt = 0.02  # The time interval between neighboring points.
    step_angle = False

    # path = r'D:\TrajSeg-Cls\Exp Demo\19 01072020_2 perfect\6700-8200\update'
    # savepath = path
    # data = scipy.io.loadmat(os.path.join(path, 'seg_traces.mat'))
    # data = data['segments'].squeeze()

    paths = [
        # r'D:\TrajSeg-Cls\TrajSEG-CLS_V3\CLS\QiPan_NEW\small',
        # r'D:\TrajSeg-Cls\Exp Data\QiPan\augdata\small',
        # r'D:\TrajSeg-Cls\Exp Data\YanYu\Results\augdata\small'
        r'D:\TrajSeg-Cls\TrajSEG-CLS_V3\CLS\YanYu_NEW\small'
    ]

    for path in paths:
        savepath = os.path.join(path,'Feature')
        if not os.path.exists(savepath):
            os.mkdir(savepath)

        data = scipy.io.loadmat(os.path.join(path, 'aug_data_resample.mat'))
        X,Y = data['data'].squeeze(), data['label'].squeeze()
        nums = X.shape[0]

        features = Parallel(n_jobs=-1)(
            delayed(exp_feature_extaction)(i, X[i], dt, step_angle, savepath) for i in range(nums))
        #TODO: Remember to modify the step_angle parameter in Get_features according to the actual data.

        # Save features
        scipy.io.savemat(os.path.join(savepath, 'exp_feature.mat'),{'features': features, 'label': Y})
