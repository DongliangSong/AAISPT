# -*- coding: utf-8 -*-
# @Time    : 2024/3/18 20:07
# @Author  : Dongliang


import os

import numpy as np
import scipy
from joblib import Parallel, delayed

from AAISPT.Feature_extraction.Rolling_extract_features import Rolling_extract_feature, process_trajectory


def Rolling_extract(traj, dt, window_ratio=None):
    """
    Extraction of multidimensional features in trajectories using the rolling window method.

    :param traj: Multidimensional trajectory dataset.
    :param dt: The time interval between neighboring points.
    :param window_ratio: Ratio of sliding window size to track length.
    :return: List of features extracted from the trajectory dataset.
    """

    return Rolling_extract_feature(traj=traj, dt=dt, step_angle=False, window_ratio=window_ratio)


def process_traj(idx, data, dt, window_ratio, savepath):
    """
    Extracting multiple features from trajectories using the rolling window method.

    :param idx:
    :param data:
    :param dt: The time interval between neighboring points.
    :param window_ratio: Ratio of sliding window size to track length.
    :param savepath: The path to save the extracted features.
    """
    return process_trajectory(idx, data, dt, window_ratio, step_angle=False, savepath=savepath)


if __name__ == '__main__':
    window_ratio = None
    dt = 0.02   # The time interval between two adjacent frames.

    paths = [
        r'.\data\CLS\Andi\2D',
        r'.\data\CLS\Andi\3D'
    ]

    for path in paths:
        for mode in ['train', 'val', 'test']:
            savename = os.path.join(path, 'Feature_based')
            savepath = os.path.join(savename, mode)
            if not os.path.exists(savepath):
                os.makedirs(savepath)

            data = scipy.io.loadmat(os.path.join(path, f'raw{mode}.mat'))
            X, Y = data['data'].squeeze(), data['label'].squeeze()

            nums = X.shape[0]
            features = Parallel(n_jobs=-1)(
                delayed(process_traj)(i, X[i], dt, window_ratio, savepath) for i in range(nums))

            # Remove the features of abnormal trajectories.
            idx = []
            error_log_path = os.path.join(savepath, 'error_log.txt')
            if os.path.exists(error_log_path):
                with open(error_log_path) as f:
                    for line in f:
                        idx.append(int(line.split(':')[-1]))

                id = np.unique(np.array(idx))
                Y = np.delete(Y, id, axis=0)
                del features[int(id)]

            features = [i.squeeze() for i in features]
            # Save features
            scipy.io.savemat(os.path.join(savename, f'Roll_{mode}_feature.mat'), {'data': features, 'label': Y})
