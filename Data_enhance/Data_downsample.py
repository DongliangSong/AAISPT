# -*- coding: utf-8 -*-
# @Time    : 2024/7/9 10:46
# @Author  : Dongliang

import os
import numpy as np
import scipy
from scipy.signal import resample


def downsample(data, threshold_length, downsample_factor):
    """
    Downsample data arrays if their length exceeds a specified threshold.

    Args:
        data: Input data, either a 2D numpy array
            where each row is a 1D array, or a list of 1D numpy arrays.
        threshold_length: The length threshold above which downsampling is applied.
        downsample_factor: The factor by which to downsample the data (e.g., if factor=2,
            the length is halved).

    Returns:
        A list of downsampled (or original) 1D arrays.
    """

    # Input validation
    if threshold_length < 1:
        raise ValueError("threshold_length must be a positive integer")
    if downsample_factor < 1:
        raise ValueError("downsample_factor must be a positive integer")

    # Convert input to list of arrays if it's a 2D numpy array
    if isinstance(data, np.ndarray):
        if data.ndim != 2:
            raise ValueError("If data is a numpy array, it must be 2D (rows of 1D arrays)")
        data_list = [data[i, :] for i in range(data.shape[0])]
    elif isinstance(data, list):
        data_list = data
    else:
        raise TypeError("data must be a 2D numpy array or a list of 1D arrays")

    # Process each array
    downsampled_data = []
    for array in data_list:
        # Ensure array is a 1D numpy array
        array = np.asarray(array, dtype=float)
        array_length = len(array)
        if array_length > threshold_length:
            new_length = array_length // downsample_factor
            array = resample(array, new_length)

        downsampled_data.append(array)

    return downsampled_data


if __name__ == '__main__':
    factor = 10
    ther_len = 2000
    path = r'D:\TrajSeg-Cls\TrajSEG-CLS_V3\CLS\YanYu_NEW\small'
    savepath = path

    if 'QiPan' in path or 'YanYu' in path:
        aug_data = scipy.io.loadmat(os.path.join(path, 'aug_data.mat'))
        data, label = aug_data['data'].squeeze(), aug_data['label'].squeeze()
        final_x = downsample(data=data, threshold_length=ther_len, downsample_factor=factor)
        scipy.io.savemat(os.path.join(savepath, 'aug_data_resample.mat'), {'data': final_x, 'label': label})

    elif 'endocytosis' in path.lower():
        path = os.path.join(path, 'aug_5D.mat')
        aug_data = scipy.io.loadmat(path)
        data, label = aug_data['data'].squeeze(), aug_data['label'].squeeze()
        final_x = downsample(data=data, threshold_length=ther_len, downsample_factor=factor)
        name = os.path.split(path)[-1]
        scipy.io.savemat(os.path.join(savepath, f'resample_{name}'), {'data': final_x, 'label': label})
