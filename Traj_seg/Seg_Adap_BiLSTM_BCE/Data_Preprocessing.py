# -*- coding: utf-8 -*-
# @Time    : 2024/7/2 18:51
# @Author  : Dongliang

import os

import numpy as np
import scipy


def label_convert(label, seq_len, FWHM):
    """
    Converts the change point to a Gaussian distribution with a specific full width at half maximum (FWHM).

    :param label: The change point of each track.
    :param seq_len: The length of the trajectory.
    :param FWHM: The full width at half maximum of the Gaussian distribution.
    :return: The transformed labelling matrix.
    """

    def gaussian(x, mu, sigma):
        return np.exp(- (x - mu) ** 2 / (2 * sigma ** 2))

    # Set parameters
    sigma = FWHM / (2 * np.sqrt(2 * np.log(2)))
    x = np.linspace(0, seq_len - 1, seq_len)
    x = x[np.newaxis, :]  # Shape: (1, seq_len)
    labels = np.array(label)[:, np.newaxis]  # Shape: (num_labels, 1)
    y = gaussian(x, labels, sigma)
    return y


def data_standard(X, use_xyz_diff):
    """
    Standardization of trajectory data.  (X-mean)/std

    :param X: Input data in the shape of (number of samples, trajectory length, trajectory dimension).
    :param use_xyz_diff: If True, use xyz_diff function instead of np.diff.
    :return: Standardized data with shape (number of samples, trajectory length, trajectory dimension).
    """
    thr = 1e-10
    N, trace_len, dimension = X.shape

    # Calculate the difference between adjacent time steps.
    if use_xyz_diff:
        r = xyz_diff(X)
    elif use_xyz_diff == False:
        r = np.diff(X, axis=1)
    elif use_xyz_diff == []:
        r = X

    # Calculate mean and std along axis 1, keeping dims
    mean_r = np.mean(r, axis=1, keepdims=True)
    std_r = np.std(r, axis=1, keepdims=True)
    std_r[std_r < thr] = 1

    # Standardization
    stand_r = (r - mean_r) / std_r
    # zero_pad = np.zeros((N, 1, dimension), dtype=np.float32)
    # stand_r = np.concatenate((zero_pad, stand_r), axis=1)

    return stand_r


def preprocess_tracks(X, use_xyz_diff):
    """
    Preprocessing of trajectory datasets (Difference and Normalization).

    :param X: Raw trajectory dataset.
    :param use_xyz_diff: If xyz is already differential data, input false; otherwise, input true.
    """

    dim = X.shape[-1]
    if use_xyz_diff:
        r = xyz_diff(X)
    elif use_xyz_diff == False:
        r = np.diff(X, axis=1)
    elif use_xyz_diff == []:
        r = X

    if dim == 2:
        dx, dy = r[:, :, 0], r[:, :, 1]
        num = dx.shape[1]
        meanstep = np.expand_dims(np.sum(((dx ** 2) + (dy ** 2)) ** 0.5, axis=1) / num, axis=-1)
        return np.dstack((dx / meanstep, dy / meanstep))

    elif dim == 3:
        dx, dy, dz = r[:, :, 0], r[:, :, 1], r[:, :, 2]
        num = dx.shape[1]
        meanstep = np.expand_dims(np.sum(((dx ** 2) + (dy ** 2) + (dz ** 2)) ** 0.5, axis=1) / num, axis=-1)
        return np.dstack((dx / meanstep, dy / meanstep, dz / meanstep))

    elif dim == 4:
        dx, dy, dp, da = r[:, :, 0], r[:, :, 1], r[:, :, 2], r[:, :, 3]
        num = dx.shape[1]
        meanstep = np.expand_dims(np.sum(((dx ** 2) + (dy ** 2)) ** 0.5, axis=1) / num, axis=-1)

        mean_dp = np.expand_dims(np.mean(np.abs(dp), axis=1), axis=-1)
        mean_da = np.expand_dims(np.mean(np.abs(da), axis=1), axis=-1)
        return np.dstack((dx / meanstep, dy / meanstep, dp / mean_dp, da / mean_da))

    elif dim == 5:
        dx, dy, dz, dp, da = r[:, :, 0], r[:, :, 1], r[:, :, 2], r[:, :, 3], r[:, :, 4]
        num = dx.shape[1]
        meanstep = np.expand_dims(np.sum(((dx ** 2) + (dy ** 2) + (dz ** 2)) ** 0.5, axis=1) / num, axis=-1)

        mean_dp = np.expand_dims(np.mean(np.abs(dp), axis=1), axis=-1)
        mean_da = np.expand_dims(np.mean(np.abs(da), axis=1), axis=-1)
        return np.dstack((dx / meanstep, dy / meanstep, dz / meanstep, dp / mean_dp, da / mean_da))


def xyz_diff(data):
    """
    Calculate the difference between adjacent time steps for xyz or xy.

    :param data: Input data in the shape of (number of samples, trajectory length, trajectory dimension).
    :return: Differential data with shape (number of samples, trajectory length, trajectory dimension).
    """
    if data.shape[-1] == 5 or data.shape[-1] == 4:
        # For 4 and 5-D dataset
        data_diff = np.diff(data[:, :, :-2], axis=1)
        new = np.concatenate((data_diff, data[:, :data.shape[1] - 1, -2:]), axis=2)
    else:
        # For 2 and 3-D dataset
        new = np.diff(data, axis=1)

    return new


def range_scaler(data):
    """
    Min-max normalization (scaling data range).

    :param data: Input data with shape (Number of samples, trajectory length, trajectory dimension)
    :return: Normalized data with the same shape.
    """
    min_vals = np.min(data, axis=1, keepdims=True)
    max_vals = np.max(data, axis=1, keepdims=True)

    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1

    norm = (data - min_vals) / range_vals
    return norm


if __name__ == '__main__':
    path = r'..\data\5D'
    dim = next((int(d[0]) for d in ['2D', '3D', '4D', '5D'] if d in path), None)
    train = scipy.io.loadmat(os.path.join(path, 'train.mat'))
    train_data, train_label = train['trainset'], train['trainlabel']

    val = scipy.io.loadmat(os.path.join(path, 'val.mat'))
    val_data, val_label = val['valset'], val['vallabel']

    test = scipy.io.loadmat(os.path.join(path, 'test.mat'))
    test_data, test_label = test['testset'], test['testlabel']

    norm_train = preprocess_tracks(train_data, use_xyz_diff=True)
    norm_val = preprocess_tracks(val_data, use_xyz_diff=True)
    norm_test = preprocess_tracks(test_data, use_xyz_diff=True)

    # Save range_scaler dataset
    scipy.io.savemat(os.path.join(path, 'norm_train.mat'), {'data': norm_train, 'label': train_label})
    scipy.io.savemat(os.path.join(path, 'norm_val.mat'), {'data': norm_val, 'label': val_label})
    scipy.io.savemat(os.path.join(path, 'norm_test.mat'), {'data': norm_test, 'label': test_label})
