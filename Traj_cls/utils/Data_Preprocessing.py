# -*- coding: utf-8 -*-
# @Time    : 2024/3/22 17:01
# @Author  : Dongliang

import json
import os

import numpy as np
import scipy


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
    std_r[std_r < thr] = 1  # Avoid division by zero or very small values

    # Standardization
    stand_r = (r - mean_r) / std_r
    zero_pad = np.zeros((N, 1, dimension), dtype=np.float32)
    stand_r = np.concatenate((zero_pad, stand_r), axis=1)

    return stand_r


def xyz_diff(data):
    """
    Calculate the difference between adjacent time steps for xyz or xy.

    :param data: Input data in the shape of (number of samples, trajectory length, trajectory dimension).
    :return: Differential data with shape (number of samples, trajectory length, trajectory dimension).
    """
    dim = data.shape[-1]
    if dim == 5 or dim == 4:
        data_diff = np.diff(data[:, :, :-2], axis=1)
        new = np.concatenate((data_diff, data[:, :data.shape[1] - 1, -2:]), axis=2)
    else:
        data_diff = np.diff(data[:, :, :2], axis=1)
        new = np.concatenate((data_diff, data[:, :data.shape[1] - 1, 2:]), axis=2)

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

def Standardization(data):
    """
    Feature-wise Standardization for train and val set
    :param data: Input data in the shape of (number of samples, trajectory length, trajectory dimension).
    :return: Standardized data with shape (number of samples, trajectory length, trajectory dimension).
    """
    thr = 1e-10
    m = np.mean(data, axis=(0, 1), keepdims=True)
    s = np.std(data, axis=(0, 1), keepdims=True)
    new = (data - m) / np.where(s > thr, s, 1)
    return new, m, s


def data_stand(data, mean_train, std_train):
    """
    Feature-wise Standardization for unknown dataset
    :param data: Input data in the shape of (number of samples, trajectory length, trajectory dimension).
    :param mean_train: The mean of the training set.
    :param std_train: Standard deviation of the training set.
    :return: Standardized data with shape (number of samples, trajectory length, trajectory dimension).
    """
    thr = 1e-10
    mean_train = np.expand_dims(mean_train, axis=(0, 1))
    std_train = np.expand_dims(std_train, axis=(0, 1))
    new = (data - mean_train) / np.where(std_train > thr, std_train, 1)
    return new


if __name__ == '__main__':
    path = r'D:\TrajSeg-Cls\TrajSEG-CLS_V3\SEG\SNR05'

    train = scipy.io.loadmat(os.path.join(path, 'train.mat'))
    train_data, train_label = train['trainset'], train['trainlabel']

    val = scipy.io.loadmat(os.path.join(path, 'val.mat'))
    val_data, val_label = val['valset'], val['vallabel']

    test = scipy.io.loadmat(os.path.join(path, 'test.mat'))
    test_data, test_label = test['testset'], test['testlabel']

    for data_preprocess in ['Normalization', 'Standardization']:
        if data_preprocess == 'Normalization':
            diff_train = xyz_diff(train_data)
            norm_train = range_scaler(diff_train)

            diff_val = xyz_diff(val_data)
            norm_val = range_scaler(diff_val)

            diff_test = xyz_diff(test_data)
            norm_test = range_scaler(diff_test)

            # Save range_scaler dataset
            scipy.io.savemat(os.path.join(path, 'norm_train.mat'), {"data": norm_train, 'label': train_label})
            scipy.io.savemat(os.path.join(path, 'norm_val.mat'), {"data": norm_val, 'label': val_label})
            scipy.io.savemat(os.path.join(path, 'norm_test.mat'), {"data": norm_test, 'label': test_label})

        elif data_preprocess == 'Standardization':
            stand_train = data_standard(train_data, use_xyz_diff=True)
            stand_val = data_standard(val_data, use_xyz_diff=True)
            stand_test = data_standard(test_data, use_xyz_diff=True)

            # Save standardized dataset
            scipy.io.savemat(os.path.join(path, 'stand_train.mat'), {"data": stand_train, 'label': train_label})
            scipy.io.savemat(os.path.join(path, 'stand_val.mat'), {"data": stand_val, 'label': val_label})
            scipy.io.savemat(os.path.join(path, 'stand_test.mat'), {"data": stand_test, 'label': test_label})

        # elif data_preprocess == 'Mean_center':
        #     # Mean-centered >>>>>>>> Global mean subtract
        #     diff_train = xyz_diff(train_data)
        #     norm_train = range_scaler(diff_train)
        #
        #     diff_val = xyz_diff(val_data)
        #     norm_val = range_scaler(diff_val)
        #
        #     diff_test = xyz_diff(test_data)
        #     norm_test = range_scaler(diff_test)
        #
        #     meanarg = np.mean(norm_train)
        #     stdarg = np.std(norm_train)
        #     print('Mean value of training set is {}, std value of training set is {}'.format(meanarg, stdarg))
        #
        #     submean_train = norm_train - meanarg
        #     submean_val = norm_val - meanarg
        #     submean_test = norm_test - meanarg
        #
        #     scipy.io.savemat(os.path.join(path, 'submean_train.mat'), {'data': submean_train, 'label': train_label})
        #     scipy.io.savemat(os.path.join(path, 'submean_val.mat'), {'data': submean_val, 'label': val_label})
        #     scipy.io.savemat(os.path.join(path, 'submean_test.mat'), {'data': submean_test, 'label': test_label})
        #
        # elif data_preprocess == 'Mean_center_feature_wise':
        #     # Mean-centered >>>>>>>> Feature-wise mean subtract
        #     diff_train = xyz_diff(train_data)
        #     norm_train = range_scaler(diff_train)
        #
        #     diff_val = xyz_diff(val_data)
        #     norm_val = range_scaler(diff_val)
        #
        #     diff_test = xyz_diff(test_data)
        #     norm_test = range_scaler(diff_test)
        #
        #     feature_mean = np.mean(norm_train, axis=(0, 1), keepdims=True)
        #     feature_std = np.std(norm_train, axis=(0, 1), keepdims=True)
        #     print(
        #         'Feature Mean value of training set is {}, Feature std value of training set is {}'.format(feature_mean,
        #                                                                                                    feature_std))
        #
        #     subFeature_mean_train = norm_train - feature_mean
        #     subFeature_mean_val = norm_val - feature_mean
        #     subFeature_mean_test = norm_test - feature_mean
        #
        #     scipy.io.savemat(os.path.join(path, 'subFeaturewise_mean_train.mat'),
        #                      {'data': subFeature_mean_train, 'label': train_label})
        #     scipy.io.savemat(os.path.join(path, 'subFeaturewise_mean_val.mat'),
        #                      {'data': subFeature_mean_val, 'label': val_label})
        #     scipy.io.savemat(os.path.join(path, 'subFeaturewise_mean_test.mat'),
        #                      {'data': subFeature_mean_test, 'label': test_label})

    path = r'D:\TrajSeg-Cls\TrajSEG-CLS_V3\CLS\SNR05'

    train = scipy.io.loadmat(os.path.join(path, 'addnoise_train.mat'))
    train_X, train_Y = train['data'].squeeze(0), train['label'].squeeze(1)

    val = scipy.io.loadmat(os.path.join(path, 'addnoise_val.mat'))
    val_X, val_Y = val['data'].squeeze(0), val['label'].squeeze(1)

    test = scipy.io.loadmat(os.path.join(path, 'addnoise_test.mat'))
    test_X, test_Y = test['data'].squeeze(0), test['label'].squeeze(1)

    for preprocess in ['Standardization', 'Standard_each']:
        if preprocess == 'Standardization':
            diff_train = xyz_diff(train_X)
            stand_train, m_train, s_train = Standardization(diff_train)
            paras = {'Trainset mean:': m_train.tolist(),'Trainset std:': s_train.tolist()}

            with open(os.path.join(path, 'paras.json'), "w") as f:
                json.dump(paras, f)

            diff_val = xyz_diff(val_X)
            diff_test = xyz_diff(test_X)
            stand_val = data_stand(diff_val, m_train, s_train)
            stand_test = data_stand(diff_test, m_train, s_train)

            # Save processed dataset
            scipy.io.savemat(os.path.join(path, 'stand_train.mat'), {'data': stand_train, 'label': train_Y})
            scipy.io.savemat(os.path.join(path, 'stand_val.mat'), {'data': stand_val, 'label': val_Y})
            scipy.io.savemat(os.path.join(path, 'stand_test.mat'), {'data': stand_test, 'label': test_Y})

        elif preprocess == 'Standard_each':
            stand_train = data_standard(train_X, use_xyz_diff=True)
            stand_val = data_standard(val_X, use_xyz_diff=True)
            stand_test = data_standard(test_X, use_xyz_diff=True)

            # Save standardized dataset
            scipy.io.savemat(os.path.join(path, 'each_stand_train.mat'), {'data': stand_train, 'label': train_Y})
            scipy.io.savemat(os.path.join(path, 'each_stand_val.mat'), {'data': stand_val, 'label': val_Y})
            scipy.io.savemat(os.path.join(path, 'each_stand_test.mat'), {'data': stand_test, 'label': test_Y})

    # # Mean-centered >>>>>>>> Global mean subtract
    # meanarg = np.mean(norm_train)
    # stdarg = np.std(norm_train)
    # print('Mean value of training set is {}, std value of training set is {}'.format(meanarg, stdarg))
    #
    # submean_train = norm_train - meanarg
    # submean_val = norm_val - meanarg
    # submean_test = norm_test - meanarg
    #
    # scipy.io.savemat(path + 'submean_train.mat', {"submean_train": submean_train})
    # scipy.io.savemat(path + 'submean_val.mat', {"submean_val": submean_val})
    # scipy.io.savemat(path + 'submean_test.mat', {"submean_test": submean_test})
    #
    # # Mean-centered >>>>>>>> Feature-wise mean subtract
    # feature_mean = np.mean(norm_train, axis=(0, 1))
    # feature_std = np.std(norm_train, axis=(0, 1))
    # print('Feature Mean value of training set is {}, Feature std value of training set is {}'.format(feature_mean,
    #                                                                                                  feature_std))
    #
    # subFeature_mean_train = norm_train - feature_mean.reshape((1, 1, 5))
    # subFeature_mean_val = norm_val - feature_mean.reshape((1, 1, 5))
    # subFeature_mean_test = norm_test - feature_mean.reshape((1, 1, 5))
    #
    # scipy.io.savemat(path + 'subFeature_mean_train.mat', {"subFeature_mean_train": subFeature_mean_train})
    # scipy.io.savemat(path + 'subFeature_mean_val.mat', {"subFeature_mean_val": subFeature_mean_val})
    # scipy.io.savemat(path + 'subFeature_mean_test.mat', {"subFeature_mean_test": subFeature_mean_test})
