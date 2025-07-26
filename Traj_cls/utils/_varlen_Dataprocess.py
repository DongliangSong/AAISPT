# -*- coding: utf-8 -*-
# @Time    : 2024/7/3 19:41
# @Author  : Dongliang

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
    norm_r = (r - mean_r) / std_r

    return norm_r


def data_standard_VarL(X):
    num = X.shape[0]
    news = []
    for i in range(num):
        data = X[i]
        xyz = np.diff(data[:, :3], axis=0)
        ap = data[:-1, 3:]
        new = np.concatenate((xyz, ap), axis=1)

        new = (new - np.mean(new, axis=0, keepdims=True)) / np.std(new, axis=0, keepdims=True)
        news.append(new)
    return news


def xyz_diff(data):
    """
    Calculate the difference between adjacent time steps for xyz or xy.

    :param data: Input data in the shape of (number of samples, trajectory length, trajectory dimension).
    :return: Differential data with shape (number of samples, trajectory length, trajectory dimension).
    """

    # For 3-D XYZ and azimuth, polar angle
    diff = []
    for i in range(len(data)):
        x = data[i]
        data_diff = np.diff(x[:, :3], axis=0)
        new = np.concatenate((data_diff, x[:x.shape[0] - 1, 3:]), axis=1)
        diff.append(new)

    return diff


def range_scaler(data):
    """
    Min-max normalization (scaling data range)
    :param data:
    :return:
    """
    m = len(data)
    norm = []
    for i in range(m):
        y = data[i]
        norm.append((y - np.min(y, axis=0, keepdims=True)) / (
                np.max(y, axis=0, keepdims=True) - np.min(y, axis=0, keepdims=True)))
    return norm


def xyz_scaler(X):
    m = len(X)
    norm = []
    for i in range(m):
        data = X[i]
        xyz = data[:, :3]
        xyz = (xyz - np.min(xyz, axis=0, keepdims=True)) / (
                np.max(xyz, axis=0, keepdims=True) - np.min(xyz, axis=0, keepdims=True))
        ap = data[:, 3:]
        new = np.concatenate((xyz, ap), axis=1)
        norm.append(new)
    return norm


def Standardization(data):
    """
    Feature-wise Standardization for train and val set.
    :param data: Input data in the shape of (number of samples, trajectory length, trajectory dimension).
    :return: Standardized data with shape (number of samples, trajectory length, trajectory dimension).
    """
    thr = 1e-10
    m = np.mean(data, axis=(0, 1), keepdims=True)
    s = np.std(data, axis=(0, 1), keepdims=True)
    new = (data - m) / np.where(s > thr, s, 1)
    return new, m, s


def xyzap_stand(data):
    """
    Sample-wise Standardization
    :param data:
    :return:
    """
    thr = 1e-10
    t = []
    for i in range(len(train_X)):
        t.extend(train_X[i])

    m = np.mean(t, axis=0, keepdims=True)
    s = np.std(t, axis=0, ddof=0, keepdims=True)

    xyzap = []
    for i in range(len(data)):
        new = (data[i] - m) / np.where(s > thr, s, 1)
        xyzap.append(new)
    return m, s, xyzap


def data_stand(data, mean_train, std_train):
    """
    Feature-wise Standardization for unknown dataset
    :param data: Input data in the shape of (number of samples, trajectory length, trajectory dimension).
    :param mean_train: The mean of the training set.
    :param std_train: Standard deviation of the training set.
    :return: Standardized data with shape (number of samples, trajectory length, trajectory dimension).
    """
    thr = 1e-10
    new = []
    for i in range(len(data)):
        new.append((data[i] - mean_train) / np.where(std_train > thr, std_train, 1))
    return new


def xyzap_submean(data, train=False):
    """
    Full dataset standardization
    :param data:
    :return:
    """
    stand_data = []
    mean = 0
    nums = data.shape[0]
    for i in range(nums):
        xyzap = data[i]

        # Diff
        xyz_diff = np.diff(xyzap[:, :3], axis=0)
        xyzap_new = np.concatenate((xyz_diff, xyzap[:xyzap.shape[0] - 1, 3:]), axis=1)

        mean += np.mean(xyzap_new, axis=0, keepdims=True)
        stand_data.append(xyzap_new)

    total_mean = mean / nums

    if train:
        return stand_data, total_mean
    else:
        return stand_data


def data_process(data):
    num = len(data)
    new = []
    for i in range(num):
        x = data[i]
        x = x - np.mean(x, axis=0, keepdims=True)
        dx = np.diff(x, axis=0)
        dx = dx / np.std(dx, axis=0, keepdims=True)
        new.append(dx)
    return new


if __name__ == '__main__':
    path = r'D:\TrajSeg-Cls\TrajSEG-CLS_V3\CLS\Var_L_500\SNR05'
    dimension = 5

    # Subtract mean
    train = scipy.io.loadmat(os.path.join(path, 'addnoise_train.mat'))
    val = scipy.io.loadmat(os.path.join(path, 'addnoise_val.mat'))
    test = scipy.io.loadmat(os.path.join(path, 'addnoise_test.mat'))

    train_X, train_Y = train['data'].squeeze(), train['label'].squeeze()
    val_X, val_Y = val['data'].squeeze(), val['label'].squeeze()
    test_X, test_Y = test['data'].squeeze(), test['label'].squeeze()

    # # 方案1
    # train_X = xyz_diff(train_X)
    # val_X = xyz_diff(val_X)
    # test_X = xyz_diff(test_X)
    #
    # mean_train, std_train, train_X = xyzap_stand(train_X)
    # val_X = data_stand(val_X, mean_train, std_train)
    # test_X = data_stand(test_X, mean_train, std_train)
    #
    # data = {'mean': mean_train.tolist(), 'std': std_train.tolist()}
    # with open(os.path.join(path, 'mean_std.json'), 'w', encoding='utf-8') as f:
    #     json.dump(data, f)
    #
    # scipy.io.savemat(os.path.join(path, 'All_stand_train.mat'), {'data': train_X, 'label': train_Y})
    # scipy.io.savemat(os.path.join(path, 'All_stand_val.mat'), {'data': val_X, 'label': val_Y})
    # scipy.io.savemat(os.path.join(path, 'All_stand_test.mat'), {'data': test_X, 'label': test_Y})

    # # 方案2
    # train_X = xyz_scaler(train_X)
    # val_X = xyz_scaler(val_X)
    # test_X = xyz_scaler(test_X)
    #
    # train_X = xyz_diff(train_X)
    # val_X = xyz_diff(val_X)
    # test_X = xyz_diff(test_X)
    #
    # mean_train, std_train, train_X = xyzap_stand(train_X)
    # val_X = data_stand(val_X, mean_train, std_train)
    # test_X = data_stand(test_X, mean_train, std_train)
    #
    # data = {'mean': mean_train.tolist(), 'std': std_train.tolist()}
    # with open(os.path.join(path, 'mean_std.json'), 'w', encoding='utf-8') as f:
    #     json.dump(data, f)
    #
    # scipy.io.savemat(os.path.join(path, 'All_stand_train.mat'), {'data': train_X, 'label': train_Y})
    # scipy.io.savemat(os.path.join(path, 'All_stand_val.mat'), {'data': val_X, 'label': val_Y})
    # scipy.io.savemat(os.path.join(path, 'All_stand_test.mat'), {'data': test_X, 'label': test_Y})

    # # 方案3
    # train_X = xyz_diff(train_X)
    # val_X = xyz_diff(val_X)
    # test_X = xyz_diff(test_X)
    #
    # train_X = xyz_scaler(train_X)
    # val_X = xyz_scaler(val_X)
    # test_X = xyz_scaler(test_X)
    #
    # mean_train, std_train, train_X = xyzap_stand(train_X)
    # val_X = data_stand(val_X, mean_train, std_train)
    # test_X = data_stand(test_X, mean_train, std_train)
    #
    # data = {'mean': mean_train.tolist(), 'std': std_train.tolist()}
    # with open(os.path.join(path, 'mean_std.json'), 'w', encoding='utf-8') as f:
    #     json.dump(data, f)
    #
    # scipy.io.savemat(os.path.join(path, 'All_stand_train.mat'), {'data': train_X, 'label': train_Y})
    # scipy.io.savemat(os.path.join(path, 'All_stand_val.mat'), {'data': val_X, 'label': val_Y})
    # scipy.io.savemat(os.path.join(path, 'All_stand_test.mat'), {'data': test_X, 'label': test_Y})

    # 方案4
    train_X = xyz_diff(train_X)
    val_X = xyz_diff(val_X)
    test_X = xyz_diff(test_X)

    train_X = xyz_scaler(train_X)
    val_X = xyz_scaler(val_X)
    test_X = xyz_scaler(test_X)

    scipy.io.savemat(os.path.join(path, 'All_stand_train.mat'), {'data': train_X, 'label': train_Y})
    scipy.io.savemat(os.path.join(path, 'All_stand_val.mat'), {'data': val_X, 'label': val_Y})
    scipy.io.savemat(os.path.join(path, 'All_stand_test.mat'), {'data': test_X, 'label': test_Y})

# for preprocess in ['norm', 'stand']:
#     if preprocess == 'submean':
#         submean_train, mean_train = xyzap_submean(train_X, train=True)
#         submean_train = [i - mean_train for i in submean_train]
#
#         submean_val = xyzap_submean(val_X)
#         submean_val = [i - mean_train for i in submean_val]
#
#         submean_test = xyzap_submean(test_X)
#         submean_test = [i - mean_train for i in submean_test]
#
#         # Save processed dataset
#         scipy.io.savemat(os.path.join(path, 'submean_train.mat'), {'data': submean_train, 'label': train_Y})
#         scipy.io.savemat(os.path.join(path, 'submean_val.mat'), {'data': submean_val, 'label': val_Y})
#         scipy.io.savemat(os.path.join(path, 'submean_test.mat'), {'data': submean_test, 'label': test_Y})
#
#         with open(os.path.join(path, 'mean_train.json'), 'w') as f:
#             json.dump(mean_train, f)
#
#     elif preprocess == 'stand':
#         stand_train = xyzap_stand(train_X)
#         stand_val = xyzap_stand(val_X)
#         stand_test = xyzap_stand(test_X)
#
#         # Save processed dataset
#         scipy.io.savemat(os.path.join(path, 'stand_train.mat'), {'data': stand_train, 'label': train_Y})
#         scipy.io.savemat(os.path.join(path, 'stand_val.mat'), {'data': stand_val, 'label': val_Y})
#         scipy.io.savemat(os.path.join(path, 'stand_test.mat'), {'data': stand_test, 'label': test_Y})
#
#     elif preprocess == 'norm':
#         diff_train = xyz_diff(train_X)
#         diff_val = xyz_diff(val_X)
#         diff_test = xyz_diff(test_X)
#         norm_train = range_scaler(diff_train)
#         norm_val = range_scaler(diff_val)
#         norm_test = range_scaler(diff_test)
#
#         # Save processed dataset
#         scipy.io.savemat(os.path.join(path, 'norm_train.mat'), {'data': norm_train, 'label': train_Y})
#         scipy.io.savemat(os.path.join(path, 'norm_val.mat'), {'data': norm_val, 'label': val_Y})
#         scipy.io.savemat(os.path.join(path, 'norm_test.mat'), {'data': norm_test, 'label': test_Y})
#
#         # # Mean-centered >>>>>>>> Global mean subtract
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
