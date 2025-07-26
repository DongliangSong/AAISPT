# -*- coding: utf-8 -*-
# @Time    : 2025/1/10 20:57
# @Author  : Dongliang

import os
from datetime import datetime

import numpy as np
import scipy
import scipy.stats as stats


def gen_truncnorm(std, num):
    """Generate random numbers with a given mean and standard deviation, but truncated at multiples of std.
    """
    dist = stats.truncnorm(0, 1, loc=0, scale=std)
    values = dist.rvs(num)
    return values


def noise_6D(data, noise_level_1, noise_level_2, noise_level_3):
    """
    Adding noise to 5D or 6D experimental data.

    :param data: Raw experimental data or data-enhanced trajectory datasets.
    :param noise_level_1: Standard deviation of Gaussian noise for XY or XYZ.
    :param noise_level_2: Standard deviation of Gaussian noise for azimuth and polar angles.
    :return: List of trajectories after adding noise.
    """

    noise_level_11 = [gen_truncnorm(noise_level_1, 1) for _ in range(len(data))]
    noise_level_22 = [gen_truncnorm(noise_level_2, 1) for _ in range(len(data))]
    noise_level_33 = [gen_truncnorm(noise_level_3, 1) for _ in range(len(data))]

    noisy_6Ddata = []
    for timeseries, level_1, level_2, level_3 in zip(data, noise_level_11, noise_level_22, noise_level_33):
        noise_1 = np.random.normal(0, level_1, timeseries[:, :-3].shape)
        noisy_s1 = np.multiply(timeseries[:, :-3], 1 + noise_1)

        noise_2 = np.random.normal(0, level_2, timeseries[:, -3:-1].shape)
        noisy_s2 = np.multiply(timeseries[:, -3:-1], 1 + noise_2)

        noise_3 = np.random.normal(0, level_3, timeseries[:, -1].shape)
        noisy_s3 = np.multiply(timeseries[:, -1], 1 + noise_3)

        noisy_s3 = np.expand_dims(noisy_s3, axis=1)
        noisy_timeseries = np.concatenate((noisy_s1, noisy_s2, noisy_s3), axis=1)
        noisy_6Ddata.append(noisy_timeseries)
    return noisy_6Ddata


def add_noise(data, noise_level_1, noise_level_2, noise_level_3):
    """
    Add noise to time series data.

    :param data: Raw experimental data or data-enhanced trajectory datasets.
    :param dim: Dimensions of time series data, e.g. 2 or 3.
    :param noise_level_1: Standard deviation of Gaussian noise in XY or XYZ.
    :param noise_level_2: Standard deviation of Gaussian noise in azimuth and polar angles.
    :return: List of trajectories with Gaussian noise.
    """

    noisy_data = noise_6D(data, noise_level_1, noise_level_2, noise_level_3)
    return noisy_data


def Multi_addnoise(data, nums, min_level, max_level):
    """
    Add noise to multiple time series trajectories.

    :param data: Raw experimental data or data-enhanced trajectory datasets.
    :param nums: The number of trajectories to be extended.
    :param max_level: The maximum value of standard deviation of noise.
    :return: List of trajectories after adding noise.
    """
    if max_level <= min_level:
        raise ValueError("max_level should be greater than min_level.")

    aug_data = []
    noise_level = np.linspace(min_level, max_level, nums)

    if len(data) != nums:
        for noise_level_val in noise_level:
            aug_data.append(add_noise(data=data,
                                      noise_level_1=noise_level_val,
                                      noise_level_2=noise_level_val,
                                      noise_level_3=noise_level_val
                                      )
                            )
    else:
        for timeseries, noise_level_val in zip(data, noise_level):
            aug_data.append(add_noise(data=timeseries,
                                      noise_level_1=noise_level_val,
                                      noise_level_2=noise_level_val,
                                      noise_level_3=noise_level_val
                                      )
                            )
    return aug_data


def single_methods(data, aug_methods, nums, params):
    """
    Apply a single method for data enhancement.

    :param data: Raw experimental data or data-enhanced trajectory datasets.
    :param aug_methods: Data enhancement methods including jitter, shift, interpolate, scaling, add noise, window_slice, reverse.
    :param nums: The number of trajectories to be extended.
    :param params: Parameters for various methods.
    :return: List of trajectories after single method enhancement.
    """
    aug_data = []

    params_list = list(params.keys())
    method_names = list(aug_methods.keys())

    for i in range(len(method_names)):
        method = aug_methods[method_names[i]]
        print(method)
        aug_data_i = method(data, nums, **params[params_list[i]])
        aug_data.append(aug_data_i)

    return aug_data


if __name__ == '__main__':
    path = r'D:\TrajSeg-Cls\endoysis\Aug_SPT_FLU'
    savepath = path

    data = scipy.io.loadmat(os.path.join(path, 'pos_neg_data.mat'))
    pos_data, pos_label = data['positive'][:, 0], data['positive'][:, 1]
    neg_data, neg_label = data['negative'][:, 0], data['negative'][:, 1]

    # Experimental dataset enhancement
    expand_nums = 5
    nums = pos_data.shape[0]

    dataset = []
    label = []
    for i in range(nums):
        lens, dim = pos_data[i].shape
        fill = np.zeros((lens,), dtype=np.float32)
        if dim != 6:
            pos_data[i] = np.insert(pos_data[i], 2, fill, axis=1)

        lens, dim = neg_data[i].shape
        fill = np.zeros((lens,), dtype=np.float32)
        if dim != 6:
            neg_data[i] = np.insert(neg_data[i], 2, fill, axis=1)

        dataset.append(pos_data[i])
        label.append(pos_label[i])
        dataset.append(neg_data[i])
        label.append(neg_label[i])

    # Set the augmentation method and parameters
    aug_methods = {
        'Multi_addnoise': Multi_addnoise
    }

    sing_params = {
        'Multi_addnoise': {'min_level': 0.001, 'max_level': 0.002}
    }

    # Single method for data enhancement
    print('Single method for data enhancement...')
    single_augs = single_methods(
        data=dataset,
        aug_methods=aug_methods,
        nums=expand_nums,
        params=sing_params
    )

    single_augs_data = []
    num_methods = len(aug_methods)
    for i, single in enumerate(single_augs):
        for j in range(len(single)):
            single_augs_data.extend(single[j])

    single_augs_label = label * (expand_nums * num_methods)
    print('single_augs_data:', len(single_augs_data), 'single_augs_label:', len(single_augs_label))

    print('Save data...')
    merge_augs = single_augs_data
    merge_label = single_augs_label
    scipy.io.savemat(os.path.join(savepath, 'aug_data.mat'), {'data': merge_augs, 'label': merge_label})

    # Parameters are written to the log and saved.
    current_date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    file_name = os.path.join(savepath, f'params_log_{current_date}.txt')

    with open(file_name, 'w', encoding='utf-8') as file:
        file.write('Single method params\n')
        for key, value in sing_params.items():
            file.write(f'{key}: {value}\n')
        file.write('\n')
        file.write('expand_nums: {}\n'.format(expand_nums))
        file.write('total_nums: {}\n'.format(len(merge_augs)))

    print('Done!')
