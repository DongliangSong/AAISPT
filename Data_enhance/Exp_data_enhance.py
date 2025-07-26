# -*- coding: utf-8 -*-
# @Time    : 2024/8/22 20:54
# @Author  : Dongliang


"""
Enhancement processing is performed on the experimental dataset, including operations such as translation,
scaling, adding random noise, interpolation, slicing, interference, etc.
"""

import glob
import os
import random
from datetime import datetime

import numpy as np
import pandas as pd
import scipy
import scipy.stats as stats
from scipy.interpolate import interp1d


def gen_truncnorm(std, num):
    """Generate random numbers with a given mean and standard deviation, but truncated at multiples of std.
    """
    dist = stats.truncnorm(0, 1, loc=0, scale=std)
    values = dist.rvs(num)
    return values


def noise_2_3D(data, noise_level_1):
    """
    Adding noise to 2D or 3D experimental data.

    :param data: Raw experimental data or data-enhanced trajectory datasets.
    :param noise_level_1: Standard deviation of Gaussian noise for XY or XYZ.
    :return: List of trajectories after adding noise.
    """
    noise_level_11 = [gen_truncnorm(noise_level_1, 1) for _ in range(len(data))]

    noisy_3Ddata = []
    for timeseries, level in zip(data, noise_level_11):
        noise = np.random.normal(0, level, timeseries.shape)
        noisy_timeseries = np.multiply(timeseries, 1 + noise)
        noisy_3Ddata.append(noisy_timeseries)
    return noisy_3Ddata


def noise_4_5D(data, noise_level_1, noise_level_2):
    """
    Adding noise to 4D or 5D experimental data.

    :param data: Raw experimental data or data-enhanced trajectory datasets.
    :param noise_level_1: Standard deviation of Gaussian noise for XY or XYZ.
    :param noise_level_2: Standard deviation of Gaussian noise for azimuth and polar angles.
    :return: List of trajectories after adding noise.
    """

    noise_level_11 = [gen_truncnorm(noise_level_1, 1) for _ in range(len(data))]
    noise_level_22 = [gen_truncnorm(noise_level_2, 1) for _ in range(len(data))]

    noisy_5Ddata = []
    for timeseries, level_1, level_2 in zip(data, noise_level_11, noise_level_22):
        noise_1 = np.random.normal(0, level_1, timeseries[:, :-2].shape)
        noisy_s1 = np.multiply(timeseries[:, :-2], 1 + noise_1)

        noise_2 = np.random.normal(0, level_2, timeseries[:, -2:].shape)
        noisy_s2 = np.multiply(timeseries[:, -2:], 1 + noise_2)

        noisy_timeseries = np.concatenate((noisy_s1, noisy_s2), axis=1)
        noisy_5Ddata.append(noisy_timeseries)
    return noisy_5Ddata


def shift(data, shift_min_ratio, shift_max_ratio):
    """
    Shifting of time series data.

    :param data: Raw experimental data or data-enhanced trajectory datasets.
    :param shift_min_ratio: minimum shift ratio.
    :param shift_max_ratio: maximum shift ratio.
    :return: List of trajectories after shifting.
    """
    if shift_min_ratio <= 0 or shift_max_ratio >= 1:
        raise ValueError("shift ratio must be in (0, 1)")

    if shift_min_ratio >= shift_max_ratio:
        raise ValueError("min_ratio must be less than max_ratio")

    shift_ratios = [random.uniform(shift_min_ratio, shift_max_ratio) for _ in range(len(data))]

    shifted_data = []
    for timeseries, shift_ratio in zip(data, shift_ratios):
        shift_amount = int(len(timeseries) * shift_ratio)
        if shift_amount >= 0 and shift_amount < len(timeseries):
            shifted_timeseries = timeseries[:-shift_amount, :]
        elif shift_amount < 0:
            shifted_timeseries = timeseries[-shift_amount:, :]
        elif shift_amount >= len(timeseries):
            shifted_timeseries = timeseries[:, :]
        shifted_data.append(shifted_timeseries)
    return shifted_data


def interpolate_1d(data, factor):
    """
    Interpolation of time series data along each dimension.

    :param data: Raw experimental data or data-enhanced trajectory datasets.
    :param factor: Interpolation factors.
    :return: Interpolated array of trajectories.
    """

    interpolated_data = []

    # Interpolation for each dimension
    for i in range(data.shape[1]):
        current_data = data[:, i]

        f = interp1d(np.arange(len(current_data)), current_data, kind='linear')
        interpolated_dimension_data = f(np.linspace(0, len(current_data) - 1, len(current_data) * factor))
        interpolated_data.append(interpolated_dimension_data)
    return np.array(interpolated_data).transpose()


def interpolate(data, factor_min, factor_max):
    """
    Time series interpolation.

    :param data: Raw experimental data or data-enhanced trajectory datasets.
    :param factor_min: Minimum interpolation factor.
    :param factor_max: Maximum interpolation factor.
    :return: List of interpolated time series data.
    """
    if factor_min >= factor_max:
        raise ValueError("Invalid interpolation factors.")

    factors = [np.random.randint(factor_min, factor_max) for _ in range(len(data))]

    interpolated_data = []
    for timeseries, factor in zip(data, factors):
        interpolated_timeseries = interpolate_1d(timeseries, factor)

        interpolated_data.append(interpolated_timeseries)
    return interpolated_data


def scaling(data, factor_min, factor_max):
    """
    Scaling of time series data.

    :param data: Raw experimental data or data-enhanced trajectory datasets.
    :param factor_min: Minimum scaling factor.
    :param factor_max: Maximum scaling factor.
    :return: List of scaled time series data.
    """
    if factor_min >= factor_max:
        raise ValueError("Invalid scaling factors.")

    bounds = [np.random.uniform(factor_min, factor_max) for _ in range(len(data))]

    scaled_data = []
    for timeseries, bound in zip(data, bounds):
        scale_factor = np.random.uniform(0, bound, [1, timeseries.shape[1]])
        scaled_timeseries = np.multiply(timeseries, scale_factor)
        scaled_data.append(scaled_timeseries)
    return scaled_data


def add_noise(data, dim, noise_level_1, noise_level_2):
    """
    Add noise to time series data.

    :param data: Raw experimental data or data-enhanced trajectory datasets.
    :param dim: Dimensions of time series data, e.g. 2 or 3.
    :param noise_level_1: Standard deviation of Gaussian noise in XY or XYZ.
    :param noise_level_2: Standard deviation of Gaussian noise in azimuth and polar angles.
    :return: List of trajectories with Gaussian noise.
    """

    if dim <= 3:
        noisy_data = noise_2_3D(data, noise_level_1)
    elif dim > 3:
        noisy_data = noise_4_5D(data, noise_level_1, noise_level_2)
    return noisy_data


def window_slice(data, ratio):
    """
    Slicing of time series data.

    :param data: Raw experimental data or data-enhanced trajectory datasets.
    :param ratio: Ratio of slices to total length.
    :return: List of tracks after slicing.
    """

    if ratio >= 1 or ratio <= 0:
        raise ValueError("Ratio must be between 0 and 1.")

    slice_data = []
    for timeseries in data:
        target_len = int(np.floor(len(timeseries) * ratio))

        start = np.random.randint(low=0, high=len(timeseries) - target_len)
        end = start + target_len
        slice_data.append(timeseries[start:end, :])
    return slice_data


def reverse_(data):
    """
    Reverse each dimension independently while keeping the order of dimensions unchanged.

    :param data: Raw experimental data or data-enhanced trajectory datasets.
    :return: Reversed multidimensional time series data.
    """

    # Iterate through each dimension and reverse the data along that dimension
    reversed_data = []
    for dim in data:
        reversed_data.append(np.flip(dim, axis=0))
    # reversed_data = [np.flip(dim, axis=0) for dim in data]

    return reversed_data


def Multi_shift(data, nums, min_shift_ratio, max_shift_ratio, const):
    """
    Shifting of multiple time series trajectories.

    :param data: Raw experimental data or data-enhanced trajectory datasets.
    :param nums: The number of trajectories to be extended.
    :param min_shift_ratio: Minimum shift ratio.
    :param max_shift_ratio: Maximum shift ratio.
    :param const: The difference between the maximum and minimum shift.
    :return: List of trajectories after shifting.
    """
    if min_shift_ratio <= 0 or max_shift_ratio >= 1:
        raise ValueError("max_ratio must be in (0, 1)")

    if min_shift_ratio >= max_shift_ratio:
        raise ValueError("min_ratio must be less than max_ratio")

    aug_data = []
    shift_min_ratios = np.linspace(min_shift_ratio, max_shift_ratio, nums)
    shift_max_ratios = shift_min_ratios + const

    if len(data) != nums:
        for shift_min_val, shift_max_val in zip(shift_min_ratios, shift_max_ratios):
            aug_data.append(shift(data=data, shift_min_ratio=shift_min_val, shift_max_ratio=shift_max_val))
    else:
        for timeseries, shift_min_val, shift_max_val in zip(data, shift_min_ratios, shift_max_ratios):
            aug_data.append(shift(data=timeseries, shift_min_ratio=shift_min_val, shift_max_ratio=shift_max_val))
    return aug_data


def Multi_interpolate(data, nums, factor, const):
    """
    Interpolation of multiple time series trajectories.

    :param data: Raw experimental data or data-enhanced trajectory datasets.
    :param nums: The number of trajectories to be extended.
    :param factor: The interpolation factor.
    :param const: The difference between the maximum and minimum interpolation factor.
    :return: List of trajectories after interpolation.
    """
    aug_data = []
    factor_min = np.linspace(1, factor, nums)
    factor_max = factor_min + const

    if len(data) != nums:
        for factor_min_val, factor_max_val in zip(factor_min, factor_max):
            aug_data.append(interpolate(data=data, factor_min=factor_min_val, factor_max=factor_max_val))
    else:
        for timeseries, factor_min_val, factor_max_val in zip(data, factor_min, factor_max):
            aug_data.append(interpolate(data=timeseries, factor_min=factor_min_val, factor_max=factor_max_val))
    return aug_data


def Multi_scaling(data, nums, factor, const):
    """
    Scaling of multiple time series trajectories.

    :param data: Raw experimental data or data-enhanced trajectory datasets.
    :param nums: The number of trajectories to be extended.
    :param factor: The scaling factor.
    :param const: The difference between the maximum and minimum scaling factor.
    :return: List of trajectories after scaling.
    """
    aug_data = []

    bounds_min = np.linspace(0, factor, nums)
    bounds_max = bounds_min + const

    if len(data) != nums:
        for bounds_min_val, bounds_max_val in zip(bounds_min, bounds_max):
            aug_data.append(scaling(data=data, factor_min=bounds_min_val, factor_max=bounds_max_val))
    else:
        for timeseries, bounds_min_val, bounds_max_val in zip(data, bounds_min, bounds_max):
            aug_data.append(scaling(data=timeseries, factor_min=bounds_min_val, factor_max=bounds_max_val))
    return aug_data


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
            aug_data.append(add_noise(data=data, dim=5, noise_level_1=noise_level_val, noise_level_2=noise_level_val))
    else:
        for timeseries, noise_level_val in zip(data, noise_level):
            aug_data.append(
                add_noise(data=timeseries, dim=5, noise_level_1=noise_level_val, noise_level_2=noise_level_val))
    return aug_data


def Multi_windowslice(data, nums, max_ratio=0.99):
    """
    Slicing multiple time series trajectories.

    :param data: Raw experimental data or data-enhanced trajectory datasets.
    :param nums: The number of trajectories to be extended.
    :param max_ratio: Maximum value of the ratio of slices to total length.
    :return: List of trajectories after slicing.
    """
    if max_ratio <= 0 or max_ratio >= 1:
        raise ValueError("max_ratio must be in (0, 1)")

    aug_data = []
    reduce_ratio = np.linspace(0.5, max_ratio, nums)

    if len(data) != nums:
        for reduce_ratio_val in reduce_ratio:
            aug_data.append(window_slice(data=data, ratio=reduce_ratio_val))
    else:
        for timeseries, reduce_ratio_val in zip(data, reduce_ratio):
            aug_data.append(window_slice(data=timeseries, ratio=reduce_ratio_val))
    return aug_data


def Multi_reverse(data, nums, const):
    """
    Inversion of multiple time series trajectories.

    :param data: Raw experimental data or data-enhanced trajectory datasets.
    :param nums: The number of trajectories to be extended.
    :param const: Placeholder for subsequent operations. It has no practical significance.
    :return: List of trajectories after inversion.
    """

    aug_data = []
    if len(data) != nums:
        aug_data = reverse_(data=data)
    else:
        for timeseries in data:
            aug_data.append(reverse_(data=timeseries))
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


def combine_methods(data, aug_methods, nums, params):
    """
    Combine multiple methods for data enhancement.

    :param data: Raw experimental data or data-enhanced trajectory datasets.
    :param aug_methods: Data enhancement methods including jitter, shift, interpolate, scaling, add noise, window_slice, reverse.
    :param nums: The number of single trajectory to be extended.
    :param params: Parameters corresponding to each method.

    """

    params_list = list(params.keys())
    method_names = list(aug_methods.keys())

    combined_data = []
    for i in range(len(method_names)):
        for j in range(len(method_names)):
            if i != j:
                method1 = aug_methods[method_names[i]]
                method2 = aug_methods[method_names[j]]
                print(method1, method2)

                aug_data_i = method1(data, nums, **params[params_list[i]])
                print(len(aug_data_i))
                aug_data_j = method2(aug_data_i, nums, **params[params_list[j]])
                combined_data.append(aug_data_j)
    return combined_data


if __name__ == '__main__':
    path = glob.glob('D:\TrajSeg-Cls\Exp Data\QiPan\Fig2\*')
    savepath = r'D:\TrajSeg-Cls\TrajSEG-CLS_V3\CLS\YanYu_NEW\small'
    if not os.path.exists(savepath):
        os.mkdir(savepath)

    label_mapping = {
        'S1_Tracks': 0,
        'S1-S2_tracks': 1,
        'S2_Tracks': 2
    }

    # Experimental dataset enhancement
    num_methods = 5
    expand_nums = 3
    dataset = []
    label = []

    for i in path:
        dirname, file = os.path.split(i)
        if file not in label_mapping.keys():
            continue

        filename = glob.glob(os.path.join(i, 'Tracks/*.xlsx'))

        for key, value in label_mapping.items():
            if key in i:
                label.extend([value] * len(filename))

        for j in filename:
            data = pd.read_excel(j).values
            trace = data[:, 1:]
            dataset.append(trace)

    aug_methods = {
        'Multi_shift': Multi_shift,
        'Multi_interpolate': Multi_interpolate,
        'Multi_addnoise': Multi_addnoise,
        'Multi_windowslice': Multi_windowslice,
        'Multi_reverse': Multi_reverse
    }

    sing_params = {
        'Multi_shift': {'min_shift_ratio': 0.1, 'max_shift_ratio': 0.6, 'const': 0.05},
        'Multi_interpolate': {'factor': 3, 'const': 1},
        'Multi_addnoise': {'min_level': 0.005, 'max_level': 0.008},
        'Multi_windowslice': {'max_ratio': 0.7},
        'Multi_reverse': {'const': 0}
    }

    # Set parameters for data augmentation
    comb_params = {
        'Multi_shift': {'min_shift_ratio': 0.1, 'max_shift_ratio': 0.6, 'const': 0.05},
        'Multi_interpolate': {'factor': 3, 'const': 1},
        'Multi_addnoise': {'min_level': 0.005, 'max_level': 0.008},
        'Multi_windowslice': {'max_ratio': 0.7},
        'Multi_reverse': {'const': 0}
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
    for i, single in enumerate(single_augs):
        if i < num_methods - 1:
            for j in range(len(single)):
                single_augs_data.extend(single[j])
        else:
            single_augs_data.extend(single)

    single_augs_label = label * (expand_nums * (num_methods - 1) + 1)
    print('single_augs_data:', len(single_augs_data), 'single_augs_label:', len(single_augs_label))

    # Combine methods for data enhancement
    print('Combine methods for data enhancement...')
    combinations = combine_methods(
        data=dataset,
        aug_methods=aug_methods,
        nums=expand_nums,
        params=comb_params
    )

    comb_augs_data = []
    for i, comb in enumerate(combinations):
        for j in range(len(comb)):
            comb_augs_data.extend(comb[j])

    comb_augs_label = label * expand_nums * (num_methods * (num_methods - 1))
    print('comb_augs_data:', len(comb_augs_data), 'comb_augs_label:', len(comb_augs_label))

    print('Save data...')
    merge_augs = single_augs_data + comb_augs_data
    merge_label = single_augs_label + comb_augs_label

    scipy.io.savemat(os.path.join(savepath, 'aug_data.mat'), {'data': merge_augs, 'label': merge_label})

    # Parameters are written to the log and saved.
    current_date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    file_name = os.path.join(savepath, f'params_log_{current_date}.txt')

    with open(file_name, 'w', encoding='utf-8') as file:
        file.write('Single method params\n')
        for key, value in sing_params.items():
            file.write(f'{key}: {value}\n')
        file.write('\n')

        file.write('Combination method params\n')
        for key, value in comb_params.items():
            file.write(f'{key}: {value}\n')
        file.write('\n')

        file.write('expand_nums: ' + str(expand_nums) + '\n')
        file.write('total_nums: ' + str(len(merge_label)) + '\n')

    print('Done!')
