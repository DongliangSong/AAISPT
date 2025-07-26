# -*- coding: utf-8 -*-
# @Time    : 2023/7/14 8:52
# @Author  : Dongliang

import os

import numpy as np
import scipy

from AAISPT.Gen_trace.BasicModule import gen_truncnorm


def rotation_addnoise(sim_data, min_step_polar, max_step_polar, min_step_azimuth, max_step_azimuth, mean_step_polar,
                      std_step_polar, mean_step_azimuth, std_step_azimuth):
    """
    The simulated rotation step angles add different levels of Gaussian noise.

    :param sim_data: Simulated step angle dataset.
    :param min_step_polar: The minimum value of the step polar angle.
    :param max_step_polar: The maximum value of the step polar angle.
    :param min_step_azimuth: The minimum value of the step azimuth angle.
    :param max_step_azimuth: The maximum value of the step azimuth angle.
    :param mean_step_polar: The mean value of the step polar angle.
    :param std_step_polar: The standard deviation of the step polar angle.
    :param mean_step_azimuth: The mean value of the step azimuth angle.
    :param std_step_azimuth: The standard deviation of the step azimuth angle.
    :return: Simulated rotational step angle dataset with Gaussian noise added.
    """

    nums = sim_data.shape[0]
    for i in range(nums):
        ap = sim_data[i]
        polar = ap[:, -2]
        azi = ap[:, -1]

        polar_noise = gen_truncnorm(min_step_polar, max_step_polar, mean_step_polar, std_step_polar, len(polar))
        azi_noise = gen_truncnorm(min_step_azimuth, max_step_azimuth, mean_step_azimuth, std_step_azimuth, len(azi))

        sim_data[i][:, -2] = polar + polar_noise
        sim_data[i][:, -1] = azi + azi_noise
    return sim_data


if __name__ == '__main__':
    path = r'D:\TrajSeg-Cls\TrajSEG-CLS_V3\CLS\Var_LD\5D\SNR03'
    savepath = path
    num_class = 5
    SNR = int(path[-2:])

    for mode in ['train', 'val', 'test']:
        data = scipy.io.loadmat(os.path.join(path, f'{mode}.mat'))[f'{mode}set']

        min_step_polar, max_step_polar, mean_step_polar, std_step_polar = 0, 0, 0, 0
        min_step_azimuth, max_step_azimuth, mean_step_azimuth, std_step_azimuth = 0, 0, 0, 0

        if SNR == 1:
            min_step_polar, max_step_polar = 0, 5
            min_step_azimuth, max_step_azimuth = 0, 10
            mean_step_polar, std_step_polar, mean_step_azimuth, std_step_azimuth = 3.2, 2.5, 6.4, 8.4

        elif SNR == 3:
            min_step_polar, max_step_polar = 0, 4
            min_step_azimuth, max_step_azimuth = 0, 5
            mean_step_polar, std_step_polar, mean_step_azimuth, std_step_azimuth = 2.0, 3.0, 3.3, 7.0

        elif SNR == 5:
            min_step_polar, max_step_polar = 0, 1
            min_step_azimuth, max_step_azimuth = 0, 3
            mean_step_polar, std_step_polar, mean_step_azimuth, std_step_azimuth = 0.6, 0.7, 1.5, 1.3

        elif SNR == 10:
            min_step_polar, max_step_polar = 0, 1
            min_step_azimuth, max_step_azimuth = 0, 2
            mean_step_polar, std_step_polar, mean_step_azimuth, std_step_azimuth = 0.4, 0.6, 1.5, 1.2

        each_class = data.shape[1] // num_class

        # Use list comprehension to create data_list
        data_list = [data[0, i * each_class: (i + 1) * each_class] for i in range(num_class)]

        new_data_list = [rotation_addnoise(sim_data=data,
                                           min_step_polar=min_step_polar,
                                           max_step_polar=max_step_polar,
                                           min_step_azimuth=min_step_azimuth,
                                           max_step_azimuth=max_step_azimuth,
                                           mean_step_polar=mean_step_polar,
                                           std_step_polar=std_step_polar,
                                           mean_step_azimuth=mean_step_azimuth,
                                           std_step_azimuth=std_step_azimuth) for data in data_list]

        new_data = np.concatenate(new_data_list, axis=0)
        label = np.vstack([i * np.ones((each_class, 1)) for i in range(num_class)])
        scipy.io.savemat(os.path.join(savepath, f'addnoise_{mode}.mat'), {'data': new_data, 'label': label})
