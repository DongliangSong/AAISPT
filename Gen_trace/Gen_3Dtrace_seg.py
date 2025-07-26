# -*- coding: utf-8 -*-
# @Time    : 2024/8/6 20:30
# @Author  : Dongliang

"""
    Generate simulated trajectories for trajectory segmentation.
"""

import os
import time

import scipy

from AAISPT.Gen_trace.BasicModule import *
from AAISPT.Gen_trace.read_traces import read_java_txt


def gen_segdata(flag, sim_data):
    """
    Generate simulated trajectories for trajectory segmentation.

    :param flag: Signs of different diffusion patterns. 0 for ND, 1 for CD, 2 for DM.
    :param sim_data: The simulated trajectories of different types.
    :return: Generate diffusion trajectories (XY or XYZ).
    """

    num_perclass = num_traces // num_class
    if flag == 0:
        # load Normal diffusion (ND)
        diff = sim_data[0:num_perclass * 1]

    elif flag == 1:
        # load Confined diffusion (CD)
        diff = sim_data[num_perclass * 1:num_perclass * 2]

    elif flag == 2:
        # load Directed motion (DM)
        diff = sim_data[num_perclass * 2:]

    return diff


def gen_trace(sim_data, num_class, num_perclass, trace_len):
    """
    The different types of trajectories were concatenated to generate simulated trajectories for the segmentation task.

    :param sim_data: The simulated trajectories of different types.
    :param num_class: The number of trajectory types.
    :param num_perclass: The number of trajectories of each type.
    :param trace_len: The length of each trajectory.
    :return: Generate a dataset containing mixed trajectories and the change point position for each trajectory.
    """
    dataset = []
    position = []

    for i in range(num_class):
        for j in range(num_class):
            if i != j:
                traj1_diff = gen_segdata(i, sim_data=sim_data)
                traj2_diff = gen_segdata(j, sim_data=sim_data)

                change_point = np.random.randint(30, trace_len - 30, num_perclass)

                traces = np.array([np.vstack((
                    traj1_diff[k, :change_point[k]],
                    traj2_diff[k, :(trace_len - change_point[k])] + traj1_diff[k, change_point[k]])) for k in
                    range(num_perclass)], dtype=np.float32)

                dataset.extend(traces)
                position.extend(change_point)
                print('first segment in {} class,second segment in {} class'.format(i, j))

    position = np.array(position).reshape(-1, 1)
    return np.array(dataset), position


if __name__ == '__main__':
    start_time = time.time()

    dir_name = r'D:\TrajSeg-Cls\TrajSEG-CLS_V3\SEG\Second batch\3D\SNR03'
    flag = 'Fixed length trajectory'
    trace_len = 201
    num_class = 3

    for mode in ['train', 'val', 'test']:
        if mode == 'train':
            path = os.path.join(dir_name, 'training.txt')
        elif mode == 'val':
            path = os.path.join(dir_name, 'validation.txt')
        else:
            path = os.path.join(dir_name, 'test.txt')

        sim_data, sim_length, sim_label = read_java_txt(path=path, flag=flag, trace_len=trace_len, dimension=3,
                                                        num_classes=num_class)
        num_traces = sim_data.shape[0]
        num_perclass = num_traces // num_class

        Ns = {}
        for key in ['NsND', 'NsCD', 'NsDM']:
            Ns[key] = np.ones(num_perclass, dtype=np.int8) * trace_len

        ##### Save datasets for CNN
        dataset, position = gen_trace(sim_data=sim_data, num_class=num_class, num_perclass=num_perclass,
                                      trace_len=trace_len)

        scipy.io.savemat(os.path.join(dir_name, f'{mode}.mat'), {f'{mode}set': dataset, f'{mode}label': position})

    end_time = time.time()
    print(' Total time consumed : {}s'.format(end_time - start_time))
