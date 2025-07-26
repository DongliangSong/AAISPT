# -*- coding: utf-8 -*-
# @Time    : 2024/8/6 21:28
# @Author  : Dongliang

import os

import scipy

from AAISPT.Gen_trace.BasicModule import *
from AAISPT.Gen_trace.read_traces import read_varlen


def adjust_length(traces, ini_frames):
    """
    Adjusts the given trace dataset "traces" to the specified length "ini_frames"

    :param traces: A simulated trajectory dataset.
    :param ini_frames: The initial length of the trajectory.
    :return: The trajectory dataset after adjusting the length.
    """
    n = len(traces)
    final_trace = np.zeros((n, ini_frames, 2), dtype='float32')
    for index, trace in enumerate(traces):
        trace_ = trace.reshape((trace.shape[0], trace.shape[1]))
        if trace_.shape[0] == ini_frames:
            final_trace[index] = trace
        else:
            c = [0, 0] * np.ones((ini_frames - trace_.shape[0], 1), dtype='float32')
            final_trace[index] = np.vstack((c, trace))
    return final_trace


if __name__ == '__main__':
    root = r'D:\TrajSeg-Cls\TrajSEG-CLS_V3\CLS\Var_LD\3D\SNR03'

    if '2D' in root or '3D' in root:
        num_class = 3
    elif '4D' in root or '5D' in root:
        num_class = 5

    if '2D' in root or '4D' in root:
        target_dim = 2
    elif '3D' in root or '5D' in root:
        target_dim = 3

    for mode in ['train', 'val', 'test']:
        if mode == 'train':
            path = os.path.join(root, 'training.txt')
        elif mode == 'val':
            path = os.path.join(root, 'validation.txt')
        else:
            path = os.path.join(root, 'test.txt')

        # Load the generated motion trajectory.
        sim_data, sim_length, sim_label = read_varlen(path=path, dimension=3, num_class=num_class)
        num_traces = len(sim_data)
        num_perclass = num_traces // num_class

        dataset = []
        for i in range(num_traces):
            xyz = sim_data[i][:, :target_dim]
            dataset.append(xyz)

        label = np.concatenate([i * np.ones((num_perclass, 1)) for i in range(num_class)])

        # Save data
        scipy.io.savemat(os.path.join(root, f'{mode}.mat'), {f'{mode}set': dataset, f'{mode}label': label})
