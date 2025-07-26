# -*- coding: utf-8 -*-
# @Time    : 2024/2/26 11:03
# @Author  : Dongliang

"""
    Generate 5D trajectories for classification.
    x,y,z,step polar,step azimuth
"""

import os
import time

import scipy

from AAISPT.Gen_trace.BasicModule import *
from AAISPT.Gen_trace.read_traces import read_varlen

# Set rotation parameter
# Normal Diffusion
step_polar_meanND, step_polar_stdND = 17.4, 21
step_azimuth_meanND, step_azimuth_stdND = 72.4, 84.5

# # Normal Diffusion + nRotation
# step_polar_meanNDnR, step_polar_stdNDnR = 7.4, 10
# step_azimuth_meanNDnR, step_azimuth_stdNDnR = 12.4, 24.5

# Tight Attachment
step_polar_meanTA, step_polar_stdTA = 4.3, 5.7
step_azimuth_meanTA, step_azimuth_stdTA = 4.9, 6.7

# Tethered Rotation
step_polar_meanTR, step_polar_stdTR = 7.0, 9.7
step_azimuth_meanTR, step_azimuth_stdTR = 12.5, 22.5

# Directed Motion + fast Rotation
step_polar_meanDMR, step_polar_stdDMR = 8, 11
step_azimuth_meanDMR, step_azimuth_stdDMR = 10, 14

# Directed Motion
step_polar_meanDM, step_polar_stdDM = 4.0, 5.5
step_azimuth_meanDM, step_azimuth_stdDM = 4.9, 7.0


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


class Gen_5_clsdata():
    def gen_rotation_data(self, num_class, num_perclass, trace_len, Ns):
        """
        Generate rotational step angle simulation data.

        :param num_class: The number of trajectory types.
        :param num_perclass: The number of trajectories of each type.
        :param trace_len: The length of the trajectory.
        :param Ns: The length of each trace stored in dictionary form.
        :return: A simulated dataset containing various diffuse rotational step angles
        """

        # Generate normal diffusion rotation information
        NDstep_polars, NDstep_azimuths = gen_ND_diff(
            step_polar_meanND=step_polar_meanND,
            step_polar_stdND=step_polar_stdND,
            step_azimuth_meanND=step_azimuth_meanND,
            step_azimuth_stdND=step_azimuth_stdND,
            num_perclass=num_perclass,
            trace_len=trace_len,
            NsND=Ns['NsND']
        )

        # Generating tight attachment diffusion rotation information
        TAstep_polars, TAstep_azimuths = gen_TA_diff(
            step_polar_meanTA=step_polar_meanTA,
            step_polar_stdTA=step_polar_stdTA,
            step_azimuth_meanTA=step_azimuth_meanTA,
            step_azimuth_stdTA=step_azimuth_stdTA,
            num_perclass=num_perclass,
            trace_len=trace_len,
            NsTA=Ns['NsTA']
        )

        # Generate tethered rotation diffusion
        TRstep_polars, TRstep_azimuths = gen_TR_diff(
            step_polar_meanTR=step_polar_meanTR,
            step_polar_stdTR=step_polar_stdTR,
            step_azimuth_meanTR=step_azimuth_meanTR,
            step_azimuth_stdTR=step_azimuth_stdTR,
            num_perclass=num_perclass,
            trace_len=trace_len,
            NsTR=Ns['NsTR']
        )

        # Generate directed diffusion and fast rotation information
        DMRstep_polars, DMRstep_azimuths = gen_DMR_diff(
            step_polar_meanDMR=step_polar_meanDMR,
            step_polar_stdDMR=step_polar_stdDMR,
            step_azimuth_meanDMR=step_azimuth_meanDMR,
            step_azimuth_stdDMR=step_azimuth_stdDMR,
            num_perclass=num_perclass,
            trace_len=trace_len,
            NsDMR=Ns['NsDMR']
        )

        # Generate directed diffusion rotation information
        DMstep_polars, DMstep_azimuths = gen_DM_diff(
            step_polar_meanDM=step_polar_meanDM,
            step_polar_stdDM=step_polar_stdDM,
            step_azimuth_meanDM=step_azimuth_meanDM,
            step_azimuth_stdDM=step_azimuth_stdDM,
            num_perclass=num_perclass,
            trace_len=trace_len,
            NsDM=Ns['NsDM']
        )

        polars = np.concatenate((NDstep_polars,
                                 TAstep_polars,
                                 TRstep_polars,
                                 DMRstep_polars,
                                 DMstep_polars), axis=0)

        azimuths = np.concatenate((NDstep_azimuths,
                                   TAstep_azimuths,
                                   TRstep_azimuths,
                                   DMRstep_azimuths,
                                   DMstep_azimuths), axis=0)

        return polars.reshape((num_perclass * num_class, trace_len, 1)), \
               azimuths.reshape((num_perclass * num_class, trace_len, 1))


def main_3_class(dir_name, dimension, num_class, trace_len):
    """
    For 2/3-D trajectory data, perform the dataset generation step.

    :param dir_name: Data file load and save paths.
    :param dimension: The dimension of the translation coordinate of the trajectory.
    :param num_class: The number of trajectory types.
    :param trace_len: The length of the timeseries data.
    """
    for mode in ['train', 'val', 'test']:
        if mode == 'train':
            path = os.path.join(dir_name, 'training.txt')
        elif mode == 'val':
            path = os.path.join(dir_name, 'validation.txt')
        else:
            path = os.path.join(dir_name, 'test.txt')

        # Load the generated motion trajectory.
        sim_data, sim_length, sim_label = read_varlen(path=path, dimension=dimension, num_class=num_class)
        num_traces = len(sim_data)
        num_perclass = num_traces // num_class

        Ns = {}
        for i, key in enumerate(['NsND', 'NsCD', 'NsDM']):
            Ns[key] = np.ones(num_perclass, dtype=np.int8) * trace_len

        dataset = []
        for i in range(num_traces):
            xyz = sim_data[i]
            length = xyz.shape[0]
            dataset.append(xyz)

        label = np.concatenate([i * np.ones((num_perclass, 1)) for i in range(num_class)])

        # Save data
        scipy.io.savemat(os.path.join(dir_name, f'{mode}.mat'), {f'{mode}set': dataset, f'{mode}label': label})


def main_5_class(dir_name, dimension, num_class, trace_len):
    """
    For 4/5-D trajectory data, perform the dataset generation step.

    :param dir_name: Data file load and save paths.
    :param dimension: The dimension of the translation coordinate of the trajectory.
    :param num_class: The number of trajectory types.
    :param trace_len: The length of the timeseries data.
    """
    for mode in ['train', 'val', 'test']:
        if mode == 'train':
            path = os.path.join(dir_name, 'training.txt')
        elif mode == 'val':
            path = os.path.join(dir_name, 'validation.txt')
        else:
            path = os.path.join(dir_name, 'test.txt')

        # Load the generated motion trajectory.
        sim_data, sim_length, sim_label = read_varlen(path=path, dimension=dimension, num_class=num_class)
        num_traces = len(sim_data)
        num_perclass = num_traces // num_class

        Ns = {}
        for i, key in enumerate(['NsND', 'NsTA', 'NsTR', 'NsDMR', 'NsDM']):
            Ns[key] = np.ones(num_perclass, dtype=np.int8) * trace_len

        # Generate rotational step angle simulation data.
        gen_5_data = Gen_5_clsdata()
        polars, azimuths = gen_5_data.gen_rotation_data(
            num_class=num_class,
            num_perclass=num_perclass,
            trace_len=trace_len,
            Ns=Ns
        )

        dataset = []
        for i in range(num_traces):
            xyz = sim_data[i]
            length = xyz.shape[0]
            dataset.append(np.concatenate((xyz, polars[i, :length], azimuths[i, :length]), axis=1))

        label = np.concatenate([i * np.ones((num_perclass, 1)) for i in range(num_class)])

        # Save data
        scipy.io.savemat(os.path.join(dir_name, f'{mode}.mat'), {f'{mode}set': dataset, f'{mode}label': label})


if __name__ == '__main__':
    start = time.time()
    trace_len = 500
    num_class = 5
    dimension = 3

    dir_name = r'D:\TrajSeg-Cls\TrajSEG-CLS_V3\CLS\Var_L_500\SNR05'

    if '2D' in dir_name or '3D' in dir_name:
        num_class = 3
        main_3_class(dir_name=dir_name, dimension=dimension, num_class=num_class, trace_len=trace_len)

    else:
        num_class = 5
        main_5_class(dir_name=dir_name, dimension=dimension, num_class=num_class, trace_len=trace_len)

    end = time.time()
    print('Time taken: {:.2f}s'.format(end - start))
