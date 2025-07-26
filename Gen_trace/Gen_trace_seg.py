# -*- coding: utf-8 -*-
# @Time    : 2024/8/6 20:30
# @Author  : Dongliang

"""
    Generate simulated trajectories for trajectory segmentation
"""

import os
import time

import scipy

from AAISPT.Gen_trace.BasicModule import *
from AAISPT.Gen_trace.read_traces import read_java_txt

# Set rotation parameter
# Normal diffusion
step_polar_meanND, step_polar_stdND = 17.4, 21
step_azimuth_meanND, step_azimuth_stdND = 72.4, 84.5

# Directed diffusion
step_polar_meanDM, step_polar_stdDM = 4.0, 5.5
step_azimuth_meanDM, step_azimuth_stdDM = 4.9, 7.0

# Directed motion with fast rotation
step_polar_meanDMR, step_polar_stdDMR = 8, 11
step_azimuth_meanDMR, step_azimuth_stdDMR = 10, 14

# Tight attachment
step_polar_meanTA, step_polar_stdTA = 4.3, 5.7
step_azimuth_meanTA, step_azimuth_stdTA = 4.9, 6.7

# Tethered rotation
step_polar_meanTR, step_polar_stdTR = 7.0, 9.7
step_azimuth_meanTR, step_azimuth_stdTR = 12.5, 22.5


class Gen_3_segdata():
    """
    Generate simulated trajectories for trajectory segmentation. For 2/3-D data.
    """

    def gen_segdata(self, flag, sim_data):
        """
        Generate simulated trajectories for trajectory segmentation.

        :param flag: Signs of different diffusion patterns. 0 for ND, 1 for CD, 2 for DM.
        :param sim_data: The simulated trajectories of different types.
        :return: Generate diffusion trajectories (XY or XYZ).
        """
        num_traces = sim_data.shape[0]
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

    def gen_trace(self, sim_data, num_class, num_perclass, trace_len):
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
                    traj1_diff = self.gen_segdata(i, sim_data=sim_data)
                    traj2_diff = self.gen_segdata(j, sim_data=sim_data)

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


class Gen_5_segdata():
    """
    Generate simulated data for trajectory segmentation. For 4/5-D data.
    """

    def gen_segdata(self, flag, sim_data, Ns):
        """
        Generate simulated trajectories for trajectory segmentation.

        :param flag: Signs of different diffusion patterns. 0 for ND, 1 for TA, 2 for TR, 3 for DMR, 4 for DM.
        :param sim_data: The simulated trajectories of different types.
        :param Ns: The length of each trace stored in dictionary form.
        :return: Generate diffusion trajectories (XY or XYZ), step azimuth and polar angles.
        """
        num_traces = sim_data.shape[0]
        num_perclass = num_traces // num_class
        if flag == 0:
            # load normal diffusion (ND)
            diff = sim_data[0:num_perclass * 1]

            # Generate normal diffusion rotation information
            step_polars, step_azimuths = gen_ND_diff(
                step_polar_meanND=step_polar_meanND,
                step_polar_stdND=step_polar_stdND,
                step_azimuth_meanND=step_azimuth_meanND,
                step_azimuth_stdND=step_azimuth_stdND,
                num_perclass=num_perclass,
                trace_len=trace_len,
                NsND=Ns['NsND']
            )

        elif flag == 1:
            # load tight attachment (TA)
            diff = sim_data[num_perclass * 1:num_perclass * 2]

            # Generating tight attachment diffusion rotation information
            step_polars, step_azimuths = gen_TA_diff(
                step_polar_meanTA=step_polar_meanTA,
                step_polar_stdTA=step_polar_stdTA,
                step_azimuth_meanTA=step_azimuth_meanTA,
                step_azimuth_stdTA=step_azimuth_stdTA,
                num_perclass=num_perclass,
                trace_len=trace_len,
                NsTA=Ns['NsTA']
            )

        elif flag == 2:
            # load tethered rotation (TR)
            diff = sim_data[num_perclass * 2:num_perclass * 3]

            # Generate tethered rotation diffusion
            step_polars, step_azimuths = gen_TR_diff(
                step_polar_meanTR=step_polar_meanTR,
                step_polar_stdTR=step_polar_stdTR,
                step_azimuth_meanTR=step_azimuth_meanTR,
                step_azimuth_stdTR=step_azimuth_stdTR,
                num_perclass=num_perclass,
                trace_len=trace_len,
                NsTR=Ns['NsTR']
            )

        elif flag == 3:
            # load directed motion with fast rotation (DMR)
            diff = sim_data[num_perclass * 4:]

            # Generate directed motion with fast rotation information
            step_polars, step_azimuths = gen_DMR_diff(
                step_polar_meanDMR=step_polar_meanDMR,
                step_polar_stdDMR=step_polar_stdDMR,
                step_azimuth_meanDMR=step_azimuth_meanDMR,
                step_azimuth_stdDMR=step_azimuth_stdDMR,
                num_perclass=num_perclass,
                trace_len=trace_len,
                NsDMR=Ns['NsDMR']
            )

        elif flag == 4:
            # load directed motion (DM)
            diff = sim_data[num_perclass * 3:num_perclass * 4]

            # Generate directed motion rotation information
            step_polars, step_azimuths = gen_DM_diff(
                step_polar_meanDM=step_polar_meanDM,
                step_polar_stdDM=step_polar_stdDM,
                step_azimuth_meanDM=step_azimuth_meanDM,
                step_azimuth_stdDM=step_azimuth_stdDM,
                num_perclass=num_perclass,
                trace_len=trace_len,
                NsDM=Ns['NsDM']
            )

        return diff, step_polars, step_azimuths

    def gen_trace(self, sim_data, num_class, Ns, num_perclass, trace_len):
        """
        The different types of trajectories were concatenated to generate simulated trajectories for the segmentation task.

        :param sim_data: The simulated data of each type of trajectory.
        :param num_class: The number of trajectory types.
        :param Ns: The number of trajectories of each type.
        :param num_perclass: The number of trajectories of each type.
        :param trace_len: The length of each trajectory.
        :return: Generate a dataset containing mixed trajectories and the change point position for each trajectory.
        """
        dataset = []
        position = []

        for i in range(num_class):
            for j in range(num_class):
                if i != j:
                    traj1_diff, traj1_steppolar, traj1_stepazimuth = self.gen_segdata(i, sim_data=sim_data, Ns=Ns)
                    traj2_diff, traj2_steppolar, traj2_stepazimuth = self.gen_segdata(j, sim_data=sim_data, Ns=Ns)

                    change_point = np.random.randint(30, trace_len - 30, num_perclass)

                    traces = np.array([np.vstack((
                        traj1_diff[k, :change_point[k]],
                        traj2_diff[k, :(trace_len - change_point[k])] + traj1_diff[k, change_point[k]])) for k in
                        range(num_perclass)], dtype=np.float32)

                    stpolars = np.array([np.hstack((
                        traj1_steppolar[k, :change_point[k]],
                        traj2_steppolar[k, :(trace_len - change_point[k])])) for k in range(num_perclass)],
                        dtype=np.float32)

                    stazis = np.array([np.hstack((
                        traj1_stepazimuth[k, :change_point[k]],
                        traj2_stepazimuth[k, :(trace_len - change_point[k])])) for k in range(num_perclass)],
                        dtype=np.float32)

                    features = np.concatenate(
                        (traces, np.expand_dims(stpolars, axis=-1), np.expand_dims(stazis, axis=-1)),
                        axis=-1)
                    dataset.extend(features)
                    position.extend(change_point)
                    print('first segment in {} class,second segment in {} class'.format(i, j))

        position = np.array(position).reshape(-1, 1)
        return np.array(dataset), position


def main_3_class(dir_name, flag, trace_len, num_class):
    """
    For 2/3-D trajectory data, perform the dataset generation step.

    :param dir_name: Data file load and save paths.
    :param flag: Trajectory type, e.g. fixed length or variable length.
    :param trace_len: The length of the timeseries data.
    :param num_class: The number of trajectory types.
    """
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
        gen_3data = Gen_3_segdata()
        dataset, position = gen_3data.gen_trace(
            sim_data=sim_data,
            num_class=num_class,
            num_perclass=num_perclass,
            trace_len=trace_len
        )

        scipy.io.savemat(os.path.join(dir_name,f'{mode}.mat'), {f'{mode}set': dataset, f'{mode}label': position})


def main_5_class(dir_name, flag, trace_len, num_class):
    """
    For 4/5-D trajectory data, perform the dataset generation step.

    :param dir_name: Data file load and save paths.
    :param flag: Trajectory type, e.g. fixed length or variable length.
    :param trace_len: The length of the timeseries data.
    :param num_class: The number of trajectory types.
    """
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
        for key in ['NsND', 'NsTA', 'NsTR', 'NsDMR', 'NsDM']:
            Ns[key] = np.ones(num_perclass, dtype=np.int8) * trace_len

        ##### Save datasets for CNN
        gen_5data = Gen_5_segdata()
        dataset, position = gen_5data.gen_trace(
            sim_data=sim_data,
            num_class=num_class,
            num_perclass=num_perclass,
            trace_len=trace_len,
            Ns=Ns
        )

        scipy.io.savemat(os.path.join(dir_name, f'{mode}.mat'), {f'{mode}set': dataset, f'{mode}label': position})


if __name__ == '__main__':
    start_time = time.time()

    root = r'D:\TrajSeg-Cls\TrajSEG-CLS_V3\SEG\Second batch'
    paths = [
        os.path.join(root, '2D\SNR03'),
        os.path.join(root, '3D\SNR03'),
        os.path.join(root, '4D\SNR03'),
        os.path.join(root, '5D\SNR03')
    ]

    trace_len = 201
    flag = 'Fixed length trajectory'

    for dir_name in paths:
        if '2D' in dir_name or '3D' in dir_name:
            num_class = 3
            main_3_class(dir_name=dir_name, flag=flag, trace_len=trace_len, num_class=num_class)

        elif '4D' in dir_name or '5D' in dir_name:
            num_class = 5
            main_5_class(dir_name=dir_name, flag=flag, trace_len=trace_len, num_class=num_class)

    end_time = time.time()
    print(' Total time consumed : {}s'.format(end_time - start_time))
