# -*- coding: utf-8 -*-
# @Time    : 2024/11/15 17:08
# @Author  : Dongliang

import os

import andi
import numpy as np
import scipy


def generate_tracks_segmentation(n, dim):
    """
    Generating a Trajectory Segmentation Dataset from the Andi Challenge.
    :param n: Number of the trajectory dataset.
    :param dim: Dimension of the trajectory dataset.
    """

    # Create tracks
    AD = andi.andi_datasets()
    X1, Y1, X2, Y2, X3, Y3 = AD.andi_dataset(N=n, tasks=[3], dimensions=[dim])
    positions = np.array(Y3[dim - 1])[:, 1].astype(int) - 1
    tracks = X3[dim - 1]

    # Package into array
    tracks_array = np.zeros([n, 200, dim])
    if dim == 1:
        for i, t in enumerate(tracks):
            tracks_array[i, :, 0] = t

    elif dim == 2:
        for i, t in enumerate(tracks):
            len_t = int(len(t) / 2)
            tracks_array[i, :, 0] = t[:len_t]
            tracks_array[i, :, 1] = t[len_t:] - t[len_t]

    elif dim == 3:
        for i, t in enumerate(tracks):
            len_t = int(len(t) / 3)
            tracks_array[i, :, 0] = t[:len_t]
            tracks_array[i, :, 1] = t[len_t:2 * len_t] - t[len_t]
            tracks_array[i, :, 2] = t[2 * len_t:] - t[2 * len_t]

    return tracks_array, positions


def preprocess_tracks(X, dim, use_xyz_diff):
    """
    Preprocessing of trajectory datasets (Differential and Normalization).
    :param tracks: Simulated trajectory segmentation dataset.
    """

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
    stand_r = (r - mean_r) / std_r

    return stand_r


if __name__ == '__main__':
    paths = [
        r'.\data\SEG\Andi\2D',
        r'.\data\SEG\Andi\3D'
    ]
    for path in paths:
        savepath = path
        if not os.path.exists(savepath):
            os.makedirs(savepath)

        if '2D' in path:
            dim = 2
        elif '3D' in path:
            dim = 3

        for mode in ['train', 'val', 'test']:
            print(f"Generate ========== {mode} ========== dataset")
            if mode == 'train':
                N = 100000
            elif mode == 'val' or mode == 'test':
                N = 12500

            # Trajectory simulation
            tracks, position = generate_tracks_segmentation(n=N, dim=dim)

            savename = os.path.join(savepath, f'{mode}.mat')
            scipy.io.savemat(savename, {'data': tracks, 'label': position})

            # Trajectory preprocessing
            norm_data = preprocess_tracks(tracks, dim=dim, use_xyz_diff=True)
            normname = os.path.join(savepath, f'norm_{mode}.mat')
            scipy.io.savemat(normname, {'data': norm_data, 'label': position})
