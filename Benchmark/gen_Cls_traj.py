# -*- coding: utf-8 -*-
# @Time    : 2024/11/15 17:11
# @Author  : Dongliang

import os

import andi
import numpy as np
import scipy


def generate_tracks_classification(n, dim, min_T=30, max_T=500):
    """
    Generating a Trajectory Segmentation Dataset from the Andi Challenge.
    :param n: Number of the trajectory dataset.
    :param dim: Dimension of the trajectory dataset.
    :param min_T: Minimum trajectory length.
    :param max_T: Maximum trajectory length.
    """

    # Create tracks
    AD = andi.andi_datasets()
    X1, Y1, X2, Y2, X3, Y3 = AD.andi_dataset(N=n, min_T=min_T, max_T=max_T, tasks=[2], dimensions=[dim])
    classes = np.array(Y2[dim - 1]).astype(int)
    tracks = X2[dim - 1]

    # Package into array
    tracks_array = np.zeros([n, max_T, dim])
    if dim == 1:
        for i, t in enumerate(tracks):
            tracks_array[i, max_T - len(t):, 0] = t
    elif dim == 2:
        for i, t in enumerate(tracks):
            len_t = int(len(t) / 2)
            tracks_array[i, max_T - len_t:, 0] = t[:len_t]
            tracks_array[i, max_T - len_t:, 1] = t[len_t:]
    elif dim == 3:
        for i, t in enumerate(tracks):
            len_t = int(len(t) / 3)
            tracks_array[i, max_T - len_t:, 0] = t[:len_t]
            tracks_array[i, max_T - len_t:, 1] = t[len_t:len_t * 2]
            tracks_array[i, max_T - len_t:, 2] = t[len_t * 2:]

    return tracks_array, classes


def preprocess_tracks(tracks):
    """
    Preprocessing of trajectory datasets (Differential and Normalization).
    :param tracks: Simulated segmentation dataset.
    """

    dim = tracks.shape[2]
    if dim == 1:
        diff = np.diff(tracks[:, :, 0], axis=1)
        meanstep = np.sum(abs(diff), axis=1) / np.sum(tracks[:, :, 0] != 0, axis=1)
        return np.expand_dims(diff / np.expand_dims(meanstep, axis=-1), axis=-1)

    elif dim == 2:
        dx = np.diff(tracks[:, :, 0], axis=1)
        dy = np.diff(tracks[:, :, 1], axis=1)
        meanstep = np.expand_dims(np.sum(((dx ** 2) + (dy ** 2)) ** 0.5, axis=1) / np.sum(tracks[:, :, 0] != 0, axis=1),
                                  axis=-1)
        return np.dstack((dx / meanstep, dy / meanstep))

    elif dim == 3:
        dx = np.diff(tracks[:, :, 0], axis=1)
        dy = np.diff(tracks[:, :, 1], axis=1)
        dz = np.diff(tracks[:, :, 2], axis=1)
        meanstep = np.expand_dims(
            np.sum(((dx ** 2) + (dy ** 2) + (dz ** 2)) ** 0.5, axis=1) / np.sum(tracks[:, :, 0] != 0, axis=1), axis=-1)
        return np.dstack((dx / meanstep, dy / meanstep, dz / meanstep))


if __name__ == '__main__':
    min_len = 30
    max_len = 500

    paths = [
        r'.\data\CLS\Andi\2D',
        r'.\data\CLS\Andi\3D'
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
                N = 100
            elif mode == 'val' or mode == 'test':
                N = 20

            tracks, classes = generate_tracks_classification(n=N, dim=dim, min_T=min_len, max_T=max_len)

            savename = os.path.join(savepath, f'{mode}.mat')
            scipy.io.savemat(savename, {'data': tracks, 'label': classes})

            # # Trajectory preprocessing
            # norm_data = preprocess_tracks(tracks)
            # normname = os.path.join(savepath, f'norm_{mode}.mat')
            # scipy.io.savemat(normname, {'data': norm_data, 'label': classes})

            # Remove all zeros from the simulated trajectory.
            track_data = []
            for i in range(tracks.shape[0]):
                traj = tracks[i]
                filtered_traj = traj[traj[:, 0] != 0]
                if len(filtered_traj) > 0:
                    track_data.append(filtered_traj)
            scipy.io.savemat(os.path.join(path, f'raw{mode}.mat'), {'data': track_data, 'label': classes})
