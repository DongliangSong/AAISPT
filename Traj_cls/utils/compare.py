# -*- coding: utf-8 -*-
# @Time    : 2024/7/17 21:45
# @Author  : Dongliang

import numpy as np
import scipy


def data_prepare(X, N, traj_length, dimension):
    import numpy as np
    thr = 1e-10
    r = np.transpose(X, axes=[0, 2, 1])
    r = np.diff(r, axis=2)
    x = np.zeros((N, 0))
    for dim in range(dimension):
        y = r[:, dim, :]
        sy = np.std(y, axis=1)
        y = (y - np.mean(y, axis=1).reshape(len(y), 1)) / np.where(sy > thr, sy, 1).reshape(len(y), 1)
        y = np.concatenate((y, np.zeros((N, 1))), axis=1)
        x = np.concatenate((x, y), axis=1)
    x = np.transpose(x.reshape((N, dimension, traj_length)), axes=[0, 2, 1])

    return x


def data_standard(X):
    thr = 1e-10
    N, trace_len, dimension = X.shape

    # Calculate the difference between adjacent time steps.
    r = np.diff(X, axis=1)

    # Calculate mean and std along axis 1, keeping dims
    mean_r = np.mean(r, axis=1, keepdims=True)
    std_r = np.std(r, axis=1, keepdims=True)
    std_r[std_r < thr] = 1

    # Standardization
    norm_r = (r - mean_r) / std_r
    zero_pad = np.zeros((N, 1, dimension), dtype=np.float32)
    norm_r = np.concatenate((zero_pad, norm_r), axis=1)

    return norm_r


if __name__ == '__main__':
    a = np.random.rand(10, 10, 3)
    b = data_prepare(a, 10, 10, 3)
    c = data_standard(a)

    scipy.io.savemat('a.mat', {'a': a, 'b': b, 'c': c, })
