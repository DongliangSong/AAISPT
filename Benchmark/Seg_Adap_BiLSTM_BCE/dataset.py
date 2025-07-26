# -*- coding: utf-8 -*-
# @Time    : 2024/6/27 16:30
# @Author  : Dongliang


import numpy as np

import torch
from torch.utils.data import Dataset


def label_convert(label, seq_len, FWHM):
    """
    Convert each trajectory's change point to a Gaussian distribution with a specific half-peak width.

    :param label: The change point of each track.
    :param seq_len: The length of the trajectory.
    :param FWHM: The full-width at half-maximum of the Gaussian distribution.
    :return: The transformed labelling matrix.
    """

    def gaussian(x, mu, sigma):
        return np.exp(- (x - mu) ** 2 / (2 * sigma ** 2))

    # Set parameters
    sigma = FWHM / (2 * np.sqrt(2 * np.log(2)))
    x = np.linspace(0, seq_len - 1, seq_len)
    x = x[np.newaxis, :]  # Shape: (1, seq_len)
    labels = np.array(label)[:, np.newaxis]  # Shape: (num_labels, 1)
    y = gaussian(x, labels, sigma)
    return y


class TSDataset(Dataset):
    def __init__(self, data_dict, label_dict):
        super(TSDataset, self).__init__()
        self.data = {key: np.array(val, dtype=np.float32) for key, val in data_dict.items()}
        self.labels = {key: np.array(val, dtype=np.float32) for key, val in label_dict.items()}
        self.keys = list(self.data.keys())  # Store keys for iteration

    def __getitem__(self, idx):
        result = []
        for key in self.keys:
            result.append(torch.tensor(self.data[key][idx], dtype=torch.float32))
            result.append(torch.tensor(self.labels[key][idx], dtype=torch.float32))
        return tuple(result)

    def __len__(self):
        return len(next(iter(self.data.values())))
