# -*- coding: utf-8 -*-
# @Time    : 2025/6/19 22:27
# @Author  : Dongliang

import os

import numpy as np
import scipy
from scipy.signal import find_peaks
from scipy.special import expit


def temperature_scaled_sigmoid(logit, T=1.0):
    """
    Apply a temperature-scaled sigmoid function to the input logits.

    :param logit: Input logits to be transformed.
    :param T: Temperature parameter to scale the logits. Defaults to 1.0.
    :return: np.ndarray: Sigmoid-transformed probabilities after temperature scaling.
    """
    return expit(logit / T)


def inverse_sigmoid(prob, eps=1e-8):
    """
    Compute the inverse sigmoid (logit) function for given probabilities.

    :param prob: Input probabilities to be transformed.
    :param eps: Small value to clip probabilities and avoid numerical issues. Defaults to 1e-8.
    :return: np.ndarray: Logits computed from the input probabilities.
    """
    prob = np.clip(prob, eps, 1 - eps)
    logit = np.log(prob / (1 - prob))
    return logit


if __name__ == '__main__':
    temperature = 4    # temperature scaling factor
    prominence = 0.07

    data_path = r'..\data\Exp_data_for_example'
    savepath = data_path
    if not os.path.exists(savepath):
        os.mkdir(savepath)

    prob = scipy.io.loadmat(os.path.join(data_path, 'Adap_CPs.mat'))['prob'].squeeze()
    logit = inverse_sigmoid(prob)
    prob = temperature_scaled_sigmoid(logit, T=temperature)

    length = len(prob)
    x = np.arange(length)
    x_min, x_max = 10, length - 10
    mask = (x >= x_min) & (x <= x_max)
    y_roi = prob[mask]

    # Find peaks
    peak, _ = find_peaks(y_roi, prominence=prominence)
    loc = peak + 10
    print(loc)

    scipy.io.savemat(os.path.join(data_path, 'new_loc.mat'), {'loc': loc})
