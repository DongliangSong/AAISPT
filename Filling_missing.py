# -*- coding: utf-8 -*-
# @Time    : 2024/5/15 19:57
# @Author  : Dongliang


import numpy as np
import pandas as pd


def fill_missing_values(data):
    """
    Define the fill function.
    :param data: Time series data.
    :return: Time series data after padding.
    """

    finals = []
    for ts in data:
        if ts.dtypes.dtype == 'object':
            # Converts a column in a DataFrame to a numeric type.
            df_numeric = ts.apply(pd.to_numeric, errors='coerce')

            # Converts a DataFrame to a NumPy array.
            ts = df_numeric.values

        for j in range(ts.shape[1]):  # For each dimension
            for i in range(ts.shape[0]):
                if np.isnan(ts[i, j]):
                    # Computes the index of the previous non-missing value.
                    prev_index = i - 1
                    while np.isnan(ts[prev_index, j]) and prev_index >= 0:
                        prev_index -= 1

                    # Computes the index of the next non-missing value.
                    next_index = i + 1
                    while np.isnan(ts[next_index, j]) and next_index < ts.shape[0]:
                        next_index += 1

                    # Use the mean value for padding.
                    if prev_index >= 0 and next_index < ts.shape[0]:
                        ts[i, j] = (ts[prev_index, j] + ts[next_index, j]) / 2
                    elif prev_index >= 0:
                        ts[i, j] = ts[prev_index, j]
                    elif next_index < ts.shape[0]:
                        ts[i, j] = ts[next_index, j]
        finals.append(ts)
    return finals
