# -*- coding: utf-8 -*-
# @Time    : 2024/3/22 20:55
# @Author  : Dongliang


import json
import pickle

import numpy as np
import pandas as pd
import scipy


def fill_missing_values(data):
    """
    Fill in missing values in the experimental data.

    :param data: Raw time series data.
    :return: Time series data after filling.
    """

    finals = []
    for timeseries in data:
        if timeseries.dtypes.dtype == 'object':
            # Converts a column in a DataFrame to a numeric type.
            df_numeric = timeseries.apply(pd.to_numeric, errors='coerce')

            # Converts a DataFrame to a NumPy array.
            timeseries = df_numeric.values

        for j in range(timeseries.shape[1]):  # For each dimension
            for i in range(timeseries.shape[0]):
                if np.isnan(timeseries[i, j]):
                    # Computes the index of the previous non-missing value.
                    prev_index = i - 1
                    while np.isnan(timeseries[prev_index, j]) and prev_index >= 0:
                        prev_index -= 1

                    # Computes the index of the next non-missing value.
                    next_index = i + 1
                    while np.isnan(timeseries[next_index, j]) and next_index < timeseries.shape[0]:
                        next_index += 1

                    # Use the mean value for padding.
                    if prev_index >= 0 and next_index < timeseries.shape[0]:
                        timeseries[i, j] = (timeseries[prev_index, j] + timeseries[next_index, j]) / 2
                    elif prev_index >= 0:
                        timeseries[i, j] = timeseries[prev_index, j]
                    elif next_index < timeseries.shape[0]:
                        timeseries[i, j] = timeseries[next_index, j]
        finals.append(timeseries)
    return finals


def read_files(file_paths):
    """
    Read particle trajectories in various formats.

    :param file_paths: Particle trajectory files.
    :return:  A list or dict of particle trajectories.
    """
    if isinstance(file_paths, str):
        file_paths = [file_paths]  # Convert single file path to list

    result = []
    for file_path in file_paths:
        file_extension = file_path.split('.')[-1]
        if file_extension == 'csv':
            data = pd.read_csv(file_path, header=None)

        elif file_extension == 'xlsx':
            data = pd.read_excel(file_path, header=None)

        elif file_extension == 'txt':
            data = pd.read_csv(file_path, sep='\t')

        elif file_extension == 'mat':
            data = scipy.io.loadmat(file_path)

        elif file_extension == 'json':
            with open(file_path, 'r') as f:
                data = json.load(f)

        elif file_extension == 'pkl':
            with open(file_path, 'rb') as f:
                data = pickle.load(f)

        else:
            raise ValueError(f"Unsupported file format: {file_extension}")

        # Convert data to list format
        if isinstance(data, pd.DataFrame):
            result.append(data.values.tolist())
        elif isinstance(data, dict):
            result = {}
            for key in data.keys():
                if key not in ["__header__", "__version__", "__globals__"]:
                    result[key] = data[key]
        elif isinstance(data, (list, tuple)):
            result = data
        else:
            raise TypeError("Unsupported data type.")

    return result


def save_files(variable, file_path):
    """
    Saves the variable to a file of the specified type.
    """
    file_extension = file_path.split('.')[-1]
    if file_extension == 'csv':
        if isinstance(variable, pd.DataFrame):
            variable.to_csv(file_path, index=False)
        elif isinstance(variable, np.ndarray):
            pd.DataFrame(variable).to_csv(file_path, index=False, header=False)
        else:
            raise TypeError("Unsupported variable type for CSV format.")

    elif file_extension == 'xlsx':
        if isinstance(variable, pd.DataFrame):
            variable.to_excel(file_path, index=False)
        else:
            raise TypeError("Unsupported variable type for Excel format.")

    elif file_extension == 'txt':
        if isinstance(variable, list):
            with open(file_path, 'w') as f:
                for item in variable:
                    f.write("%s\n" % item)
        else:
            raise TypeError("Unsupported variable type for text format.")

    elif file_extension == 'pkl':
        with open(file_path, 'wb') as f:
            pickle.dump(variable, f)

    elif file_extension == 'json':
        with open(file_path, 'w') as f:
            json.dump(variable, f)

    else:
        raise ValueError("Unsupported file format.")
