# -*- coding: utf-8 -*-
# @Time    : 2025/7/25 17:39
# @Author  : Dongliang

import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class TrajectoryScaler:
    def __init__(self):
        """
        Initializes the TrajectoryScaler to store scalers and statistics for each task.
        """
        self.scalers = []  # Stores StandardScaler for each task.
        self.means = []  # Stores mean for each task.
        self.stds = []  # Stores standard deviation for each task.

    def fit(self, datasets):
        """
        Fits a StandardScaler to each dataset and stores the mean and standard deviation.

        :param datasets: List of variable-length sequence data for each task, where each task is a list of 2D tensors.
        """
        if not datasets or not all(isinstance(task, list) and task for task in datasets):
            raise ValueError('Each task in datasets must be a non-empty list of tensors')

        self.scalers = []
        self.means = []
        self.stds = []
        for task_id, X_list in enumerate(datasets):
            if not all(isinstance(x, torch.Tensor) and x.ndim == 2 for x in X_list):
                raise ValueError('Each item must be a 2D torch.Tensor')
            feature_dim = X_list[0].shape[1]
            if not all(x.shape[1] == feature_dim for x in X_list):
                raise ValueError('All tensors in a task must have the same feature dimension')

            # Flatten all sequences into (num_samples, feature_dim).
            X_flat = torch.cat([x.view(-1, feature_dim) for x in X_list], dim=0).numpy()

            # Standardization
            scaler = StandardScaler()
            scaler.fit(X_flat)

            # Store scaler and statistics
            self.scalers.append(scaler)
            self.means.append(scaler.mean_)
            self.stds.append(scaler.scale_)

    def transform(self, datasets, task_id=None):
        """
        Standardizes the input dataset(s) using the fitted scalers.

        :param datasets: List of variable-length sequence data, either for a single task or multiple tasks.
        :param task_id: Optional integer specifying the task ID for single-task standardization; if None, standardizes all tasks.
        :return: List of standardized tensors for a single task or a list of lists for multiple tasks.
        """

        # Add judgment: If the data is for a single task or there is only one task, task_id must be specified.
        is_single_task_input = isinstance(datasets, list) and all(isinstance(x, torch.Tensor) for x in datasets)
        is_single_task_dataset = isinstance(datasets, list) and len(datasets) == 1 and isinstance(datasets[0], list)
        if (is_single_task_input or is_single_task_dataset) and task_id is None:
            raise ValueError('task_id must be specified for single-task data or datasets with only one task')

        if task_id is not None:
            # Standardize data for a specific task
            if task_id >= len(self.scalers):
                raise ValueError(f'task_id {task_id} exceeds number of fitted tasks {len(self.scalers)}')
            X_list = datasets
            if not isinstance(X_list, list) or not all(isinstance(x, torch.Tensor) and x.ndim == 2 for x in X_list):
                raise ValueError('Input must be a list of 2D torch.Tensor for single task')

            feature_dim = X_list[0].shape[1]
            X_flat = torch.cat([x.view(-1, feature_dim) for x in X_list], dim=0).numpy()
            X_scaled_flat = self.scalers[task_id].transform(X_flat)
            start = 0
            X_scaled_list = []
            for x in X_list:
                length = x.shape[0]
                X_scaled_list.append(torch.tensor(X_scaled_flat[start:start + length]).view(length, feature_dim))
                start += length
            return X_scaled_list

        else:
            # Standardize data for all tasks
            if not isinstance(datasets, list) or not all(isinstance(task, list) for task in datasets):
                raise ValueError('Input must be a List[List[torch.Tensor]] for multi-task')
            X_scaled_datasets = []
            for task_id, X_list in enumerate(datasets):
                X_scaled_list = self.transform(X_list, task_id)
                X_scaled_datasets.append(X_scaled_list)
            return X_scaled_datasets

    def get_params(self):
        """
        Retrieves the mean and standard deviation for each task.

        :return: Tuple of lists containing means and standard deviations for each task.
        """
        return self.means, self.stds


# Customize Dataset to handle variable-length sequences
class VariableLengthDataset(Dataset):
    def __init__(self, X_list, y):
        self.X_list = X_list
        self.y = y

    def __len__(self):
        return len(self.X_list)

    def __getitem__(self, idx):
        return self.X_list[idx], self.y[idx]


# Customize collate_fn to handle variable-length sequences
def collate_fn(batch):
    X_batch, y_batch = zip(*batch)
    lengths = [x.shape[0] for x in X_batch]
    max_len = max(lengths)
    feature_dim = X_batch[0].shape[1]

    X_padded = torch.zeros(len(X_batch), max_len, feature_dim)
    mask = torch.ones(len(X_batch), max_len, dtype=torch.bool)
    for i, x in enumerate(X_batch):
        length = x.shape[0]
        X_padded[i, :length] = x
        mask[i, :length] = 0

    y_batch = torch.tensor(y_batch)
    return X_padded, y_batch, mask, lengths
