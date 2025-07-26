# -*- coding: utf-8 -*-
# @Time    : 2025/4/3 11:16
# @Author  : Dongliang

import json
import os

import numpy as np
import scipy
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from AAISPT.Traj_cls.AdapTransformer.dataset import TrajectoryScaler, VariableLengthDataset, collate_fn
from AAISPT.Traj_cls.AdapTransformer.model import MultiTaskTransformer
from AAISPT.Traj_cls.AdapTransformer.train_val import test_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set parameters
task_id = 0
feature_dims = [32]
num_classes_list = [5]
batch_size = 2048

# Hyperparameters (Fixed)
d_model = 64
num_layers = 2
num_heads = 4
hidden_dim = 128

# Load dataset
data_path = r'..\data\CLS\Andi\3D\Feature_based'
test_file = os.path.join(data_path, 'test_feature.mat')

# Load trained model
model_path = r'..\data\CLS\Andi\3D\AdapTransformer'
savepath = os.path.join(model_path, r'Evaluation')
if not os.path.exists(savepath):
    os.makedirs(savepath)

# Load test dataset
test_mat = scipy.io.loadmat(test_file)
test_X_data = test_mat['data'].squeeze()
test_y_data = test_mat['label'].flatten()
test_X_list = [torch.tensor(x, dtype=torch.float32) for x in test_X_data]
test_y = torch.tensor(test_y_data, dtype=torch.long)

# # Determine the task ID based on the feature dimension.
# test_feature_dim = test_X_list[0].shape[1]
# task_id = None
# for i, dim in enumerate(feature_dims):
#     if test_feature_dim == dim:
#         task_id = i
#         break
# if task_id is None:
#     raise ValueError(f"Test feature dimension {test_feature_dim} does not match any task: {feature_dims}")
#
# print(f"Test data belongs to Task {task_id} (feature_dim={test_feature_dim})")


# Load the standardized parameters (mean and standard deviation).
scaler_params_path = os.path.join(model_path, 'train_param.json')
with open(scaler_params_path, 'r', encoding='utf-8') as f:
    scaler_params = json.load(f)

# Create a TrajectoryScaler and set the parameters.
scaler = TrajectoryScaler()
scaler.scalers = []
for task in scaler_params['tasks']:
    scaler_task = StandardScaler()
    scaler_task.mean_ = np.array(task['mean'])
    scaler_task.scale_ = np.array(task['std'])
    scaler_task.var_ = np.square(scaler_task.scale_)  # StandardScaler need var_
    scaler_task.n_samples_seen_ = 1  # Avoid warnings.
    scaler.scalers.append(scaler_task)
    scaler.means.append(scaler_task.mean_)
    scaler.stds.append(scaler_task.scale_)

# Standardize the test data.
test_X_lists_scaled = scaler.transform(test_X_list, task_id)

# Create DataLoaders.
test_dataset = VariableLengthDataset(test_X_lists_scaled, test_y)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

model = MultiTaskTransformer(
    feature_dims=feature_dims,
    num_classes_list=num_classes_list,
    d_model=d_model,
    num_layers=num_layers,
    num_heads=num_heads,
    hidden_dim=hidden_dim
)

model.load_state_dict(torch.load(os.path.join(model_path, 'AdapTransformer_model.pth')))
model.to(device)

predictions, probabilities = test_model(model, test_dataloader, task_id)
scipy.io.savemat(
    os.path.join(savepath, 'cls_pre.mat'),
    {
        'clspre': predictions,
        'clsgt': test_y_data,
        'probs': probabilities
    }
)
