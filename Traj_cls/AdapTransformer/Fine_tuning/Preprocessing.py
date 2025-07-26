# -*- coding: utf-8 -*-
# @Time    : 2025/4/3 20:43
# @Author  : Dongliang

import json
import os

import scipy
import torch

from AAISPT.Traj_cls.AdapTransformer.dataset import VariableLengthDataset, TrajectoryScaler

path = r'D:\TrajSeg-Cls\TrajSEG-CLS_V3\CLS\QiPan_NEW\Roll_Feature'

if 'QiPan' in path:
    feature_dims = [32]
elif 'YanYu' in path:
    feature_dims = [46]
elif 'Endocytosis' in path:
    feature_dims = [46]

savepath = os.path.join(path, 'AdapTransformer')
if not os.path.exists(savepath):
    os.makedirs(savepath)

mat_files = {
    'train': os.path.join(path, 'train_feature.mat'),
    'val': os.path.join(path, 'val_feature.mat'),
    'test': os.path.join(path, 'test_feature.mat')
}

# Load data
train_mat = scipy.io.loadmat(mat_files['train'])
train_X_data = train_mat['data'].squeeze()
train_y_data = train_mat['label'].flatten()
train_X_list = [torch.tensor(x, dtype=torch.float32) for x in train_X_data]
train_y = torch.tensor(train_y_data, dtype=torch.long)

val_mat = scipy.io.loadmat(mat_files['val'])
val_X_data = val_mat['data'].squeeze()
val_y_data = val_mat['label'].flatten()
val_X_list = [torch.tensor(x, dtype=torch.float32) for x in val_X_data]
val_y = torch.tensor(val_y_data, dtype=torch.long)

test_mat = scipy.io.loadmat(mat_files['test'])
test_X_data = test_mat['data'].squeeze()
test_y_data = test_mat['label'].flatten()
test_X_list = [torch.tensor(x, dtype=torch.float32) for x in test_X_data]
test_y = torch.tensor(test_y_data, dtype=torch.long)

# Validating feature dimensions
assert train_X_list[0].shape[1] == feature_dims[0], \
    f"Train feature dimension mismatch: expected {feature_dims[0]}, got {train_X_list[0].shape[1]}"
assert val_X_list[0].shape[1] == feature_dims[0], \
    f"Val feature dimension mismatch: expected {feature_dims[0]}, got {val_X_list[0].shape[1]}"
assert test_X_list[0].shape[1] == feature_dims[0], \
    f"Val feature dimension mismatch: expected {feature_dims[0]}, got {test_X_list[0].shape[1]}"

# Normalize datasets
scaler = TrajectoryScaler()
scaler.fit([train_X_list])
train_X_list_scaled = scaler.transform([train_X_list],task_id=0)[0]
val_X_list_scaled = scaler.transform([val_X_list],task_id=0)[0]
test_X_list_scaled = scaler.transform([test_X_list],task_id=0)[0]

# Save standardized parameters
means, stds = scaler.get_params()
scaler_params = {
    'task_0': {
        'mean': means[0].tolist(),
        'std': stds[0].tolist()
    }
}
with open(os.path.join(savepath, 'finetune_scaler_params.json'), 'w', encoding='utf-8') as f:
    json.dump(scaler_params, f, indent=4)

# Creating a Dataset
train_dataset = VariableLengthDataset(train_X_list_scaled, train_y)
val_dataset = VariableLengthDataset(val_X_list_scaled, val_y)
test_dataset = VariableLengthDataset(test_X_list_scaled, test_y)

# Save Dataset as .pt file
torch.save(train_dataset, os.path.join(savepath, 'train_dataset.pt'))
torch.save(val_dataset, os.path.join(savepath, 'val_dataset.pt'))
torch.save(test_dataset, os.path.join(savepath, 'test_dataset.pt'))
print(f"Preprocessed data saved to {savepath}")
