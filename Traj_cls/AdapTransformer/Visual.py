# -*- coding: utf-8 -*-
# @Time    : 2025/4/11 11:30
# @Author  : Dongliang

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from AAISPT.Traj_cls.AdapTransformer.dataset import TrajectoryScaler, VariableLengthDataset
from AAISPT.Traj_cls.AdapTransformer.model import MultiTaskTransformer
from AAISPT.Traj_cls.AdapTransformer.train_val import test_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path = r'D:\TrajSeg-Cls\TrajSEG-CLS_V3\CLS\QiPan_NEW\Roll_Feature'
model_path = os.path.join(path, 'AdapTransformer\全参微调\Pre-trained model from All dimensional')
savepath = model_path
if not os.path.exists(savepath):
    os.mkdir(savepath)

if 'QiPan' in path:
    feature_dims = [32]
    num_classes_list = [3]
elif 'YanYu' in path:
    feature_dims = [46]
    num_classes_list = [3]
elif 'Endocytosis' in path:
    feature_dims = [46]
    num_classes_list = [4]

# Load test dataset
test_file = os.path.join(path, 'test_feature.mat')
test_mat = scipy.io.loadmat(test_file)
test_X_data = test_mat['data'].squeeze()
test_y_data = test_mat['label'].flatten()
test_X_list = [torch.tensor(x, dtype=torch.float32) for x in test_X_data]
test_y = torch.tensor(test_y_data, dtype=torch.long)

# Load the standardized parameters (mean and standard deviation).
scaler_params_path = os.path.join(path, 'train_param.json')
with open(scaler_params_path, 'r', encoding='utf-8') as f:
    scaler_params = json.load(f)

# Create a TrajectoryScaler and set the parameters.
scaler = TrajectoryScaler()
scaler.scalers = []

scaler_task = StandardScaler()
scaler_task.mean_ = np.array(scaler_params['train_mean'])
scaler_task.scale_ = np.array(scaler_params['train_std'])
scaler_task.var_ = np.square(scaler_task.scale_)  # StandardScaler requires var_.
scaler_task.n_samples_seen_ = 1  # Avoid warnings.
scaler.scalers.append(scaler_task)
scaler.means.append(scaler_task.mean_)
scaler.stds.append(scaler_task.scale_)

# Standardize the test data.
test_X_lists_scaled = scaler.transform(test_X_list, task_id=0)

# Create a DataLoader.
batch_size = 1024
test_dataset = VariableLengthDataset(test_X_lists_scaled, test_y)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# Hyperparameters (Fixed)
d_model = 64
num_layers = 2
num_heads = 4
hidden_dim = 128

model = MultiTaskTransformer(
    feature_dims=feature_dims,
    num_classes_list=num_classes_list,
    d_model=d_model,
    num_layers=num_layers,
    num_heads=num_heads,
    hidden_dim=hidden_dim
)

model.load_state_dict(torch.load(os.path.join(model_path, 'finetuned_model.pth')))
model.to(device)

predictions, probabilities = test_model(model, test_dataloader, task_id=0)
probabilities = np.array(probabilities)

model.eval()
fixed_features = []
num_samples = test_y.size(0)
with torch.no_grad():
    for X_batch, _, mask, _ in test_dataloader:
        X_batch = X_batch.to(device)
        mask = mask.to(device)
        output = model(X_batch, mask, task_id=0)
        fixed_features.extend(output.detach().cpu().numpy())

fixed_features = np.array(fixed_features)

# Dimensionality reduction to 2D using t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_2d = tsne.fit_transform(fixed_features)  # [num_samples, 2]
scipy.io.savemat(os.path.join(savepath, 'tSNE.mat'), {'tsne': X_2d})

X_2d = scipy.io.loadmat(os.path.join(savepath, 'tSNE.mat'))['tsne']
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 20
fontsize = 20
labelsize = 20
lw = 2
cmap = 'coolwarm'

# # Color encoding based on maximum probability values
# max_probs = np.max(probabilities, axis=1)  # [num_samples]
# plt.figure(figsize=(8, 6))
# scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=max_probs, cmap=cmap, s=50, alpha=0.6)
# cbar = plt.colorbar(scatter, label='Max Probability')
# cbar.set_label('Max Probability', fontsize=fontsize, color='black', fontweight='bold')
# cbar.ax.tick_params(labelsize=labelsize, labelcolor='black')
# plt.title('t-SNE Visualization', fontsize=fontsize, color='black', fontweight='bold')
# plt.xlabel('t-SNE Dimension 1', fontsize=fontsize, color='black', fontweight='bold')
# plt.ylabel('t-SNE Dimension 2', fontsize=fontsize, color='black', fontweight='bold')
# plt.xticks(fontsize=fontsize, color='black', fontweight='bold')
# plt.yticks(fontsize=fontsize, color='black', fontweight='bold')
# plt.gca().set_aspect('equal')
#
# ax = plt.gca()
# for spine in ax.spines.values():
#     spine.set_linewidth(lw)
# plt.tick_params(axis='both', which='major', width=lw, color='black')
# plt.tight_layout()
# plt.savefig(os.path.join(savepath, 'MaxProbability.png'), dpi=600)
# plt.show()
#
# # Color encoding based on the probability of each sample
# num_classes = num_classes_list[0]
# plt.figure(figsize=(12, 10))
# for class_idx in range(num_classes):
#     plt.subplot(2, 2, class_idx + 1)
#     scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=probabilities[:, class_idx], cmap=cmap, s=50, alpha=0.6)
#
#     cbar = plt.colorbar(scatter, label=f'Probability')
#     cbar.set_label(f'Probability', fontsize=fontsize, color='black', fontweight='bold')
#     cbar.ax.tick_params(width=lw, labelsize=labelsize, labelcolor='black')
#
#     plt.title(f'Class {class_idx} Probability', fontsize=fontsize, color='black', fontweight='bold')
#     plt.xlabel('t-SNE Dimension 1', fontsize=fontsize, color='black', fontweight='bold')
#     plt.ylabel('t-SNE Dimension 2', fontsize=fontsize, color='black', fontweight='bold')
#
#     ax = plt.gca()
#     for spine in ax.spines.values():
#         spine.set_linewidth(lw)
#
#     plt.gca().set_aspect('equal')
#     plt.xticks(fontsize=fontsize, color='black', fontweight='bold')
#     plt.yticks(fontsize=fontsize, color='black', fontweight='bold')
#     plt.tick_params(axis='both', which='major', width=lw, color='black')
#
# plt.tight_layout()
# plt.savefig(os.path.join(savepath, 'EachClassProbability.png'), dpi=600)
# plt.show()

# Color encoding based on prediction labels
predictions = scipy.io.loadmat(os.path.join(savepath, 'cls_pre.mat'))['clspre'].squeeze()
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=predictions, cmap=cmap, s=50, alpha=0.6)

cbar = plt.colorbar(scatter, label='Prediction Labels')
cbar.set_label('Prediction Labels', fontsize=fontsize, color='black', fontweight='bold')
cbar.ax.tick_params(labelsize=labelsize, labelcolor='black')
for label in cbar.ax.get_yticklabels():
    label.set_fontweight('bold')
    label.set_fontsize(labelsize)

plt.title('t-SNE Visualization', fontsize=fontsize, color='black', fontweight='bold')
plt.xlabel('t-SNE Dimension 1', fontsize=fontsize, color='black', fontweight='bold')
plt.ylabel('t-SNE Dimension 2', fontsize=fontsize, color='black', fontweight='bold')
plt.xticks(fontsize=fontsize, color='black', fontweight='bold')
plt.yticks(fontsize=fontsize, color='black', fontweight='bold')
plt.gca().set_aspect('equal')

ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(lw)
plt.tick_params(axis='both', which='major', width=lw, color='black')
plt.tight_layout()
plt.savefig(os.path.join(savepath, 'Prediction_Labels.png'), dpi=600)
plt.show()
