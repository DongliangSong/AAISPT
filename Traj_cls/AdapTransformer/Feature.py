# -*- coding: utf-8 -*-
# @Time    : 2025/5/11 21:46
# @Author  : Dongliang

import os

import numpy as np
import scipy
import shap
import torch
from matplotlib import pyplot as plt

from AAISPT.Traj_cls.AdapTransformer.model import MultiTaskTransformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path = r'D:\TrajSeg-Cls\TrajSEG-CLS_V3\CLS\YanYu_NEW\Roll_Feature'
model_path = os.path.join(path, 'AdapTransformer\全参微调\Pre-trained model from All dimensional')
savepath = model_path
if not os.path.exists(savepath):
    os.makedirs(savepath)

# Model hyperparameters
d_model = 64
num_layers = 2
num_heads = 4
hidden_dim = 128
batch_size = 2048

if 'QiPan' in path:
    feature_dims = [32]
    num_classes_list = [3]
elif 'YanYu' in path:
    feature_dims = [46]
    num_classes_list = [3]
elif 'Endocytosis' in path:
    feature_dims = [46]
    num_classes_list = [4]

# Initialize the model
model = MultiTaskTransformer(
    feature_dims=feature_dims,
    num_classes_list=num_classes_list,
    d_model=d_model,
    num_layers=num_layers,
    num_heads=num_heads,
    hidden_dim=hidden_dim
)

# Load well-trained model
model.load_state_dict(torch.load(os.path.join(model_path, 'finetuned_model.pth')))
model.to(device)
model.eval()

# Load dataset
data = scipy.io.loadmat(os.path.join(path, 'test_feature_all_stand.mat'))
bg_data = data['data'].squeeze()
test = [torch.tensor(bg_data[i], dtype=torch.float32) for i in range(100)]  # 100 trajectories

# Calculate SHAP values
explainer = shap.DeepExplainer(model, test)
shap_values = explainer.shap_values(bg_data)

num_features = feature_dims[0]
feature_importance = np.abs(shap_values).mean(axis=0)
feature_names = [f"Feature_{i}" for i in range(num_features)]
sorted_idx = np.argsort(feature_importance)[::-1]
feature_importance = feature_importance[sorted_idx]
feature_names = [feature_names[i] for i in sorted_idx]

plt.figure(figsize=(10, 6))
plt.bar(np.arange(num_features), feature_importance, color='skyblue')
plt.xlabel('Feature Index')
plt.ylabel('Mean Absolute SHAP Value')
plt.title('Feature Importance (SHAP)')
plt.xticks(np.arange(num_features), feature_names, rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(savepath, 'feature_importance.png'), dpi=600)
plt.show()
