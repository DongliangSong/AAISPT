# -*- coding: utf-8 -*-
# @Time    : 2024/7/6 14:12
# @Author  : Dongliang

import os
from pathlib import Path

import numpy as np
import scipy
import torch

from AAISPT.Traj_seg.Seg_Adap_BiLSTM_BCE.main import Seg_Adap_BiLSTM_BCE, inference

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Example usage
data_path = r'..\data\2D'
model_path = r'..\model'
save_path = os.path.join(model_path, '2Dtest')
if not os.path.exists(save_path):
    os.mkdir(save_path)

# Set parameters
FWHM = 20
batch_size = 512

# Fixed network parameters
max_dim = 5
d_model = 128
hidden_size = 128
num_layers = 3
out_dim = 1

# Load test dataset
test = scipy.io.loadmat(os.path.join(data_path, 'norm_test.mat'))
test_X, test_Y = test['data'].squeeze(), test['label'].squeeze()

# Create and load model
model = Seg_Adap_BiLSTM_BCE(
    max_dim=max_dim,
    d_model=d_model,
    hidden_size=hidden_size,
    num_layers=num_layers,
    out_dim=out_dim,
)

model.load_state_dict(torch.load(os.path.join(model_path, 'seg_model.pth')))

probs, locs = inference(model, test_X, test_Y, FWHM=FWHM, batch_size=batch_size, device=device)

# Save results
scipy.io.savemat(
    os.path.join(save_path, 'seg_pre.mat'),
    {
        'probs': np.array(probs),
        'pre': np.array(locs),
        'gt': test_Y
    }
)
print(f"Results saved to {os.path.join(save_path, 'seg_pre.mat')}")
