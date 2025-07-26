# -*- coding: utf-8 -*-
# @Time    : 2024/7/6 14:12
# @Author  : Dongliang

import os

import numpy as np
import scipy
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from AAISPT.Benchmark.Seg_Adap_BiLSTM_BCE.dataset import label_convert, TSDataset
from AAISPT.Benchmark.Seg_Adap_BiLSTM_BCE.model import Seg_Adap_BiLSTM_BCE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set parameters
FWHM = 10
num_epochs = 100
batch_size = 512
max_dim = 3
d_model = 128
hidden_size = 128
num_layers = 3
out_dim = 1
trace_len = 200

# Define paths
root = r'..\..\data\SEG\Andi'
model_path = r'../data/SEG/Andi/model'

# Dynamically discover test dimension directories
test_dims = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)) and d.endswith('D')]
test_dims.sort()  # Ensure consistent order (e.g., 2D, 3D, 4D, ...)
test_paths = [os.path.join(root, dim) for dim in test_dims]
print(f"Found test dimensions: {test_dims}")

# Initialize dictionaries for test data
test_datasets = {}

# Load test data for each dimension
for dim, path in zip(test_dims, test_paths):
    test_path = os.path.join(path, 'norm_test.mat')
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing norm_test.mat for {dim}")

    # Load .mat file
    test = scipy.io.loadmat(test_path)
    if 'data' not in test or 'label' not in test:
        raise KeyError(f"Missing 'data' or 'label' in {test_path}")

    test_X, test_Y = test['data'].squeeze(), test['label'].squeeze()

    # Convert labels to Gaussian distribution
    test_Y = label_convert(test_Y, trace_len, FWHM=FWHM)
    test_Y = test_Y[:, :, np.newaxis]

    # Convert to tensors
    test_X = torch.tensor(test_X, dtype=torch.float32)
    test_Y = torch.tensor(test_Y, dtype=torch.long)

    # Store in dictionary for TSDataset
    dim_key = dim.lower()  # e.g., '2d', '3d', '5d'
    test_datasets[dim_key] = TSDataset({dim_key: test_X}, {dim_key: test_Y})
    print(f'{dim} test data loaded')

# Create and load model
model = Seg_Adap_BiLSTM_BCE(
    max_dim=max_dim,
    d_model=d_model,
    hidden_size=hidden_size,
    num_layers=num_layers,
    out_dim=out_dim,
)

model_path_file = os.path.join(model_path, 'seg_model.pth')
if not os.path.exists(model_path_file):
    raise FileNotFoundError(f"Model file not found at {model_path_file}")

model.load_state_dict(torch.load(model_path_file, map_location=device))
model.to(device)
model.eval()

# Perform inference for each dimension
all_probs = {}
all_locs = {}
all_gt = {}

for dim_key, dataset in test_datasets.items():
    print(f"Running inference for {dim_key.upper()}")
    probs, locs = [], []
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=False)
    progress_bar = tqdm(enumerate(test_loader), total=len(test_loader), desc="Processing batches")

    with torch.no_grad():
        for i, data in progress_bar:
            data = data[0].to(device)
            pred = model(data)
            pred = pred.squeeze().cpu().numpy()

            # # Define region of interest (ROI) for peak detection
            # x = np.linspace(1, trace_len, trace_len)
            # x_min, x_max = 10, trace_len - 10
            # mask = (x >= x_min) & (x <= x_max)
            # y_roi = pred[:, mask]
            #
            # # Peak detection
            # loc = []
            # for i in range(len(y_roi)):
            #     peaks, _ = find_peaks(y_roi[i])
            #     if len(peaks) > 0:
            #         max_peak_idx = peaks[np.argmax(y_roi[i][peaks])]
            #         loc.append(max_peak_idx + 10)  # Adjust for ROI offset
            #     else:
            #         loc.append(-1)
            #
            # probs.extend(pred)
            # locs.extend(loc)

            loc = np.argmax(pred, axis=1)
            probs.extend(pred)
            locs.extend(loc)

    # Get ground truth peak locations
    test_Y = dataset.labels[dim_key]  # Access labels from TSDataset
    gt_locs = np.argmax(test_Y, axis=1)

    all_probs[dim_key] = probs
    all_locs[dim_key] = locs
    all_gt[dim_key] = gt_locs

# Save results for each dimension
for dim_key in test_datasets.keys():
    savepath = os.path.join(root, dim_key)
    if not os.path.exists(savepath):
        os.mkdir(savepath)

    scipy.io.savemat(
        os.path.join(savepath, f'seg_pre.mat'),
        {
            'probs': all_probs[dim_key],
            'pre': all_locs[dim_key],
            'gt': all_gt[dim_key]
        }
    )
    print(f'Saved results for {dim_key.upper()} to seg_pre.mat')

# scipy.io.savemat(os.path.join(savepath, 'seg_pre.mat'),
#                  {'probs': probs,
#                   'pre': locs,
#                   'gt': np.argmax(test_Y, axis=1)
#                   }
#                  )
