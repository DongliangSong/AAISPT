# -*- coding: utf-8 -*-
# @Time    : 2024/8/10 16:36
# @Author  : Dongliang

import os
import pickle
from datetime import datetime

import numpy as np
import scipy
import torch
from matplotlib import pyplot as plt
from torch import nn, optim

from AAISPT.Benchmark.Seg_Adap_BiLSTM_BCE.dataset import label_convert, TSDataset
from AAISPT.Benchmark.Seg_Adap_BiLSTM_BCE.model import Seg_Adap_BiLSTM_BCE
from AAISPT.Benchmark.Seg_Adap_BiLSTM_BCE.train_val import training
from AAISPT.pytorch_tools import EarlyStopping

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set parameters
FWHM = 10
num_epochs = 100
batch_size = 512
init_lr = 1e-3
weight_decay = 1e-4
patience = 10

max_dim = 3
d_model = 128
hidden_size = 128
num_layers = 3
out_dim = 1
trace_len = 199
num_channels_list = [2, 3]

# Load dataset
root = r'..\..\data\SEG\Andi'
savepath = os.path.join(root, 'Seg_Adap_BiLSTM_BCE')
if not os.path.exists(savepath):
    os.mkdir(savepath)

# Dynamically discover dimension directories
dims = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)) and d.endswith('D')]
dims.sort()  # Ensure consistent order (e.g., 2D, 3D, 4D, ...)
paths = [os.path.join(root, dim) for dim in dims]
print(f"Found dimensions: {dims}")

# Initialize dictionaries for training and validation data
data_dict_train,label_dict_train = {},{}
data_dict_val ,label_dict_val= {},{}
for dim, path in zip(dims, paths):
    train = scipy.io.loadmat(os.path.join(path, 'norm_train.mat'))
    val = scipy.io.loadmat(os.path.join(path, 'norm_val.mat'))

    train_x, train_y = train['data'].squeeze(), train['label'].squeeze()
    val_x, val_y = val['data'].squeeze(), val['label'].squeeze()

    train_y = label_convert(train_y, trace_len, FWHM=FWHM)
    val_y = label_convert(val_y, trace_len, FWHM=FWHM)

    train_y = train_y[:, :, np.newaxis]
    val_y = val_y[:, :, np.newaxis]

    # Store in dictionaries with dimension as key (e.g., '2d', '3d')
    dim_key = dim.lower()  # Use lowercase for consistency (e.g., '2d', '3d')
    data_dict_train[dim_key] = train_x
    label_dict_train[dim_key] = train_y
    data_dict_val[dim_key] = val_x
    label_dict_val[dim_key] = val_y
    print(f'{dim} data loaded')

# Create datasets
train_dataset = TSDataset(data_dict_train,label_dict_train)
val_dataset = TSDataset(data_dict_val, label_dict_val)

# Create model, loss function, and optimizer
model = Seg_Adap_BiLSTM_BCE(
    max_dim=max_dim,
    d_model=d_model,
    hidden_size=hidden_size,
    num_layers=num_layers,
    out_dim=out_dim
)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params}")

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=init_lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, eps=1e-06)

# Initialize the early_stopping object.
early_stopping = EarlyStopping(patience, verbose=True)
train_losses, val_losses, best_loss = training(
    model=model,
    batch_size=batch_size,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    num_epochs=num_epochs,
    scheduler=scheduler,
    optimizer=optimizer,
    criterion=criterion,
    device=device,
    savepath=savepath,
    early_stopping=early_stopping
)

# Save model and training history
torch.save(model.state_dict(), os.path.join(savepath, 'seg_model.pth'))
with open(os.path.join(savepath, 'history.pkl'), 'wb') as f:
    pickle.dump({'train_losses': train_losses, 'val_losses': val_losses}, f)

plt.figure(figsize=(8, 8))
plt.plot(np.arange(0, num_epochs), train_losses, label='train_loss')
plt.plot(np.arange(0, num_epochs), val_losses, label='val_loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='upper left')
plt.savefig(os.path.join(savepath, 'train_and_val.png'), dpi=600)
plt.show()

params = {
    'num_epochs': num_epochs,
    'batch_size': batch_size,
    'init_lr': init_lr,
    'weight_decay': weight_decay,
    'max_dim': max_dim,
    'd_model': d_model,
    'hidden_size': hidden_size,
    'num_layers': num_layers,
    'out_dim': out_dim,
    'savepath': savepath
}
# Get current date and time
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

file_name = os.path.join(savepath, f'params_{current_time}.txt')
with open(file_name, 'w') as f:
    f.write('Seg_Adap_BiLSTM_BCE train parameters:\n')
    for key, value in params.items():
        f.write(f"{key} = {value}\n")
