# -*- coding: utf-8 -*-
# @Time    : 2025/4/12 17:51
# @Author  : Dongliang


import json
import os
import pickle
from datetime import datetime

import scipy
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from AAISPT.Benchmark.Cls_Adap_Transformer.dataset import VariableLengthDataset, collate_fn, TrajectoryScaler
from AAISPT.Benchmark.Cls_Adap_Transformer.model import MultiTaskTransformer
from AAISPT.Benchmark.Cls_Adap_Transformer.train_val import train_model
from AAISPT.pytorch_tools import EarlyStopping

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set parameters
num_tasks = 2
feature_dims = [32, 32]  # Feature dimensions for each task
num_classes_list = [5, 5]  # Number of classes for each task
batch_size = 2048
init_lr = 0.01
weight_decay = 0.001
epochs = 100
patience = 10

# Set network parameters
d_model = 64
num_layers = 2
num_heads = 4
hidden_dim = 128

# Define paths
root = r'..\data\CLS\Andi'
savepath = os.path.join(root, 'AdapTransformer')
if not os.path.exists(savepath):
    os.makedirs(savepath, exist_ok=True)

# Dynamically discover task directories
task_dirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)) and d.endswith('D')]
task_dirs.sort()  # Ensure consistent order (e.g., 2D, 3D, ...)
if len(task_dirs) < num_tasks:
    raise ValueError(f"Found {len(task_dirs)} task directories, but num_tasks is {num_tasks}")
print(f"Found task directories: {task_dirs}")

# Construct paths for train and val .mat files
mat_files = {
    'train': [os.path.join(root, dim, 'Feature_based', 'train_feature.mat') for dim in task_dirs],
    'val': [os.path.join(root, dim, 'Feature_based', 'val_feature.mat') for dim in task_dirs]
}

train_datasets, train_X_lists = [], []
val_datasets, val_X_lists = [], []
for task_id in range(num_tasks):
    # Load training data
    train_mat = scipy.io.loadmat(mat_files['train'][task_id])
    train_X_data = train_mat['data'].squeeze()
    train_y_data = train_mat['label'].flatten()
    train_X_list = [torch.tensor(x, dtype=torch.float32) for x in train_X_data]
    train_y = torch.tensor(train_y_data, dtype=torch.long)

    # Load validation data
    val_mat = scipy.io.loadmat(mat_files['val'][task_id])
    val_X_data = val_mat['data'].squeeze()
    val_y_data = val_mat['label'].flatten()
    val_X_list = [torch.tensor(x, dtype=torch.float32) for x in val_X_data]
    val_y = torch.tensor(val_y_data, dtype=torch.long)

    # Validate feature dimensions
    assert train_X_list[0].shape[1] == feature_dims[task_id], \
        f"Task {task_id} train feature dimension mismatch: expected {feature_dims[task_id]}, got {train_X_list[0].shape[1]}"
    assert val_X_list[0].shape[1] == feature_dims[task_id], \
        f"Task {task_id} val feature dimension mismatch: expected {feature_dims[task_id]}, got {val_X_list[0].shape[1]}"

    # Store data
    train_X_lists.append(train_X_list)
    val_X_lists.append(val_X_list)
    train_datasets.append((train_X_list, train_y))
    val_datasets.append((val_X_list, val_y))

# Standardize data
scaler = TrajectoryScaler()
scaler.fit(train_X_lists)
train_X_lists_scaled = scaler.transform(train_X_lists)
val_X_lists_scaled = scaler.transform(val_X_lists)

means, stds = scaler.get_params()
for task_id in range(num_tasks):
    print(f"Task {task_id} Mean: {means[task_id]}")
    print(f"Task {task_id} Std: {stds[task_id]}")

# Save scaler parameters
scaler_params = {'tasks': []}
for task_id in range(num_tasks):
    task_data = {
        'task_id': task_id,
        'description': f"Statistics for Task {task_id}",
        'mean': means[task_id].tolist(),
        'std': stds[task_id].tolist()
    }
    scaler_params['tasks'].append(task_data)

with open(os.path.join(savepath, 'train_param.json'), 'w', encoding='utf-8') as f:
    json.dump(scaler_params, f, indent=4)

# Create DataLoaders
train_dataloaders = []
val_dataloaders = []
for task_id in range(num_tasks):
    train_dataset = VariableLengthDataset(train_X_lists_scaled[task_id], train_datasets[task_id][1])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    train_dataloaders.append(train_dataloader)
    val_dataset = VariableLengthDataset(val_X_lists_scaled[task_id], val_datasets[task_id][1])
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    val_dataloaders.append(val_dataloader)

# Initialize model and training components
model = MultiTaskTransformer(
    feature_dims=feature_dims,
    num_classes_list=num_classes_list,
    d_model=d_model,
    num_layers=num_layers,
    num_heads=num_heads,
    hidden_dim=hidden_dim
)

optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, weight_decay=weight_decay)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, eps=1e-06)

# Initialize the early_stopping object
early_stopping = EarlyStopping(patience, verbose=True)

# Train model
History = train_model(
    model,
    train_dataloaders,
    val_dataloaders,
    num_epochs=epochs,
    optimizer=optimizer,
    scheduler=scheduler,
    criterion=criterion,
    device=device,
    savepath=savepath,
    early_stopping=early_stopping
)

# Save model and history
torch.save(model.state_dict(), os.path.join(savepath, 'AdapTransformer_model.pth'))
history = {'total_train_losses': History['total_train_losses'],
           'total_val_losses': History['total_val_losses'],
           'total_train_accuracies': History['total_train_accuracies'],
           'total_val_accuracies': History['total_val_accuracies'],
           'train_task_losses': History['train_task_losses'],
           'val_task_losses': History['val_task_losses']
           }

with open(os.path.join(savepath, 'history.pkl'), 'wb') as f:
    pickle.dump(history, f)

print('======================== Visualization =======================')
# Plot training & validation loss
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(History['total_train_losses'])
plt.plot(History['total_val_losses'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['Train', 'Validation'], loc='upper right')

plt.subplot(1, 2, 2)
plt.plot(History['total_train_accuracies'])
plt.plot(History['total_val_accuracies'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.savefig(os.path.join(savepath, 'Loss_Acc.png'), dpi=600)
plt.show()

params = {
    'd_model': d_model,
    'num_layers': num_layers,
    'num_heads': num_heads,
    'hidden_dim': hidden_dim,
    'num_tasks': num_tasks,
    'batch_size': batch_size,
    'learning_rate': init_lr,
    'weight_decay': weight_decay,
    'epochs': epochs,
    'savepath': savepath
}

# Get current date and time
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

file_name = os.path.join(savepath, f'params_{current_time}.txt')
with open(file_name, 'w') as f:
    f.write('Train parameters:\n')
    for key, value in params.items():
        f.write(f"{key} = {value}\n")
