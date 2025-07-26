# -*- coding: utf-8 -*-
# @Time    : 2025/4/10 21:46
# @Author  : Dongliang

import json
import os

import numpy as np
import scipy
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from AAISPT.Traj_cls.AdapTransformer.dataset import VariableLengthDataset, collate_fn, TrajectoryScaler
from AAISPT.Traj_cls.AdapTransformer.model import MultiTaskTransformer, compute_feature_importance, \
    compute_feature_importance_by_gradient, compute_input_attention_interaction

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def attention_and_importance(model, test_dataloader, task_id, device):
    """
    Evaluates the model on test data, extracting attention weights and feature importance metrics.

    :param model: Trained MultiTaskTransformer model.
    :param test_dataloader: DataLoader containing test data.
    :param task_id: Integer index of the task to evaluate.
    :param device: Device to perform computations on (e.g., 'cuda' or 'cpu').
    :return:
            tuple: Lists of:
            - Attention weights for each batch.
            - Mean attention weights across heads.
            - Time step importance scores.
            - Gradient-based feature importance.
            - Attention-input interaction scores.
    """

    model.eval()
    model.to(device)

    # Lists to store computed metrics
    all_attn_weights = []
    all_attn_mean = []
    all_time_importance = []
    all_feature_importance = []
    all_attn_input = []

    for batch_idx, (X_batch, _, mask, lengths) in enumerate(test_dataloader):
        X_batch = X_batch.to(device)
        mask = mask.to(device)

        # Perform forward pass and retrieve attention weights
        with torch.no_grad():
            logits, attn_weights = model(X_batch, mask=mask, task_id=task_id, return_attention=True)

        print(f"Batch {batch_idx}:")
        print(f"Logits shape: {logits.shape}")
        print(f"Attention Weights shape: {attn_weights.shape}")

        # Calculate feature and time step importance
        attn_weights, attn_mean, time_importance = compute_feature_importance(model, X_batch, mask, task_id)
        all_attn_weights.append(attn_weights.cpu().numpy())
        all_attn_mean.append(attn_mean.cpu().numpy())
        all_time_importance.append(time_importance.cpu().numpy())

        # Calculate feature importance using gradients
        feature_importance = compute_feature_importance_by_gradient(model, X_batch, mask, task_id)
        all_feature_importance.append(feature_importance.cpu().numpy())

        # Calculate attention-input interaction
        attn_input = compute_input_attention_interaction(model, X_batch, mask, task_id)
        all_attn_input.append(attn_input.cpu().numpy())

        print(f"Time Step Importance shape: {time_importance.shape}")

    return all_attn_weights, all_attn_mean, all_time_importance, all_feature_importance, all_attn_input


if __name__ == "__main__":
    path = r'D:\TrajSeg-Cls\TrajSEG-CLS_V3\CLS\QiPan_NEW\Roll_Feature'
    model_path = os.path.join(path, r'AdapTransformer\全参微调\Pre-trained model from All dimensional')
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

    # Determine the task ID based on the feature dimension.
    test_feature_dim = test_X_list[0].shape[1]
    task_id = None
    for i, dim in enumerate(feature_dims):
        if test_feature_dim == dim:
            task_id = i
            break
    if task_id is None:
        raise ValueError(f'Test feature dimension {test_feature_dim} does not match any task: {feature_dims}')

    print(f'Test data belongs to Task {task_id} (feature_dim={test_feature_dim})')

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
    scaler_task.var_ = np.square(scaler_task.scale_)  # StandardScaler requires var_
    scaler_task.n_samples_seen_ = 1
    scaler.scalers.append(scaler_task)
    scaler.means.append(scaler_task.mean_)
    scaler.stds.append(scaler_task.scale_)

    # Standardize the test data.
    test_X_lists_scaled = scaler.transform(test_X_list, task_id)

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

    all_attn_weights, all_attn_mean, all_time_importance, all_feature_importance, all_attn_input = attention_and_importance(
        model, test_dataloader, task_id=0, device=device)

    scipy.io.savemat(os.path.join(savepath, 'Feature_timestep_importance.mat'),
                     {
                         # 'all_attn_weights': all_attn_weights,
                         # 'all_attn_mean': all_attn_mean,
                         # 'all_time_importance': all_time_importance,
                         'all_feature_importance': all_feature_importance,
                         # 'all_attn_input': all_attn_input
                     }
                     )
