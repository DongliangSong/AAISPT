# -*- coding: utf-8 -*-
# @Time    : 2025/7/2 9:34
# @Author  : Dongliang

import os

import matplotlib.pyplot as plt
import numpy as np
import scipy
import shap
import torch
import torch.nn.utils.rnn as rnn_utils
from tqdm import tqdm

import AAISPT.Traj_cls.utils.config as c

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
fontsize = 18
linewidth = 2


class WrappedModel(torch.nn.Module):
    def __init__(self, model, task_id):
        super().__init__()
        self.model = model
        self.task_id = task_id

    def forward(self, x):
        lengths = (x.abs().sum(-1) > 0).sum(dim=1)
        mask = torch.arange(x.size(1), device=x.device)[None, :] < lengths[:, None]

        return self.model(x, mask, self.task_id)


def compute_shap_for_trajectories(model, trajectories, background_samples=20):
    """
    Computes SHAP values for each trajectory in the dataset.

    :param model: The trained model for which SHAP values are computed.
    :param trajectories: List of trajectory tensors to analyze.
    :param background_samples: Number of background samples for SHAP explainer (default: 20).
    :return: List of SHAP values for each trajectory.
    """

    # Pad sequences
    padded_data = rnn_utils.pad_sequence(trajectories, batch_first=True)
    padded_data = padded_data.to(device)

    # Select random background samples
    idx = np.random.choice(len(trajectories), background_samples, replace=False)
    background = padded_data[idx]
    explainer = shap.DeepExplainer(model, background)

    # Compute SHAP values for each trajectory
    all_shap_values = []
    for traj in tqdm(padded_data, desc="Calculating SHAP"):
        traj_tensor = traj.unsqueeze(0)  # (1, seq_len, 3)
        shap_values = explainer.shap_values(traj_tensor)
        all_shap_values.append(shap_values)

    return all_shap_values


def plot_single_traj_shap(shap_values, traj_idx):
    """
    Plots feature importance for a single trajectory using SHAP values.

    :param shap_values: SHAP values for the trajectory.
    :param traj_idx: Index of the trajectory for labeling.
    """
    plt.figure(figsize=(10, 4))
    shap_values_class0 = np.array(shap_values[0][0])

    # Plot feature importance as a bar chart
    shap.summary_plot(
        shap_values_class0,
        feature_names=feature_names,
        plot_type="bar",
        show=False
    )
    plt.title(f"Feature Importance (Trajectory {traj_idx + 1})")
    plt.show()


def plot_global_shap(all_shap_values):
    """
    Plots global feature importance by averaging absolute SHAP values across all trajectories.

    :param all_shap_values: List of SHAP values for all trajectories.
    """

    # Concatenate SHAP values for class 0 across all trajectories
    global_shap = np.concatenate([shap[0][0] for shap in all_shap_values])

    # Compute mean absolute SHAP values
    mean_abs_shap = np.mean(np.abs(global_shap), axis=0)

    # Plot global feature importance
    plt.bar(feature_names, mean_abs_shap)
    plt.title("Global Feature Importance (Mean |SHAP|)")
    plt.ylabel("Mean Absolute SHAP Value")
    plt.show()


if __name__ == '__main__':
    # path = r'D:\TrajSeg-Cls\TrajSEG-CLS_V3\CLS\QiPan_NEW\Roll_Feature'
    # path = r'D:\TrajSeg-Cls\TrajSEG-CLS_V3\CLS\YanYu_NEW\Roll_Feature'
    path = r'D:\TrajSeg-Cls\TrajSEG-CLS_V3\CLS\Endocytosis_NEW\5D\Roll_Feature'
    model_path = os.path.join(path, 'AdapTransformer\全参微调\Pre-trained model from All dimensional')
    savepath = os.path.join(model_path, r'SHAP_importance')
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

    _, feature_names = c.config_feature(path)

    # # Load test dataset
    # test = scipy.io.loadmat(os.path.join(path, 'test_feature_all_stand.mat'))
    # test_X, test_Y = test['data'].squeeze(), test['label'].squeeze()
    # test_X = test_X.tolist()
    # test_X = [torch.tensor(i, dtype=torch.float32) for i in test_X]
    # seq_len = [len(i) for i in test_X]
    #
    # # Hyperparameters (Fixed)
    # d_model = 64
    # num_layers = 2
    # num_heads = 4
    # hidden_dim = 128
    #
    # model = MultiTaskTransformer(
    #     feature_dims=feature_dims,
    #     num_classes_list=num_classes_list,
    #     d_model=d_model,
    #     num_layers=num_layers,
    #     num_heads=num_heads,
    #     hidden_dim=hidden_dim
    # )
    #
    # model.load_state_dict(torch.load(os.path.join(model_path, 'finetuned_model.pth')))
    # model.to(device)
    #
    # # Calculate SHAP values for all trajectories
    # wrapped_model = WrappedModel(model, task_id=0)
    #
    # all_shap = compute_shap_for_trajectories(wrapped_model, test_X)
    # scipy.io.savemat(os.path.join(savepath,'new_shap.mat'),{'shap':all_shap})

    all_shap = scipy.io.loadmat(os.path.join(savepath, 'new_shap.mat'))['shap']
    all_shap = all_shap.squeeze()

    # Merge SHAP values across all classes (take absolute values to represent importance magnitude)
    merged_shap = []
    for shap_tuple in all_shap:
        # Sum SHAP values across classes for each trajectory
        combined = np.sum([np.abs(shap) for shap in shap_tuple], axis=0)  # shape (seq_len, n_features)
        merged_shap.append(combined)

    # Find the maximum sequence length
    max_len = max([shap.shape[0] for shap in merged_shap])

    # Initialize a matrix for time-dimension statistics
    time_agg_shap = np.zeros((max_len, len(feature_names)))
    time_counts = np.zeros(max_len)

    # Aggregate SHAP values across all trajectories
    for shap in merged_shap:
        seq_len = shap.shape[0]
        for t in range(seq_len):
            time_agg_shap[t] += shap[t]
            time_counts[t] += 1

    # Compute average importance
    time_avg_shap = time_agg_shap / time_counts[:, np.newaxis]

    # Visualize
    plt.figure(figsize=(12, 6))
    for feat_idx in range(len(feature_names)):
        plt.plot(time_avg_shap[:, feat_idx], label=feature_names[feat_idx])

    plt.xlabel('Time Step')
    plt.ylabel('Mean Combined SHAP Value')
    plt.title('Temporal Importance (All Classes Combined)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Method 1: Simple average
    global_feat_importance = np.mean([np.mean(shap, axis=0) for shap in merged_shap], axis=0)

    # Method 2: Weighted average (considering trajectory length)
    total_shap = np.zeros(len(feature_names))
    total_steps = 0
    for shap in merged_shap:
        total_shap += np.sum(shap, axis=0)
        total_steps += shap.shape[0]
    global_feat_importance_weighted = total_shap / total_steps
    # Sort features by global_feat_importance_weighted in descending order and select top 10
    sorted_indices = np.argsort(global_feat_importance_weighted)[::-1][:10]  # Top 10 indices
    sorted_feature_names = [feature_names[i] for i in sorted_indices]
    sorted_importance = global_feat_importance_weighted[sorted_indices]

    # Plot global feature importance (top 10, sorted)
    cmap = plt.get_cmap('coolwarm')
    colors = cmap(np.linspace(0, 1, len(sorted_indices)))
    plt.figure(figsize=(10, 6))
    bars = plt.bar(sorted_feature_names, sorted_importance, capsize=5, color=colors, edgecolor='black', ecolor='black')
    plt.title('Top-10 Feature Importance', fontsize=fontsize, fontweight='bold', fontfamily='Arial')
    plt.xticks(rotation=45, ha='right', fontname='Arial', fontweight='bold', fontsize=fontsize)
    plt.yticks(fontsize=fontsize, fontweight='bold', fontfamily='Arial')
    plt.ylabel('Mean SHAP Value', fontname='Arial', fontweight='bold', fontsize=fontsize)
    plt.xlabel('Features', fontsize=fontsize, fontweight='bold', fontfamily='Arial')

    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(linewidth)

    plt.tight_layout()
    plt.savefig(os.path.join(savepath, 'Global_importance.png'), dpi=600, bbox_inches='tight')
    plt.show()

    # Compute the average value for each (time, feature) position
    time_feature_matrix = np.zeros((max_len, len(feature_names)))
    count_matrix = np.zeros((max_len, len(feature_names)))
    for shap in merged_shap:
        seq_len = shap.shape[0]
        for t in range(seq_len):
            time_feature_matrix[t] += shap[t]
            count_matrix[t] += 1

    avg_time_feature = time_feature_matrix / count_matrix

    # Plot a heatmap
    plt.figure(figsize=(10, 8))
    im = plt.imshow(avg_time_feature.T, aspect='auto', cmap='coolwarm')
    cbar = plt.colorbar(im)
    cbar.set_label('Mean Combined SHAP Value', fontname='Arial', fontweight='bold', fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontname='Arial', fontweight='bold')
    plt.xticks(fontname='Arial', fontweight='bold', fontsize=14)
    plt.yticks(np.arange(len(feature_names)), feature_names, fontname='Arial', fontweight='bold', fontsize=14)
    plt.xlabel('Window', fontname='Arial', fontweight='bold', fontsize=16)
    plt.ylabel('Feature', fontname='Arial', fontweight='bold', fontsize=16)
    plt.title('Time-Feature Importance (All Classes)', fontname='Arial', fontweight='bold', fontsize=18)
    plt.savefig(os.path.join(savepath, 'Time-feature.png'), dpi=600, bbox_inches='tight')  # Save with 600 DPI
    plt.show()

    # Collect all SHAP values for all features (merged across classes)
    feat_shap_dict = {feat: [] for feat in feature_names}
    for shap in merged_shap:
        for t in range(shap.shape[0]):
            for feat_idx in range(len(feature_names)):
                feat_shap_dict[feature_names[feat_idx]].append(shap[t, feat_idx])

    # Plot a distribution graph
    plt.figure(figsize=(12, 6))
    for feat in feature_names:
        plt.hist(feat_shap_dict[feat], bins=50, alpha=0.5, label=feat)
    plt.xlabel('Combined SHAP Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of Feature Importance (All Classes)')
    plt.legend()
    plt.show()

    # Compute statistical measures
    for feat in feature_names:
        vals = np.array(feat_shap_dict[feat])
        print(f"{feat}:")
        print(f"  Mean = {np.mean(vals):.4f}")
        print(f"  Std = {np.std(vals):.4f}")
        print(f"  % Significant (SHAP>0.01) = {np.mean(vals > 0.01) * 100:.1f}%")

    scipy.io.savemat(
        os.path.join(savepath, 'shap_statis.mat'),
        {
            # 'feat_shap_dict': feat_shap_dict,
            'avg_time_feature': avg_time_feature,
            'time_feature_matrix': time_feature_matrix,
            'global_feat_importance_weighted': global_feat_importance_weighted,
            'time_avg_shap': time_avg_shap,
            'time_agg_shap': time_agg_shap,
            'merged_shap': merged_shap
        }
    )
