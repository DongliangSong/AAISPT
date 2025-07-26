# -*- coding: utf-8 -*-
# @Time    : 2025/4/12 20:38
# @Author  : Dongliang

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy
import seaborn as sns
import torch
from matplotlib.font_manager import FontProperties
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from AAISPT.Traj_cls.AdapTransformer.dataset import VariableLengthDataset, collate_fn, TrajectoryScaler
from AAISPT.Traj_cls.AdapTransformer.model import MultiTaskTransformer
from AAISPT.Traj_cls.AdapTransformer.train_val import test_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Global constants
default_cmap = 'coolwarm'
fontsize = 20
spine_linewidth = 2
tick_label_size = 16


def configure_axes(ax, linewidth=2, tick_label_size=16):
    """
    Configure axes with standardized settings.

    :param ax: Matplotlib axes object to configure.
    :param linewidth: Line width for spines.
    :param tick_label_size: Font size for tick labels.
    """
    ax.spines['bottom'].set_linewidth(linewidth)
    ax.spines['left'].set_linewidth(linewidth)
    ax.spines['right'].set_linewidth(linewidth)
    ax.spines['top'].set_linewidth(linewidth)
    ax.tick_params(axis='both', which='major', labelsize=tick_label_size, width=2)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily('Arial')
        label.set_fontweight('bold')


def get_fontdict(fontsize=20, fontfamily='Arial', fontweight='bold', include_color=True):
    """
    Create a font dictionary for consistent text styling.

    :param fontsize: Font size.
    :param fontfamily: Font family.
    :param fontweight: Font weight.
    :param include_color: Whether to include color in the fontdict.
    :return: Dictionary with font properties.
    """
    fontdict = { 'family': fontfamily,'size': fontsize,'weight': fontweight}
    if include_color:
        fontdict['color'] = 'black'
    return fontdict


def plot_histogram(data_groups, labels, title, xlabel, ylabel, savepath, filename, cmap=default_cmap):
    """
    Plot a histogram for multiple data groups.

    :param data_groups: List of 1D arrays, each containing data for a group.
    :param labels: List of group labels for the legend.
    :param title: Plot title.
    :param xlabel: X-axis label.
    :param ylabel: Y-axis label.
    :param savepath: Directory to save the plot.
    :param filename: Output filename.
    :param cmap: Colormap for histogram colors.
    """
    colors = plt.cm.get_cmap(cmap)(np.linspace(0, 1, len(data_groups)))

    with plt.style.context('default'):
        fig, ax = plt.subplots(figsize=(10, 6))
        for group, label, color in zip(data_groups, labels, colors):
            if group.size > 0:
                ax.hist(group, bins=50, density=True, alpha=0.5, label=label,
                        histtype='stepfilled', color=color, edgecolor='black')
        configure_axes(ax)
        fontdict = get_fontdict()
        ax.set_title(title, fontdict=fontdict)
        ax.set_xlabel(xlabel, fontdict=fontdict)
        ax.set_ylabel(ylabel, fontdict=fontdict)
        ax.legend(loc='upper right', fontsize=fontsize, frameon=False)
        plt.tight_layout()
        plt.savefig(os.path.join(savepath, filename), dpi=600, bbox_inches='tight')
        plt.close(fig)


def plot_tsne(data_2d, labels, title, savepath, filename, cmap=default_cmap):
    """
    Plot a t-SNE scatter plot.

    :param data_2d: 2D array of t-SNE coordinates (n_samples, 2).
    :param labels: Array of labels for coloring points.
    :param title: Plot title.
    :param savepath: Directory to save the plot.
    :param filename: Output filename.
    :param cmap: Colormap for scatter points.
    """
    with plt.style.context('default'):
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(data_2d[:, 0], data_2d[:, 1], c=labels, cmap=cmap)
        cbar = fig.colorbar(scatter)
        fontdict = get_fontdict()
        cbar.set_label('Predicted Label', fontdict=fontdict)
        cbar.ax.tick_params(labelsize=tick_label_size)
        for label in cbar.ax.get_yticklabels():
            label.set_fontproperties(FontProperties(family='Arial', weight='bold', size=tick_label_size))
        configure_axes(ax)
        ax.set_title(title, fontdict=fontdict)
        ax.set_xlabel('t-SNE Dimension 1', fontdict=fontdict)
        ax.set_ylabel('t-SNE Dimension 2', fontdict=fontdict)
        plt.tight_layout()
        plt.savefig(os.path.join(savepath, filename), dpi=600, bbox_inches='tight')
        plt.close(fig)


def plot_heatmap(data, title, x_step, y_step, xlabel, ylabel, savepath, filename, cmap=default_cmap):
    """
    Plot a heatmap for a weight matrix.

    :param data: 2D array of weights.
    :param title: Plot title.
    :param xlabel: X-axis label.
    :param ylabel: Y-axis label.
    :param savepath: Directory to save the plot.
    :param filename: Output filename.
    :param cmap: Colormap for the heatmap.
    """
    with plt.style.context('default'):
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(data, cmap=cmap, annot=False, ax=ax)
        configure_axes(ax)
        fontdict = get_fontdict()

        # Custom ticks based on data shape
        rows, cols = data.shape
        x_ticks = np.arange(0, cols, x_step)
        y_ticks = np.arange(0, rows, y_step)
        x_labels = [str(i) for i in x_ticks]
        y_labels = [str(i) for i in y_ticks]

        ax.set_xticks(x_ticks + 0.5)  # Center ticks in heatmap cells
        ax.set_yticks(y_ticks + 0.5)
        ax.set_xticklabels(x_labels, fontdict=get_fontdict(fontsize=tick_label_size))
        ax.set_yticklabels(y_labels, fontdict=get_fontdict(fontsize=tick_label_size))
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties(FontProperties(family='Arial', weight='bold', size=tick_label_size))

        ax.set_title(title, fontdict=fontdict)
        ax.set_xlabel(xlabel, fontdict=fontdict)
        ax.set_ylabel(ylabel, fontdict=fontdict)
        cbar = ax.collections[0].colorbar
        for label in cbar.ax.get_yticklabels():
            label.set_fontproperties(FontProperties(family='Arial', weight='bold', size=tick_label_size))
        plt.tight_layout()
        plt.savefig(os.path.join(savepath, filename), dpi=600, bbox_inches='tight')
        plt.close(fig)


def visualize_projection_layer(model, x, task_id, savepath, cmap=default_cmap):
    """
    Visualize the projection layer weights and output distribution.

    :param model: Model with projection layer.
    :param x: Input tensor.
    :param task_id: Task identifier.
    :param savepath: Directory to save plots.
    :param cmap: Colormap for visualization.
    :return: Weights and projected output.
    """
    proj_layer = model.projs[task_id]
    with torch.no_grad():
        weights = proj_layer.weight.cpu().numpy()
        projected_x = proj_layer(x).cpu().numpy().flatten()

    plot_heatmap(weights, 'Projection Layer Weight Matrix', 'Input Dim', 'd_model', 3, 5,
                 savepath, 'proj_weights.png', cmap)
    plot_histogram([projected_x], ['Projected'], 'Projection Layer Output Distribution',
                   'Projected Value', 'Density', savepath, 'proj_output.png', cmap)
    return weights, projected_x


def visualize_tsne(x_out, layer_idx, savepath, cmap=default_cmap):
    """
    Visualize t-SNE of encoder layer output.

    :param x_out: Encoder output tensor (batch, seq_len, d_model).
    :param layer_idx: Layer index.
    :param savepath: Directory to save plot.
    :param cmap: Colormap for scatter points.
    """
    x_flat = x_out.reshape(x_out.shape[0], -1)  # (batch, features)
    tsne = TSNE(n_components=2, random_state=42)
    x_2d = tsne.fit_transform(x_flat)
    plot_tsne(x_2d, np.arange(x_out.shape[0]), f'Encoder Layer {layer_idx} Output t-SNE',
              savepath, f'tsne_layer_{layer_idx}.png', cmap)


def calculate_key_layer(model, x, mask, task_id):
    """
    Calculate key layer outputs and t-SNE visualizations.

    :param model: Model with projection, encoder, and FC layers.
    :param x: Input tensor (batch, seq_len, feature_dim).
    :param mask: Boolean padding mask (batch, seq_len).
    :param task_id: Task identifier.
    :return: Dictionary of layer outputs, weights, and t-SNE results.
    """

    tsne = TSNE(n_components=2, random_state=42)
    with torch.no_grad():
        # Projection layer
        proj_layer = model.projs[task_id]
        proj_output = proj_layer(x)  # (batch, seq_len, d_model)
        proj_flat = proj_output.cpu().numpy()

        proj_2d = proj_flat.reshape(proj_flat.shape[0], -1)  # (batch, seq_len*d_model)
        proj_tsne = tsne.fit_transform(proj_2d)
        proj_weights = proj_layer.weight.cpu().numpy()  # (d_model, feature_dims[task_id])

        x_enc = proj_output + model.positional_encoding[:, :x.size(1), :].to(x.device)
        enc_output = model.transformer_encoder(x_enc.transpose(0, 1), src_key_padding_mask=mask)
        enc_output = enc_output.transpose(0, 1)  # (batch, seq_len, d_model)
        enc_flat = enc_output.cpu().numpy()
        enc_2d = enc_flat.reshape(enc_flat.shape[0], -1)
        enc_tsne = tsne.fit_transform(enc_2d)

        mask_expanded = mask.unsqueeze(-1)
        valid_mask = (~mask_expanded).float()
        pooled_output = (enc_output * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1e-9)  # (batch, d_model)
        fc_layer = model.fcs[task_id]
        fc_output = fc_layer(pooled_output)  # (batch, num_classes)
        fc_flat = fc_output.cpu().numpy()
        fc_tsne = tsne.fit_transform(fc_flat)
        fc_weights = fc_layer.weight.cpu().numpy()  # (num_classes, d_model)
        pred_labels = torch.argmax(fc_output, dim=-1).cpu().numpy()
        probs = torch.softmax(fc_output, dim=-1).cpu().numpy()  # (batch, num_classes)

    return {
        'proj_flat': proj_flat,
        'proj_tsne': proj_tsne,
        'proj_weights': proj_weights,
        'enc_flat': enc_flat,
        'enc_tsne': enc_tsne,
        'fc_flat': fc_flat,
        'fc_tsne': fc_tsne,
        'fc_weights': fc_weights,
        'pred_labels': pred_labels,
        'probs': probs,
    }


def visual_AdapProj(pred_labels, proj_flat, proj_tsne, proj_weights, savepath, cmap=default_cmap):
    """
    Visualize adaptive projection layer outputs and weights.

    :param pred_labels: Predicted labels (n_samples,).
    :param proj_flat: Projection layer output (n_samples, seq_len, d_model).
    :param proj_tsne: t-SNE coordinates (n_samples, 2).
    :param proj_weights: Projection weights (d_model, feature_dim).
    :param savepath: Directory to save plots.
    :param cmap: Colormap for visualization.
    """
    unique_labels = np.unique(pred_labels)
    groups = [proj_flat[pred_labels == label].reshape(-1) for label in unique_labels]
    labels = [f'Class {label}' for label in unique_labels]
    plot_histogram(groups, labels, 'Adaptive Projection Layer Output Distribution',
                   'Value', 'Density', savepath, 'proj_output_dist.png', cmap)
    plot_tsne(proj_tsne, pred_labels, 'Adaptive Projection Layer Output t-SNE',
              savepath, 'proj_tsne.png', cmap)
    plot_heatmap(proj_weights, 'Adaptive Projection Layer Weight Matrix', 3, 5,
                 'Input Dim', 'Hidden Dim', savepath, 'proj_weights.png', cmap)


def visual_Encoder(pred_labels, enc_flat, enc_tsne, savepath, cmap=default_cmap):
    """
    Visualize transformer encoder outputs.

    :param pred_labels: Predicted labels (n_samples,).
    :param enc_flat: Encoder output (n_samples, seq_len, d_model).
    :param enc_tsne: t-SNE coordinates (n_samples, 2).
    :param savepath: Directory to save plots.
    :param cmap: Colormap for visualization.
    """

    unique_labels = np.unique(pred_labels)
    groups = [enc_flat[pred_labels == label].reshape(-1) for label in unique_labels]
    labels = [f'Class {label}' for label in unique_labels]
    plot_histogram(groups, labels, 'Transformer Encoder Output Distribution',
                   'Value', 'Density', savepath, 'enc_output_dist.png', cmap)
    plot_tsne(enc_tsne, pred_labels, 'Transformer Encoder Output t-SNE',
              savepath, 'enc_tsne.png', cmap)


def visual_FClayer(pred_labels, fc_flat, fc_tsne, fc_weights, savepath, cmap=default_cmap):
    """
    Visualize fully connected layer outputs, weights, and probabilities.

    :param pred_labels: Predicted labels (n_samples,).
    :param fc_flat: FC layer output (n_samples, num_classes).
    :param fc_tsne: t-SNE coordinates (n_samples, 2).
    :param fc_weights: FC weights (num_classes, d_model).
    :param savepath: Directory to save plots.
    :param cmap: Colormap for visualization.
    """
    unique_labels = np.unique(pred_labels)
    groups = [fc_flat[pred_labels == label].reshape(-1) for label in unique_labels]
    labels = [f'Class {label}' for label in unique_labels]
    plot_histogram(groups, labels, 'FC Layer Output Distribution',
                   'Value', 'Density', savepath, 'fc_output_dist.png', cmap)
    plot_tsne(fc_tsne, pred_labels, 'FC Layer Output t-SNE',
              savepath, 'fc_tsne.png', cmap)
    plot_heatmap(fc_weights, 'FC Layer Weight Matrix', 3, 1,
                 'Hidden Dim', 'Class', savepath, 'fc_weights.png', cmap)


if __name__ == '__main__':
    path = r'D:\TrajSeg-Cls\TrajSEG-CLS_V3\CLS\Var_LD_500\SNR05_2-3D\Roll_Feature'
    model_path = os.path.join(path, 'AdapTransformer')
    savepath = os.path.join(model_path, 'Visualization/batch0')
    if not os.path.exists(savepath):
        os.mkdir(savepath)

    # Set parameters
    task_id = 0
    batch_size = 2500

    # Hyperparameters (Fixed)
    d_model = 64
    num_layers = 2
    num_heads = 4
    hidden_dim = 128

    if 'QiPan' in path:
        feature_dims = [32]
        num_classes_list = [3]
    elif 'YanYu' in path:
        feature_dims = [46]
        num_classes_list = [3]
    elif 'Endocytosis' in path:
        feature_dims = [46]
        num_classes_list = [4]

    if '4-5D' in path:
        feature_dims = [46]
        num_classes_list = [5]
    elif '2-3D' in path:
        feature_dims = [32]
        num_classes_list = [3]

    # Load test dataset
    test_file = os.path.join(path, 'Roll_test_feature.mat')
    test_mat = scipy.io.loadmat(test_file)
    test_X_data = test_mat['data'].squeeze()
    test_y_data = test_mat['label'].flatten()
    test_X_list = [torch.tensor(x, dtype=torch.float32) for x in test_X_data]
    test_y = torch.tensor(test_y_data, dtype=torch.long)

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
        scaler_task.n_samples_seen_ = 1
        scaler.scalers.append(scaler_task)
        scaler.means.append(scaler_task.mean_)
        scaler.stds.append(scaler_task.scale_)

    # Standardize the test data.
    test_X_lists_scaled = scaler.transform(test_X_list, task_id=task_id)
    test_X_lists_scaled = test_X_lists_scaled[::10]     # Sample data every 10 samples for testing.
    test_y = test_y[::10]

    # Create DataLoaders.
    test_dataset = VariableLengthDataset(test_X_lists_scaled, test_y)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Initialize model
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

    # Test
    predictions, probabilities = test_model(model, test_dataloader, task_id=task_id)
    probabilities = np.array(probabilities)

    model.eval()
    fixed_features = []
    num_samples = test_y.size(0)

    with torch.no_grad():
        for i, (X_batch, y_batch, mask, _) in enumerate(test_dataloader):
            X_batch = X_batch.to(device)
            mask = mask.to(device)
            output = model(X_batch, mask, task_id=task_id)
            fixed_features.extend(output.detach().cpu().numpy())

            task_id = 0
            savename = os.path.join(savepath, 'batch' + str(i))
            if not os.path.exists(savename):
                os.mkdir(savename)

            proj_weight, projected_x = visualize_projection_layer(model, X_batch, task_id, savepath=savename)
            visual_ = calculate_key_layer(model, X_batch, mask, task_id)
            visual_.update({'proj_weight': proj_weight, 'projected_x': projected_x})
            scipy.io.savemat(os.path.join(savename, 'Visualization.mat'), visual_)

            fixed_features = np.array(fixed_features)

    result = scipy.io.loadmat(os.path.join(savepath, 'Visualization.mat'))
    pred_labels = result['pred_labels'].squeeze()
    proj_flat = result['proj_flat']
    proj_tsne = result['proj_tsne']
    proj_weights = result['proj_weights']
    enc_flat = result['enc_flat']
    enc_tsne = result['enc_tsne']
    fc_flat = result['fc_flat']
    fc_tsne = result['fc_tsne']
    fc_weights = result['fc_weights']
    probs = result['probs']

    visual_AdapProj(pred_labels, proj_flat, proj_tsne, proj_weights, savepath)
    visual_Encoder(pred_labels, enc_flat, enc_tsne, savepath)
    visual_FClayer(pred_labels, fc_flat, fc_tsne, fc_weights, probs, savepath)
