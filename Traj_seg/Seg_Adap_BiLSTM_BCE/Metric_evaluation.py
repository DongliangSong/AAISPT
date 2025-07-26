# -*- coding: utf-8 -*-
# @Time    : 2024/5/23 15:18
# @Author  : Dongliang

import os

import matplotlib.pyplot as plt
import numpy as np
import scipy
from matplotlib.font_manager import FontProperties

labelsize = 20


def configure_axes(ax, linewidth=2, grid_alpha=0.7, tick_label_size=16):
    """
    Configure axes with standardized settings.

    :param ax: Matplotlib axes object to configure.
    :param spine_linewidth: Line width for spines.
    :param grid_alpha: Alpha value for grid lines.
    :param tick_label_size: Font size for tick labels.
    """
    ax.spines['bottom'].set_linewidth(linewidth)
    ax.spines['left'].set_linewidth(linewidth)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.grid(True, linestyle='--', alpha=grid_alpha)
    ax.tick_params(axis='both', which='major', labelsize=tick_label_size, width=2)


def get_fontdict(fontsize=20, fontfamily='Arial', fontweight='bold', include_color=True):
    """
    Create a font dictionary for consistent text styling.
    """
    fontdict = {'family': fontfamily, 'size': fontsize, 'weight': fontweight}
    if include_color:
        fontdict['color'] = 'black'
    return fontdict


def metric_seg(predicted, GT, trace_len, savepath):
    """
    Evaluation of segmentation results.

    :param predicted: Predicted segmentation results.
    :param GT: Ground truth segmentation labels.
    :param trace_len: Length of the trace used for segmentation.
    :param savepath: Path to save the evaluation results.
    """

    nums = max(predicted.size, GT.size)
    pre = np.round(predicted[:nums])
    GT = GT[:nums]

    # Sort by ground truth
    indices = np.argsort(GT)
    pre = pre[indices]
    GT = GT[indices]

    # Calculate RMSE for each segment
    num_segments = trace_len // 20
    RMSE = np.zeros(num_segments)
    for i in range(num_segments):
        index = (i * 20 <= GT) & (GT < (i + 1) * 20)
        RMSE[i] = np.sqrt(np.mean((pre[index] - GT[index]) ** 2))

    # Create plot
    with plt.style.context('default'):
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(np.arange(num_segments), RMSE, marker='o', markersize=6, linewidth=2)

        # Configure axes and labels
        configure_axes(ax)
        fontdict = get_fontdict()
        ax.set_xlabel('Switch Point', fontdict=fontdict)
        ax.set_ylabel('RMSE', fontdict=fontdict)
        ax.set_title('Segmentation RMSE', fontdict=fontdict)

        # Set x-tick labels
        str_list = [f'{i * 20} - {((i + 1) * 20)}' for i in range(num_segments)]
        ax.set_xticks(np.arange(num_segments))
        ax.set_xticklabels(str_list, rotation=45, ha='right', fontdict=fontdict)

        ax.yaxis.set_tick_params(labelsize=labelsize)
        for label in ax.get_yticklabels():
            label.set_fontfamily('Arial')
            label.set_fontweight('bold')

        # Save and show plot
        plt.tight_layout()
        plt.savefig(os.path.join(savepath, 'segRMSE.png'), dpi=600, bbox_inches='tight')
        scipy.io.savemat(os.path.join(savepath, 'segRMSE.mat'), {'RMSE': RMSE})


def plot_2Dhistogram(predicted, GT, savepath):
    """
    Plot a 2D histogram comparing predicted and ground truth values.

    :param predicted: Predicted values (1D array).
    :param ground_truth: Ground truth values (1D array).
    :param savepath: Directory path to save the plot.
    """
    # Create plot
    with plt.style.context('default'):
        fig, ax = plt.subplots(figsize=(10, 6))
        hb = ax.hexbin(predicted, GT, gridsize=50, cmap='coolwarm', mincnt=1)

        # Configure axes
        configure_axes(ax)
        ax.set_xlim(0, 200)
        ax.set_ylim(0, 200)
        ax.set_aspect('equal', adjustable='box')

        # Set x and y ticks
        tick_values = np.arange(0, 201, 25)
        ax.set_xticks(tick_values)
        ax.set_yticks(tick_values)
        ax.set_xticklabels([str(int(v)) for v in tick_values], fontdict=get_fontdict(fontsize=16))
        ax.set_yticklabels([str(int(v)) for v in tick_values], fontdict=get_fontdict(fontsize=16))

        # Add colorbar
        cbar = fig.colorbar(hb)
        cbar.set_label('Counts', fontdict=get_fontdict())
        cbar.ax.tick_params(labelsize=16)
        for label in cbar.ax.get_yticklabels():
            fontprops = FontProperties(**get_fontdict(fontsize=16, include_color=False))
            label.set_fontproperties(fontprops)
            label.set_color('black')

        # Set labels and title
        fontdict = get_fontdict()
        ax.set_title('2D Histogram', fontdict=fontdict)
        ax.set_xlabel('Predicted', fontdict=fontdict)
        ax.set_ylabel('Ground Truth', fontdict=fontdict)
        plt.grid(True, linestyle='--', alpha=0.7)

        # Save and close
        plt.tight_layout()
        plt.savefig(os.path.join(savepath, '2d_histogram.png'), dpi=600, bbox_inches='tight')


if __name__ == '__main__':
    path = r'..\model\2Dtest'
    savepath = path
    if not os.path.exists(savepath):
        os.mkdir(savepath)

    segmat = scipy.io.loadmat(os.path.join(path, 'seg_pre.mat'))

    predicted, GT = segmat['pre'].squeeze(), segmat['gt'].squeeze()
    metric_seg(predicted=predicted, GT=GT, trace_len=200, savepath=path)
    pre = segmat['pre'].squeeze()
    GT = segmat['gt'].squeeze()
    plot_2Dhistogram(pre, GT, savepath)
