# -*- coding: utf-8 -*-
# @Time    : 2023/10/18 22:52
# @Author  : Dongliang

import os

import matplotlib.pyplot as plt
import numpy as np
import scipy

import AAISPT.Traj_cls.utils.config as c


def configure_axes(ax, linewidth=2, tick_label_size=16):
    """
    Configure axes with standardized settings.

    :param ax: Matplotlib axes object to configure.
    :param linewidth: Line width for spines.
    :param tick_label_size: Font size for tick labels.
    """
    ax.spines['bottom'].set_linewidth(linewidth)
    ax.spines['left'].set_linewidth(linewidth)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=tick_label_size, width=2)


def get_fontdict(fontsize=20, fontfamily='Arial', fontweight='bold', include_color=True):
    """
    Create a font dictionary for consistent text styling.

    :param fontsize: Font size.
    :param fontfamily: Font family.
    :param fontweight: Font weight.
    :param include_color: Whether to include color in the fontdict.
    :return: Dictionary with font properties.
    """
    fontdict = {
        'family': fontfamily,
        'size': fontsize,
        'weight': fontweight
    }
    if include_color:
        fontdict['color'] = 'black'
    return fontdict


def plot_feature_histogram(feature_data, labels, feature_name, label_names, savepath, colors, fontsize=20):
    """
    Plot a histogram for a given feature across different classes.

    :param feature_data: 1D array of feature values for all traces.
    :param labels: 1D array of class labels for each trace.
    :param feature_name: Name of the feature (used for xlabel and filename).
    :param label_names: List of class names for the legend.
    :param savepath: Directory path to save the plot.
    :param colors: List of colors for each class histogram.
    :param fontsize: Font size for labels and legend.
    """

    # Create figure
    with plt.style.context('default'):
        fig, ax = plt.subplots(figsize=(10, 8))

        # Group data by class
        num_classes = len(label_names)
        class_data = [feature_data[labels == j] for j in range(num_classes)]

        # Determine bin edges
        data_min = min(np.min(group) for group in class_data if group.size > 0)
        data_max = max(np.max(group) for group in class_data if group.size > 0)
        bins = np.linspace(data_min, data_max, 30)

        # Plot histograms
        for j, (class_group, color, label) in enumerate(zip(class_data, colors, label_names)):
            if class_group.size > 0:
                alpha = 1 - 0.2 * j
                ax.hist(class_group, bins=bins, edgecolor='black', histtype='bar', alpha=alpha,
                        label=label, density=True, color=color)

        # Configure axes and labels
        configure_axes(ax)
        fontdict = get_fontdict(fontsize=fontsize)
        ax.set_xlabel(feature_name, fontdict=fontdict)
        ax.set_ylabel('Density', fontdict=fontdict)
        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontfamily('Arial')
            label.set_fontweight('bold')
        plt.tight_layout()


if __name__ == '__main__':
    path = r'D:\TrajSeg-Cls\TrajSEG-CLS_V3\CLS\Endocytosis_NEW\5D\Feature_based V1'
    savepath = os.path.join(path, 'Feature_img')
    if not os.path.exists(savepath):
        os.mkdir(savepath)

    dim, feature_names = c.config_feature(path=path)

    # Load feature
    feature = scipy.io.loadmat(os.path.join(path, 'test_feature.mat'))
    data, label = feature['data'], feature['label'].squeeze()
    num_trace, num_features = data.shape

    mode = 'Exp'
    num_class, label_name = c.config(mode=mode, path=path)
    num_perclass = num_trace // num_class

    colors = plt.cm.coolwarm(np.linspace(0, 1, num_class))
    fontsize = 20
    for i in range(num_features):
        plot_feature_histogram(data[:, i], label, feature_names[i], label_name, savepath, colors)
        # da = []
        # for j in range(num_class):
        #     da.append(data[label == j, i])
        #
        # data_min = min(min(group) for group in da)
        # data_max = max(max(group) for group in da)
        # bin = np.linspace(data_min, data_max, 30)
        #
        # for j, k in enumerate(da):
        #     alpha = 1 - 0.1 * j
        #     plt.hist(k, bins=bin, edgecolor='r', histtype='bar', alpha=alpha, label=label_name[j], density=True)
        #
        # plt.tick_params(labelsize=fontsize)
        # plt.rcParams['font.sans-serif'] = ['Arial']
        # plt.rcParams['axes.unicode_minus'] = False
        # plt.xlabel(feature_names[i], fontsize=fontsize)
        # plt.ylabel('Counts', fontsize=fontsize)
        # plt.xticks(fontsize=fontsize)
        # plt.yticks(fontsize=fontsize)
        # plt.legend(loc='upper right', fontsize=fontsize)
        # plt.savefig(os.path.join(savepath, feature_names[i] + '.png'), dpi=600)

        feature_names[i] = feature_names[i].replace(' ', '_').replace('/', '_')
        plt.savefig(os.path.join(savepath, f'{i}_{feature_names[i]}.png'), dpi=600, bbox_inches='tight')
