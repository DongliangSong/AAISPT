# -*- coding: utf-8 -*-
# @Time    : 2025/5/17 22:36
# @Author  : Dongliang

import os

import matplotlib.pyplot as plt
import numpy as np
import scipy
import shap
import torch
import torch.nn.utils.rnn as rnn_utils

from AAISPT.Traj_cls.AdapTransformer.model import MultiTaskTransformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class WrappedModel(torch.nn.Module):
    def __init__(self, model, task_id):
        super().__init__()
        self.model = model
        self.task_id = task_id

    def forward(self, x):
        lengths = (x.abs().sum(-1) > 0).sum(dim=1)
        mask = torch.arange(x.size(1), device=x.device)[None, :] < lengths[:, None]

        return self.model(x, mask, self.task_id)


# 可视化 SHAP 热图
def plot_shap_heatmap(shap_matrix, mask, title='SHAP Time × Feature Heatmap'):
    """
    shap_matrix: [T, D]
    mask: [T] boolean, True for valid time steps
    """
    valid_shap = shap_matrix[mask]
    plt.figure(figsize=(10, 4))
    plt.imshow(valid_shap, aspect='auto', cmap='bwr', interpolation='nearest')
    plt.colorbar(label='SHAP Value')
    plt.xlabel('Feature Dimension')
    plt.ylabel('Time Step')
    plt.title(title)
    plt.tight_layout()
    plt.show()


# def compute_shap_values(model, sequences, lengths, task_id, num_background=50, num_test=10):
#     """
#     使用 SHAP 计算特征重要性
#     :param model: 训练好的模型
#     :param sequences: 序列列表
#     :param lengths: 长度列表
#     :param num_background: 背景样本数
#     :param num_test: 测试样本数
#     :return: SHAP 值
#     """
#     # 选择背景数据（用于近似期望）
#     background_idx = np.random.choice(len(sequences), num_background, replace=False)
#     background_sequences = pad_sequence([sequences[i] for i in background_idx],
#                                         batch_first=True, padding_value=0.0)
#     background_lengths = torch.tensor([lengths[i] for i in background_idx])
#
#     # 选择测试数据
#     test_idx = np.random.choice(len(sequences), num_test, replace=False)
#     test_sequences = pad_sequence([sequences[i] for i in test_idx], batch_first=True, padding_value=0.0)
#     test_lengths = torch.tensor([lengths[i] for i in test_idx])
#
#     # 转换为 Tensor
#     background_sequences = background_sequences.to(device)
#     test_sequences = test_sequences.to(device)
#     background_lengths = background_lengths.to(device)
#     test_lengths = test_lengths.to(device)
#
#     mask = torch.ones(len(lengths), max(lengths), dtype=torch.bool)
#     for i, length in enumerate(lengths):
#         mask[i, :length] = 0
#     mask = mask.to(device)
#
#     # 定义 SHAP 解释器
#     def model_wrapper(x):
#         with torch.no_grad():
#             # 根据输入大小选择长度
#             current_lengths = test_lengths if x.shape[0] == test_sequences.shape[0] else background_lengths
#             # 前向传播
#             outputs = model(x, mask, task_id)
#             # 返回指定 task_id 的输出
#             return outputs
#
#
#     explainer = shap.DeepExplainer(model_wrapper, background_sequences)
#     shap_values = explainer.shap_values(test_sequences)
#
#     return shap_values, test_sequences, test_lengths
#
#
# def plot_shap_values(shap_values, test_sequences, test_lengths, feature_names=None):
#     """
#     可视化 SHAP 值
#     :param shap_values: SHAP 值，形状 [num_classes, num_samples, max_len, num_features]
#     :param test_sequences: 测试序列
#     :param test_lengths: 测试序列长度
#     :param feature_names: 特征名称
#     """
#     num_samples = test_sequences.shape[0]
#     max_len = test_sequences.shape[1]
#     num_features = test_sequences.shape[2]
#
#     if feature_names is None:
#         feature_names = [f"Feature {i}" for i in range(num_features)]
#
#     # 针对第一个类别的 SHAP 值（可根据任务调整）
#     shap_values_class = shap_values[0]  # [num_samples, max_len, num_features]
#
#     # 全局特征重要性（平均绝对 SHAP 值）
#     global_importance = np.zeros(num_features)
#     for i in range(num_samples):
#         for t in range(test_lengths[i]):
#             global_importance += np.abs(shap_values_class[i, t])
#     global_importance /= np.sum(test_lengths)
#
#     # 绘制全局特征重要性
#     plt.figure(figsize=(8, 4))
#     plt.bar(feature_names, global_importance)
#     plt.title("Global Feature Importance (Mean |SHAP|)")
#     plt.xlabel("Features")
#     plt.ylabel("Mean |SHAP|")
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     plt.show()
#
#     # 局部分析：绘制第一个测试样本的 SHAP 热力图
#     sample_idx = 0
#     sample_shap = shap_values_class[sample_idx, :test_lengths[sample_idx]]
#     plt.figure(figsize=(10, 6))
#     sns.heatmap(sample_shap.T, cmap="coolwarm", xticklabels=range(1, test_lengths[sample_idx] + 1),
#                 yticklabels=feature_names, cbar_kws={'label': 'SHAP Value'})
#     plt.title(f"SHAP Values for Sample {sample_idx + 1} (Class 0)")
#     plt.xlabel("Time Step")
#     plt.ylabel("Features")
#     plt.tight_layout()
#     plt.show()


if __name__ == "__main__":
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

    # Load test dataset
    test = scipy.io.loadmat(os.path.join(path, 'test_feature_all_stand.mat'))
    test_X, test_Y = test['data'].squeeze(), test['label'].squeeze()
    test_X = test_X.tolist()
    test_X = [torch.tensor(i, dtype=torch.float32) for i in test_X]
    seq_len = [len(i) for i in test_X]

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

    # Padding & mask 构造
    lengths = torch.tensor([t.shape[0] for t in test_X])
    padded_data = rnn_utils.pad_sequence(test_X, batch_first=True)
    mask = torch.arange(padded_data.shape[1])[None, :] < lengths[:, None]  # [B, T_max]
    padded_data = padded_data.to(device)
    mask = mask.to(device)

    idx = np.random.choice(len(test_X), 50, replace=False)
    background = padded_data[idx]
    background_mask = mask[idx]

    wrapped_model = WrappedModel(model, task_id=0)

    # SHAP 分析器
    explainer = shap.DeepExplainer(wrapped_model, background)

    # 计算 SHAP 值
    idx = np.random.choice(len(test_X), 200, replace=False)
    shap_values = explainer.shap_values(padded_data[idx])  # list of [B, T, D] for each class
    scipy.io.savemat(os.path.join(savepath,'SHAP_importance.mat'),{'shap_values':shap_values})


    shap_val = torch.tensor(shap_values[0])  # shape [B, T, D]

    # 可视化第0条轨迹的 SHAP
    plot_shap_heatmap(shap_val[0].cpu().numpy(), mask[0].cpu().numpy())


