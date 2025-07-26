# -*- coding: utf-8 -*-
# @Time    : 2025/5/18 11:30
# @Author  : Dongliang

import os

import numpy as np
import scipy
import shap
import torch
import torch.nn.utils.rnn as rnn_utils
from matplotlib import pyplot as plt

import AAISPT.Traj_cls.utils.config as c
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


def main():
    path = r'D:\TrajSeg-Cls\TrajSEG-CLS_V3\CLS\QiPan_NEW\Roll_Feature'
    model_path = os.path.join(path, 'AdapTransformer\全参微调\Pre-trained model from All dimensional')
    savepath = os.path.join(model_path, r'SHAP_importance')
    if not os.path.exists(savepath):
        os.mkdir(savepath)

    dim, feature_names = c.config_feature(path)
    num_class, label_name = c.config('Exp', path)
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
    # test_X = test_X.tolist()
    test_X = [torch.tensor(i, dtype=torch.float32) for i in test_X]

    lengths = torch.tensor([t.shape[0] for t in test_X])
    padded_data = rnn_utils.pad_sequence(test_X, batch_first=True)
    mask = torch.arange(padded_data.shape[1])[None, :] < lengths[:, None]  # [B, T_max]
    padded_data = padded_data.to(device)
    mask = mask.to(device)

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
    model.eval()

    # Prepare background dataset (randomly select 50 trajectories)
    # try:
    #     train = scipy.io.loadmat(os.path.join(path, 'train_feature_all_stand.mat'))
    #     train_X = [torch.tensor(x, dtype=torch.float32) for x in train['data'].squeeze()]
    #     train_padded = rnn_utils.pad_sequence(train_X, batch_first=True)
    #     idx = np.random.choice(len(train_X), 50, replace=False)
    #     background = train_padded[idx].to(device)

    print("Train data not found, using test data for background")
    idx = np.random.choice(len(test_X), 50, replace=False)
    background = padded_data[idx]
    background_mask = mask[idx]

    # Initialize DeepExplainer
    wrapped_model = WrappedModel(model, task_id=0)
    explainer = shap.DeepExplainer(wrapped_model, background)

    # Compute SHAP values (randomly sample 200 trajectories)
    idx = np.random.choice(len(test_X), 4853, replace=False)
    test_data = padded_data[idx]
    test_mask = mask[idx]
    shap_values = explainer.shap_values(test_data)

    # Apply mask
    shap_values = [sv * test_mask[:, :, None].cpu().numpy() for sv in shap_values]

    # Global importance
    global_importance = np.mean([np.abs(sv).mean(axis=(0, 1)) for sv in shap_values], axis=0)
    class_importance = [np.abs(sv).mean(axis=(0, 1)) for sv in shap_values]

    scipy.io.savemat(os.path.join(savepath, 'SHAP_all.mat'), {
        'shap_values': shap_values,
        'global_importance': global_importance,
        'class_importance': class_importance,
        'test_idx': idx,
        'feature_names': feature_names,
        'label_names': label_name
    })

    # Visualize
    plt.figure(figsize=(10, 5))
    shap.summary_plot(global_importance.reshape(1, -1), feature_names=feature_names, plot_type="bar", show=False)
    plt.title("Global Feature Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(savepath, 'global_shap_summary_plot.pdf'), dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 5))
    for i, imp in enumerate(class_importance):
        plt.bar(np.arange(len(feature_names)) + i * 0.2, imp, width=0.2, label=label_name[i])
    plt.xticks(np.arange(len(feature_names)), feature_names, rotation=45)
    plt.xlabel('Feature')
    plt.ylabel('SHAP Importance')
    plt.title("Per-Class Feature Importance")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(savepath, 'class_shap_summary_plot.pdf'), dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    main()
    # path = r'D:\TrajSeg-Cls\TrajSEG-CLS_V3\CLS\QiPan_NEW\Roll_Feature'
    # model_path = os.path.join(path, 'AdapTransformer\全参微调\Pre-trained model from All dimensional')
    # savepath = os.path.join(model_path, r'SHAP_importance')
    # if not os.path.exists(savepath):
    #     os.mkdir(savepath)
    #
    # data = scipy.io.loadmat(os.path.join(savepath, 'SHAP_all.mat'))
    # idx, shap_values = data['idx'].squeeze(), data['shap_values']
    #
    # test = scipy.io.loadmat(os.path.join(path, 'test_feature_all_stand.mat'))
    # test_X = test['data'].squeeze()
    # test_X = test_X[idx].tolist()
    # dim, feature_names = c.config_feature(path)
    # num_class, label_name = c.config('Exp', path)
    #
    # shap_values = shap_values[0]
    #
    # D = len(feature_names)
    # shap_valid = []
    # input_valid = []
    # for i in range(len(test_X)):
    #     valid_len = len(test_X[i])
    #     shap_valid.append(shap_values[i, :valid_len, :].reshape(-1, D))
    #     input_valid.append(test_X[i][:valid_len].reshape(-1,D))
    #
    # shap_2d = np.concatenate(shap_valid, axis=0)  # (sum(Ti), D)
    # X_2d = np.concatenate(input_valid, axis=0)  # 同上
    #
    # shap.summary_plot(shap_2d, X_2d, show=False, feature_names=feature_names, max_display=D)
    # plt.tight_layout()
    # plt.savefig(os.path.join(savepath, 'shap_summary_plot.pdf'), dpi=300, bbox_inches='tight')
    # plt.close()
