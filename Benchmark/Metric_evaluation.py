# -*- coding: utf-8 -*-
# @Time    : 2025/1/2 20:53
# @Author  : Dongliang

from AAISPT.Traj_cls.utils.Metric_evaluation import *

path = r'.\data\CLS\Andi\3D\AdapTransformer\Evaluation'
model_path = r'..\data\CLS\Andi\3D\AdapTransformer'
save_path = path
if not os.path.exists(save_path):
    os.mkdir(save_path)

# Load datasets
classification = scipy.io.loadmat(os.path.join(path, 'cls_pre.mat'))
pre = classification['clspre'].squeeze()
probs = classification['probs']
GT = classification['clsgt'].squeeze()

num_class = 5
label_name = ['ATTM', 'CTRW', 'FBM', 'LW', 'SBM']

evaluator = MetricEvaluation(num_classes=num_class, label_names=label_name, save_path=save_path, model_path=model_path)
evaluator.evaluation(gt=GT, predictions=pre, normalize=True)
