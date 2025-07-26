# -*- coding: utf-8 -*-
# @Time    : 2024/8/1 11:12
# @Author  : Dongliang


import os

import joblib
import matplotlib.pyplot as plt
import scipy
import shap

import AAISPT.Traj_cls.utils.config as c

shap.initjs()

mode = 'Exp'
path = r'D:\TrajSeg-Cls\TrajSEG-CLS_V3\CLS\QiPan_NEW\Feature_based'
model_path = os.path.join(path, 'LR')
savepath = model_path

num_class, label_name = c.config(mode=mode, path=path)
dim, feature_names = c.config_feature(path=path)

# Load dataset
train = scipy.io.loadmat(os.path.join(path, 'stand_train_feature.mat'))
val = scipy.io.loadmat(os.path.join(path, 'stand_val_feature.mat'))
train_X, train_Y = train['data'].squeeze(), train['label'].squeeze()
val_X, val_Y = val['data'].squeeze(), val['label'].squeeze()

# Load model
model = joblib.load(os.path.join(model_path, 'model.pkl'))

K = 100
train_X = shap.sample(train_X, K)

# Initialize SHAP explainer
if 'LR' in model_path:
    explainer = shap.LinearExplainer(model, train_X)
    shap_values = explainer.shap_values(val_X)

elif 'SVM' in model_path:
    explainer = shap.KernelExplainer(model.predict_proba, train_X)
    shap_values = explainer.shap_values(val_X)

elif 'Decision_tree' in model_path or 'RandomForest' in model_path:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(val_X)

# plt.figure()
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = 13

# Plot summary plot
# The type can be bar, dot, violin.
shap.summary_plot(shap_values, val_X, feature_names=feature_names, plot_type='bar', max_display=train_X.shape[1],
                  show=False, class_names=label_name)
plt.legend(loc='lower right')
plt.savefig(os.path.join(savepath, 'shap_summary.pdf'), dpi=600, bbox_inches='tight')
plt.show()

# Plotted separately according to sample type.
for i in range(num_class):
    shap.summary_plot(shap_values[i], val_X, feature_names=feature_names, plot_type='violin',
                      max_display=train_X.shape[1], show=False)
    plt.savefig(os.path.join(savepath, str(i) + "shap_summary.pdf"), dpi=600, bbox_inches='tight')
    plt.show()
