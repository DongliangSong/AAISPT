# -*- coding: utf-8 -*-
# @Time    : 2025/4/14 16:07
# @Author  : Dongliang

import os

import matplotlib.pyplot as plt
import numpy as np

linewidth = 2
fontsize = 20
models = ['DT', 'RF', 'GBDT', 'KNN', 'LR', 'Linear SVM', 'Poly SVM', 'RBF SVM']

# savepath = r'D:\TrajSeg-Cls\TrajSEG-CLS_V3\CLS\QiPan_NEW\Feature_based V1'
# values = [0.85077, 0.8996, 0.90512, 0.80768, 0.71959, 0.73424, 0.83634, 0.86011]  #2D

# savepath = 'D:\TrajSeg-Cls\TrajSEG-CLS_V3\CLS\YanYu_NEW\Feature_based V1'
# values = [0.95551,0.98814,0.98814,0.96985,0.79783,0.81414,0.96722,0.9822]   #4D

savepath = r'D:\TrajSeg-Cls\TrajSEG-CLS_V3\CLS\Endocytosis_NEW\5D\Feature_based V1'
values = [0.91058, 0.95767, 0.94557, 0.91058, 0.71577, 0.76241, 0.92383, 0.93348]  # 5D

cmap = plt.get_cmap('coolwarm')
colors = cmap(np.linspace(0, 1, len(models)))

# Draw the performance comparison of the models
plt.figure(figsize=(10, 6))
bars = plt.bar(models, values, color=colors, edgecolor='black')
plt.title('Model Performance Comparison', fontsize=fontsize, fontweight='bold', fontfamily='Arial')
plt.ylabel('Accuracy', fontsize=fontsize, fontweight='bold', fontfamily='Arial')
plt.ylim([0, 1])
plt.xticks(rotation=45, ha='right', fontsize=fontsize, fontweight='bold', fontfamily='Arial')
plt.yticks(fontsize=fontsize, fontweight='bold', fontfamily='Arial')

ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(linewidth)

plt.tight_layout()
plt.savefig(os.path.join(savepath, 'ML_comparison.tif'), dpi=600)
plt.show()

##############
# Plot top-10 features
# features = [
#     'D', 'Sum_SL', 'Mean_msds', 'alpha', 'Volume_area',
#     'Std_SL', 'Mean_signdot_p', 'MSDratio', 'Mean_SL', 'Gaussianity'
# ]
# values = [0.08898, 0.06303, 0.04933, 0.04721, 0.02999,0.02496, 0.01916, 0.01901, 0.01882, 0.0175]
# stds = [0.00433, 0.00431, 0.00366, 0.00164, 0.00209, 0.00176, 0.00331, 0.00184, 0.00123, 0.00142]

# features = [
#     'mean_Δφ', 'Var_coeff_Δφ', 'std_Δφ', 'mean_Δθ', 'Trappedness',
#     'std_Δθ', 'Kurtosis_Δθ', 'Mean_msds', 'Mean_dot', 'Kurtosis_Δφ'
# ]
# values = [0.00669, 0.00579, 0.00467, 0.00395, 0.00352,0.00292, 0.00289, 0.00283, 0.00247, 0.00219]
# stds = [ 0.00234, 0.00142, 0.00125, 0.000932238, 0.00128,  0.000427362, 0.000601695, 0.00178, 0.000328429, 0.00125]

# 特征名称
features = [
    'std_Δφ', 'mean_Δφ', 'MSDratio', 'Stat_Δφ', 'alpha',
    'mean_Δθ', 'Gaussianity', 'Skewness_Δφ', 'Stat_SL', 'Var_coeff_Δφ'
]

values = [0.00624, 0.00506, 0.00371, 0.00309, 0.0023, 0.00207, 0.00203, 0.00163, 0.00149, 0.00148]
stds = [0.00118, 0.00113, 0.00265, 0.00222, 0.0012, 0.00134, 0.000971597, 0.00107, 0.00124, 0.00101]

cmap = plt.get_cmap('coolwarm')
colors = cmap(np.linspace(0, 1, len(features)))
plt.figure(figsize=(10, 6))
bars = plt.bar(features, values, yerr=stds, capsize=5, color=colors, edgecolor='black', ecolor='black')
plt.title('Top-10 Feature Importance', fontsize=fontsize, fontweight='bold', fontfamily='Arial')
plt.xlabel('Features', fontsize=fontsize, fontweight='bold', fontfamily='Arial')
plt.ylabel('Importance Score', fontsize=fontsize, fontweight='bold', fontfamily='Arial')
plt.xticks(rotation=45, ha='right', fontsize=fontsize, fontweight='bold', fontfamily='Arial')
plt.yticks(fontsize=fontsize, fontweight='bold', fontfamily='Arial')

ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(linewidth)

plt.tight_layout()
plt.savefig(os.path.join(savepath, 'Feature_importance.tif'), dpi=600)
plt.show()
