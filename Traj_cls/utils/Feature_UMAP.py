# -*- coding: utf-8 -*-
# @Time    : 2025/3/19 16:06
# @Author  : Dongliang

import os

import matplotlib.pyplot as plt
import pandas as pd
import scipy
import seaborn as sns
import umap
from sklearn.preprocessing import StandardScaler

path = r'D:\TrajSeg-Cls\TrajSEG-CLS_V3\CLS\QiPan_NEW\Roll_Feature'
savepath = path
if not os.path.exists(savepath):
    os.mkdir(savepath)

data = scipy.io.loadmat(os.path.join(path, 'mean_test_features.mat'))
features = data['test']

# 标准化特征
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# UMAP 降维
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
embedding = reducer.fit_transform(features_scaled)
scipy.io.savemat(os.path.join(savepath, 'Features_UMAP_embedding.mat'), {'embedding': embedding})

pre_label = scipy.io.loadmat(os.path.join(path, 'Fine_tuning\Attention_BiLSTM_all_stand-2\cls_pre.mat'))
categories = pre_label['clspre'].squeeze()

# 创建 DataFrame
df = pd.DataFrame(
    {
        'UMAP1': embedding[:, 0],
        'UMAP2': embedding[:, 1],
        'Category': categories
    }
)

# 绘制 UMAP 散点图
plt.figure(figsize=(6, 4), facecolor='white')
sns.scatterplot(data=df, x='UMAP1', y='UMAP2', hue='Category', palette=['#4682B4', '#FFA500', '#2E8B57'])
plt.title('UMAP Visualization of Trajectory Categories', fontsize=14, fontname='Arial')
plt.xlabel('UMAP1')
plt.ylabel('UMAP2')
plt.savefig(os.path.join(savepath, 'umap_visualization.png'), dpi=600, bbox_inches='tight')
plt.show()
