# -*- coding: utf-8 -*-
# @Time    : 2023/10/14 18:53
# @Author  : Dongliang

import os

import joblib
import numpy as np
import scipy

# Load experimental data
data_path = r'D:\UsefulData\TrajSEG-CLS\Exp Demo\19 01072020_2 perfect\6700-8200\update'
test = scipy.io.loadmat(os.path.join(data_path, 'test_feature.mat'))
test_x = test['test_feature']

# Dataset Preprocessing
path = r'D:\UsefulData\TrajSEG-CLS_V2\SNR05\Feature_based_optimal'
mean = scipy.io.loadmat(os.path.join(path, 'stand_mean.mat'))['stand_mean']
var = scipy.io.loadmat(os.path.join(path, 'stand_var.mat'))['stand_var']

test_x = (test_x - mean) / np.sqrt(var)
scipy.io.savemat(os.path.join(data_path, 'stand_test_feature.mat'), {'stand_test_feature': test_x})

# Load model
model_path = os.path.join(path, 'LR')
model = joblib.load(os.path.join(model_path, 'model.pkl'))

# Predict Output
predicted = []
for i in range(test_x.shape[0]):
    b = model.predict(test_x[i, :].reshape(1, -1))
    predicted.append(b)

scipy.io.savemat(os.path.join(data_path, 'predicted.mat'), {'predicted': predicted})

# Save predicted label
for i in range(len(predicted)):
    pre_label = label_name[int(predicted[i] - 1)]

    with open(os.path.join(data_path, 'predicted.txt'), 'a') as f:
        f.write(str(i + 1) + ' segment, predicted label is : ' + pre_label + '\n')

with open(os.path.join(data_path, 'predicted.txt'), 'a') as f:
    f.write('model is : ' + model_path + '\n')
