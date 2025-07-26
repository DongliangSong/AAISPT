# -*- coding: utf-8 -*-
# @Time    : 2023/7/4 16:47
# @Author  : Dongliang

import os

import joblib
import scipy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

import AAISPT.Traj_cls.utils.config as c
from AAISPT.Traj_cls.utils.Metric_evaluation import MetricEvaluation

path = r'D:\TrajSeg-Cls\TrajSEG-CLS_V3\CLS\Endocytosis_NEW\5D\Feature_based V1'
savepath = os.path.join(path, 'LR')
if not os.path.exists(savepath):
    os.mkdir(savepath)

mode = 'Exp'
num_class, label_name = c.config(mode=mode, path=path)

# Load data
train = scipy.io.loadmat(os.path.join(path, 'stand_train_feature.mat'))
train_x, train_y = train['data'], train['label'].T.ravel()
val = scipy.io.loadmat(os.path.join(path, 'stand_val_feature.mat'))
val_x, val_y = val['data'], val['label'].T.ravel()

# Create logistic regression object
model = LogisticRegression(
    n_jobs=-1,
    penalty='l1',
    solver='saga'
)

# Train the Models using the training sets and check score
model.fit(train_x, train_y)
model.score(train_x, train_y)

# Save Models
joblib.dump(model, os.path.join(savepath, 'model.pkl'))

# Equation coefficient and Intercept
print('Coefficient: \n', model.coef_)
print('Intercept: \n', model.intercept_)
scipy.io.savemat(os.path.join(savepath, 'coefficient.mat'), {'coefficient': model.coef_})

# Predict Output
predicted = model.predict(val_x)
probs = model.predict_proba(val_x)

# Save predicted label
scipy.io.savemat(os.path.join(savepath, 'cls_pre.mat'),
                 {'clspre': predicted,
                  'clsgt': val_y,
                  'probs': probs
                  }
                 )

# Evaluation model
confusionmat = confusion_matrix(val_y, predicted)
evaluator = MetricEvaluation(num_classes=num_class, label_names=label_name, save_path=savepath)
evaluator.plot_confusion_matrix(confusionmat=confusionmat, normalize=True)
metrics = evaluator.calculate_metrics(gt=val_y, predicted=predicted)
evaluator.write_metric_log(confusion_mat=confusionmat, metrics=metrics)
print('Done!')
