# -*- coding: utf-8 -*-
# @Time    : 2023/5/17 9:30
# @Author  : Dongliang

import os
from itertools import cycle

import numpy as np
import scipy
from matplotlib import pyplot as plt
from numpy import interp
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import StratifiedKFold

# Global config
fontsize = 20
linewidth = 2
labelsize = 20
figsize_ROC = (6, 6)
figsize_feature = (32, 8)
dpi = 600
plt_style = {
    'font.sans-serif': ['Arial'],
    'axes.unicode_minus': False
}

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.weight'] = 'bold'

label_style = {'fontsize': fontsize, 'fontweight': 'bold', 'fontfamily': 'Arial'}


class MetricEvaluation:
    """Class to handle trajectory classification evaluation and visualization."""

    def __init__(self, num_classes, label_names, save_path, model_path):
        """
        Initialize the evaluation class.

        :param num_classes: Number of trajectory categories.
        :param label_names: Names of each category.
        :param save_path: Path to save evaluation results and plots.
        :param model_path: Pathname of the model used.
        """

        self.num_classes = num_classes
        self.label_names = label_names
        self.save_path = save_path
        self.model_path = model_path
        self._ensure_directory()

    def _ensure_directory(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    @staticmethod
    def K_fold_cross_validation(kfold, num_class, model, testX, testY):
        """
        Perform K-fold cross-validation for trajectory classification.

        :param kfold: Number of folds in cross-validation.
        :param num_class: Number of trajectory categories.
        :param model: Model used for prediction.
        :param testX: Test dataset.
        :param testY: Test labels.
        :return: Precision matrix of shape (kfold, num_classes, num_classes).
        """

        skfolds = StratifiedKFold(n_splits=kfold, random_state=42, shuffle=True)
        precision = np.zeros((kfold, num_class, num_class))

        for k, (train_index, test_index) in enumerate(skfolds.split(testX, testY)):
            X_train, X_test = testX[train_index], testX[test_index]
            y_train, y_test = testY[train_index], testY[test_index]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            confusion_mat = confusion_matrix(y_test, y_pred)
            precision[k, :] = confusion_mat / np.sum(confusion_mat, axis=0, keepdims=True)

        return precision

    @staticmethod
    def calculate_metrics(gt, predicted):
        """
        Compute evaluation metrics for multi-class classification based on ground truth and predicted labels.

        :param gt: Ground truth labels, an array of shape (n_samples,) containing integer class indices.
        :param predicted: Predicted labels, an array of shape (n_samples,) with the same format as gt.
        :return: A dictionary containing the following metrics:
                - 'sensitivity': Per-class sensitivity (recall), TP / (TP + FN).
                - 'specificity': Per-class specificity, TN / (TN + FP).
                - 'precision': Per-class precision, TP / (TP + FP).
                - 'each_accuracy': Per-class accuracy, (TP + TN) / total_samples_per_class.
                - 'total_accuracy': Overall accuracy across all classes, sum(TP) / total_samples.
                - 'f1_score': Per-class F1 score, 2 * precision * sensitivity / (precision + sensitivity).
                - 'macro_f1_score': Macro-averaged F1 score across all classes.
        """

        confusionmat = confusion_matrix(gt, predicted)
        TP = np.diag(confusionmat)
        FP = np.sum(confusionmat, axis=0) - TP
        FN = np.sum(confusionmat, axis=1) - TP
        TN = np.sum(confusionmat) - (TP + FP + FN)

        sensitivity = TP / (TP + FN)
        specificity = TN / (TN + FP)
        precision = TP / (TP + FP)

        f1_score = 2 * precision * sensitivity / (precision + sensitivity)
        f1_score = np.nan_to_num(f1_score, nan=0.0)
        macro_f1_score = np.mean(f1_score)

        each_accuracy = (TP + TN) / np.sum(confusionmat)
        total_accuracy = np.sum(np.diag(confusionmat)) / np.sum(confusionmat)

        return {
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'each_accuracy': each_accuracy,
            'total_accuracy': total_accuracy,
            'f1_score': f1_score,
            'macro_f1_score': macro_f1_score
        }

    def plot_confusion_matrix(self, confusionmat, normalize):
        """
        Plot the confusion matrix.

        :param confusionmat: Confusion matrix.
        :param normalize: Whether to normalize the confusion matrix.
        """

        title = 'Normalized Confusion Matrix' if normalize else 'Confusion Matrix'
        mat = confusionmat.copy()
        # Only use the labels that appear in the data
        if normalize:
            row_sums = mat.sum(axis=1, keepdims=True)
            mat = np.round(mat.astype('float') / row_sums, decimals=2)

            # Fix rows where the sum is not 1 due to rounding errors
            for i in range(mat.shape[0]):
                row_sum = mat[i].sum()
                if not np.isclose(row_sum, 1):
                    diff = 1 - row_sum
                    min_index = np.argmin(mat[i])
                    mat[i, min_index] += np.round(diff, decimals=2)

        fig, ax = plt.subplots()
        cax = ax.imshow(mat, interpolation='nearest', cmap='coolwarm', vmin=0, vmax=1)

        # Set axis scales and labels
        tick_marks = np.arange(len(self.label_names))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(self.label_names, rotation=45, **label_style)
        ax.set_yticklabels(self.label_names, rotation=45, **label_style)
        ax.tick_params(axis='both', which='major', width=linewidth, labelsize=fontsize, labelcolor='black')
        ax.set_xlabel('Predicted label', color='black', **label_style)
        ax.set_ylabel('Ground truth', color='black', **label_style)
        ax.set_title(title, color='black', **label_style)

        cbar = plt.colorbar(cax)
        cbar.ax.tick_params(labelsize=labelsize, width=linewidth)
        for label in cbar.ax.get_yticklabels():
            label.set_fontweight('bold')
            label.set_fontsize(labelsize)
            label.set_family('Arial')

        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                plt.text(j, i, f'{mat[i, j]:.2f}' if normalize else f'{int(mat[i, j])}',
                         ha='center', va='center', fontsize=fontsize, fontweight='bold', color='black')

        plt.rcParams.update(plt_style)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, 'ConMat.png'), dpi=600)
        plt.close()

    def plot_confusion_matrix_acc(self, precision):
        """
        Plot confusion matrix using mean and standard deviation of precision.

        :param precision: Precision matrix from cross-validation.
        """
        means = precision.mean(axis=0) * 100
        stds = precision.std(axis=0) * 100

        plt.figure()
        plt.imshow(means, interpolation='nearest', cmap='coolwarm')
        plt.title('Confusion Matrix', fontsize=fontsize)
        tick_marks = np.arange(len(self.label_names))
        plt.xticks(tick_marks, self.label_names, fontsize=fontsize, rotation=45)
        plt.yticks(tick_marks, self.label_names, fontsize=fontsize, rotation=45)

        for i in range(len(self.label_names)):
            for j in range(len(self.label_names)):
                plt.text(j, i, f"{means[i, j]:.2f} ± {stds[i, j]:.2f}",
                         ha='center', va='center', fontsize=fontsize)

        plt.tick_params(labelsize=fontsize)
        plt.rcParams.update(plt_style)
        plt.xlabel('Predicted label', fontsize=fontsize)
        plt.ylabel('Ground truth', fontsize=fontsize)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, 'Precision.png'), dpi=600)
        plt.close()

    def plot_roc_curve(self, gt, predictions):
        """
        Plot ROC curves for multi-class classification.

        :param gt: Ground truth labels.
        :param predictions: Predicted probabilities.
        """

        fpr, tpr, roc_auc = {}, {}, {}
        for i in range(self.num_classes):
            fpr[i], tpr[i], _ = roc_curve(gt[:, i], predictions[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Micro-average ROC
        fpr['micro'], tpr['micro'], _ = roc_curve(gt.ravel(), predictions.ravel())
        roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])
        print('micro-average AUC:\n', roc_auc)

        # Macro-average ROC
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(self.num_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(self.num_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        mean_tpr /= self.num_classes
        fpr['macro'] = all_fpr
        tpr['macro'] = mean_tpr
        roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])

        # Plot all ROC curves
        plt.figure(figsize=figsize_ROC)
        colors = cycle(['darkred', 'darkorange', 'darkblue', 'k', 'brown'])
        for i, color, label in zip(range(self.num_classes), colors, self.label_names):
            plt.plot(fpr[i], tpr[i], color=color, lw=linewidth,
                     label=f'{label} (AUC = {roc_auc[i]:.3f})')
        # plt.plot(fpr['micro'], tpr['micro'], color='green',
        #          lw=lw, label='Micro ROC (area = %0.3f)' % roc_auc['micro'])
        #
        # plt.plot(fpr['macro'], tpr['macro'], 'cornflowerblue',linestyle='--',
        #          lw=lw, label='Macro ROC (area = %0.3f)' % roc_auc['macro'])

        plt.plot([0, 1], [0, 1], 'navy', lw=linewidth, linestyle='--')
        plt.axis('equal')
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate', color='black', **label_style)
        plt.ylabel('True Positive Rate', color='black', **label_style)
        plt.title('ROC for multi-class', color='black', **label_style)

        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_linewidth(linewidth)

        ax.tick_params(axis='both', which='major', labelsize=fontsize, width=linewidth, labelcolor='black')
        plt.legend(loc='lower right', prop={'family': 'Arial', 'weight': 'bold', 'size': fontsize})

        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, 'ROC.png'), dpi=600)
        plt.close()

    def plot_features(self, feature_names, feature_importance):
        """
        Plot feature importance.

        :param feature_names: Names of all features.
        :param feature_importance: Importance values for all features.
        :return:
        """

        plt.figure(figsize=figsize_feature)
        plt.barh(feature_names, feature_importance, height=0.7, color='#008792', edgecolor='#005344')
        plt.xlabel('Feature Importance', fontsize=fontsize)
        plt.ylabel('Features', fontsize=fontsize)
        plt.title('Feature Importance')
        plt.rcParams.update(plt_style)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, 'Feature-importance.png'), dpi=600)
        plt.close()

    def write_feature_importance(self, feature_names, feature_importance):
        """
        Write feature importance to a .txt file.

        :param feature_names: Names of all features.
        :param feature_importance: Importance values for all features.
        """

        with open(os.path.join(self.save_path, 'Feature-importance.txt'), 'a') as f:
            for name, importance in zip(feature_names, feature_importance):
                f.write(f"{name} : {importance}\n")

    # def write_metric_log(self, confusion_mat, sensitivity, specificity,
    #                      precison, each_accuracy, total_accuracy, f1_score,
    #                      macro_f1_score) :
    #
    #     """
    #     Write classification evaluation results to a .txt file.
    #
    #     Args:
    #         confusion_mat: Confusion matrix.
    #         sensitivity: Sensitivity for each class.
    #         specificity: Specificity for each class.
    #         precison: Precision for each class.
    #         each_accuracy: Accuracy for each class.
    #         total_accuracy: Overall accuracy.
    #         f1_score: F1 score for each class.
    #         macro_f1_score: Macro-averaged F1 score.
    #     """
    #     with open(os.path.join(self.save_path, 'metric.txt'), 'a') as f:
    #         print(f"Confusion Matrix:\n{confusion_mat}", file=f)
    #         print("=" * 50, file=f)
    #         print(f"Sensitivity:\n{sensitivity}", file=f)
    #         print("=" * 50, file=f)
    #         print(f"Specificity:\n{specificity}", file=f)
    #         print("=" * 50, file=f)
    #         print(f"Precision:\n{precison}", file=f)
    #         print("=" * 50, file=f)
    #         print(f"Each_accuracy:\n{each_accuracy}", file=f)
    #         print("=" * 50, file=f)
    #         print(f"Total Accuracy:\n{total_accuracy}", file=f)
    #         print("=" * 50, file=f)
    #         print(f"F1 Score:\n{f1_score}", file=f)
    #         print("=" * 50, file=f)
    #         print(f"Macro F1 Score:\n{macro_f1_score}", file=f)
    #         print(f"Model Path: {self.model_path}", file=f)
    #         print(f"Save Path: {self.save_path}", file=f)

    def write_metric_log(self, confusion_mat, metrics, metric_names):
        """
        Write evaluation metrics to a log file.

        :param confusion_mat: Confusion matrix.
        :param metrics: Dictionary of metric names and values, or a tuple of metric values.
        :param metric_names: List of metric names (required if metrics is a tuple).
        """

        with open(os.path.join(self.save_path, 'metric.txt'), 'a') as f:
            f.write('Confusion Matrix:\n')
            f.write(str(confusion_mat) + '\n')
            f.write('=' * 50 + '\n')

            if isinstance(metrics, dict):
                for metric_name, metric_value in metrics.items():
                    f.write(f'{metric_name}: {metric_value}\n')
                    f.write('=' * 50 + '\n')
            elif isinstance(metrics, (tuple, list)) and metric_names:
                for name, value in zip(metric_names, metrics):
                    f.write(f'{name}: {value}\n')
                    f.write('=' * 50 + '\n')
            else:
                raise ValueError("Metrics must be a dict, or a tuple/list with corresponding metric_names")

    def evaluation(self, gt, predictions, normalize=True):
        """
        Evaluate classification results.

        :param gt: Ground truth labels.
        :param predictions: Predicted labels.
        :param normalize: Whether to normalize the confusion matrix.
        """

        confusionmat = confusion_matrix(gt, predictions)

        # # Calculate Metric
        # metrics = self.calculate_metrics(gt, predictions)
        #
        # self.write_metric_log(confusionmat, metrics)

        # Plot confusion matrix
        self.plot_confusion_matrix(confusionmat, normalize=normalize)

        # # Plot ROC curve
        # true_label = label_binarize(gt, classes=np.arange(self.num_classes))
        # self.plot_roc_curve(true_label, probs)


if __name__ == '__main__':
    # root = r'D:\TrajSeg-Cls\TrajSEG-CLS_V3\CLS\Endocytosis_NEW\5D\Feature_based V1'
    # paths = [
    #     os.path.join(root, 'DecisionTree'),
    #     os.path.join(root, 'GBDT'),
    #     os.path.join(root, 'KNN'),
    #     os.path.join(root, 'LR'),
    #     os.path.join(root, 'RandomForest'),
    #     os.path.join(root, 'SVM_Linear'),
    #     os.path.join(root, 'SVM_Poly'),
    #     os.path.join(root, 'SVM_Rbf'),
    # ]

    paths = [
        r'D:\TrajSeg-Cls\TrajSEG-CLS_V3\CLS\YanYu_NEW\Roll_Feature\AdapTransformer\全参微调\Pre-trained model from All dimensional'
    ]
    for path in paths:
        model_path = path
        save_path = path

        mode = 'Exp'

        # Load datasets
        classification = scipy.io.loadmat(os.path.join(path, 'cls_pre.mat'))
        pre = classification['clspre'].squeeze()
        probs = classification['probs']
        gt = classification['clsgt'].squeeze()

        if mode == 'Sim':
            if '2D' in path or '3D' in path:
                num_class = 3
                label_name = ['ND', 'CD', 'DM']
            elif '4D' in path or '5D' in path:
                num_class = 5
                label_name = ['ND', 'TA', 'TR', 'DMR', 'DM']

        elif mode == 'Exp':
            if 'YanYu' in path or 'Janus' in path:
                num_class = 3
                label_name = ['Circling', 'Confined', 'Rocking']

            elif 'QiPan' in path or 'Phase_separation' in path:
                num_class = 3
                label_name = ['Semi-fluidic', 'Transition', 'Non-fluidic']

            elif 'endocytosis' in path.lower():
                num_class = 4
                label_name = ['EI', 'CCP_F', 'CCP_M', 'VR']
        else:
            raise ValueError(f'Unsupported mode: {mode}')

        # Initialize evaluator and run evaluation
        evaluator = MetricEvaluation(num_class, label_name, save_path, model_path)
        evaluator.evaluation(gt=gt, predictions=pre, normalize=True)
