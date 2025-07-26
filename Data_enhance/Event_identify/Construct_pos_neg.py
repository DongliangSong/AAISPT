# -*- coding: utf-8 -*-
# @Time    : 2025/1/10 23:15
# @Author  : Dongliang

import os
import random
from collections import defaultdict

import scipy

from Double_check import *


class TrajectoryAugmentation:
    def __init__(self, window_sizes=None, window_strides=None,pos_iou_threshold=0.5,neg_iou_threshold=0.3):
        if window_strides is None:
            window_strides = 5
        if window_sizes is None:
            window_sizes = [10, 20, 30, 40]
        self.window_sizes = window_sizes
        self.window_strides = window_strides
        self.pos_iou_threshold = pos_iou_threshold
        self.neg_iou_threshold = neg_iou_threshold

    def calculate_iou(self, segment1, segment2):
        """Calculate the IoU for both time segments."""
        start1, end1 = segment1
        start2, end2 = segment2

        intersection_start = max(start1, start2)
        intersection_end = min(end1, end2)

        if intersection_end <= intersection_start:
            return 0.0

        intersection = intersection_end - intersection_start
        union = (end1 - start1) + (end2 - start2) - intersection

        return intersection / union

    def generate_samples(self, trajectory, event_segment):
        """
        Generate positive and negative samples.

        :param trajectory: Track data, shape is (n_timestamps, n_features)
        :param event_segment: The start and end position of the target event fragment, in the format (start_idx, end_idx).
        :return: A dictionary containing positive and negative samples in the format:
            {
                'positive': [[sample1, label1]), [sample2, label2], ...],
                'negative': [[sample1, label1]), [sample2, label2], ...]
            }
        """

        samples = defaultdict(list)

        for window_size, stride in zip(self.window_sizes, self.window_strides):
            for start_idx in range(0, len(trajectory) - window_size + 1, stride):
                end_idx = start_idx + window_size
                current_segment = (start_idx, end_idx)
                current_data = trajectory[start_idx:end_idx]

                similarity = self.calculate_iou(current_segment, event_segment)

                if similarity >= self.pos_iou_threshold:
                    label = 1
                    samples['positive'].append([current_data, label])
                elif similarity <= self.neg_iou_threshold:
                    label = 0
                    samples['negative'].append([current_data, label])

        if len(samples['positive']) != len(samples['negative']):
            positive_samples = samples['positive']
            negative_samples = samples['negative']
            positive_samples, negative_samples = self.balance_samples(positive_samples, negative_samples,
                                                                      method='undersample')
            samples['positive'] = positive_samples
            samples['negative'] = negative_samples

        return samples

    def balance_samples(self, positive_samples, negative_samples, method='undersample'):
        """
        Balance the number of positive and negative samples. Oversampling and undersampling are supported.

        :param positive_samples: List of positive samples.
        :param negative_samples: List of negative samples.
        :param method: Str, select the method 'oversample' or 'undersample'.
        :return: Return the balanced positive and negative samples.
        """

        if method == 'oversample':
            # Oversampling: Copying positive samples until the number of negative samples is equal
            num_positives = len(positive_samples)
            num_negatives = len(negative_samples)

            if num_positives < num_negatives:
                positive_samples = positive_samples * (num_negatives // num_positives) + positive_samples[:(
                        num_negatives % num_positives)]

        elif method == 'undersample':
            # Undersampling: Reduce the number of negative samples until the number of positive samples is equal
            num_positives = len(positive_samples)
            num_negatives = len(negative_samples)

            if num_negatives > num_positives:
                negative_samples = random.sample(negative_samples, num_positives)

        return positive_samples, negative_samples

    def adaptive_window_sizes(self, trajectory, event_segment):
        """Adaptive determination of sliding window size."""
        event_length = event_segment[1] - event_segment[0]
        trajectory_length = len(trajectory)

        # Adaptive window size based on event length
        base_sizes = [int(event_length.item() * i * 0.1) for i in range(5, 15)]  # 0.5 fold ~ 1.5 fold

        # Make sure the window size does not exceed the track length
        window_sizes = [size for size in base_sizes if size < trajectory_length]

        if not window_sizes:
            window_sizes = [trajectory_length // 2]

        return window_sizes


if __name__ == '__main__':
    path = r'D:\TrajSeg-Cls\endoysis\Aug_SPT_FLU'
    origin_trace = scipy.io.loadmat(os.path.join(path, 'origin_set.mat'))

    trace = origin_trace['trace'].squeeze()
    tf_time = origin_trace['tf_time']

    aug_data = defaultdict(list)
    nums = trace.shape[0]
    for i in range(nums):
        traj = trace[i]

        # Interpolate the trajectory if necessary.
        interpolator = DataInterpolation()
        traj = interpolator.interpolate_array(traj)

        tf_t = tf_time[i, 1:]
        t = traj[:, 0]
        tolerance = 1e-10
        start = np.where(abs(t - tf_t[0]) < tolerance)[0][0]
        end = np.where(abs(t - tf_t[1]) < tolerance)[0][0]

        true_twist_event = traj[start: end + 1, 2:]
        trajectory = traj[:, 2:]
        event_segment = [start, end]

        # Initializes the data augmentor
        augmentor = TrajectoryAugmentation()

        # Adaptive determination of window size.
        adaptive_windows = augmentor.adaptive_window_sizes(trajectory, event_segment)
        augmentor.window_sizes = adaptive_windows
        augmentor.window_strides = [w // 5 for w in adaptive_windows]

        samples = augmentor.generate_samples(trajectory, event_segment)
        print(f"Generated {len(samples['positive'])} positive samples")
        print(f"Generated {len(samples['negative'])} negative samples")

        aug_data['positive'].extend(samples['positive'])
        aug_data['negative'].extend(samples['negative'])

    scipy.io.savemat(os.path.join(path, 'pos_neg_data.mat'), aug_data)
