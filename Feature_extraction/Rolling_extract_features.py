# -*- coding: utf-8 -*-
# @Time    : 2024/12/27 16:40
# @Author  : Dongliang

from AAISPT.Feature_extraction.Fingerprint_feat_gen import *


def Rolling_extract_feature(traj, dt, step_angle=False, window_ratio=None):
    """
    Extraction of multidimensional features in trajectories using the rolling window method.

    :param traj: Multidimensional trajectory dataset.
    :param dt: The time interval between neighboring points.
    :param window_ratio: Ratio of sliding window size to track length.
    :param step_angle: Whether the input data is a step angle.
    :return: List of features extracted from the trajectory dataset.
    """

    length, dim = traj.shape
    if dim == 2 or dim == 3:
        num_feature = 32
    elif dim == 4 or dim == 5:
        num_feature = 46

    if window_ratio is None:
        window_size = 35
        step = 7
    else:
        window_size = max(int(length * window_ratio), 30)
        step = int(0.1 * window_size)

    half_window = window_size // 2
    timeseries = np.full((length, length, num_feature), -10, dtype=np.float32)

    for center in range(0, length, step):
        min_value = np.max([0, center - half_window])
        max_value = np.min([length, center + half_window + 1])
        corr_min = center + half_window + 1 - max_value
        corr_max = min_value + half_window - center

        start = np.max([0, min_value - corr_min])
        ends = np.min([max_value + corr_max, length])
        window = traj[start:ends, :]

        # Extract coordinates based on dimensionality
        x, y = window[:, 0], window[:, 1]
        if dim == 2:
            z, po, azi = None, None, None
        elif dim == 3:
            z = window[:, 2]
            po, azi = None, None
        elif dim == 4:
            z = None
            po, azi = window[:, 2], window[:, 3]
        elif dim == 5 or dim == 6:
            z, po, azi = window[:, 2], window[:, 3], window[:, 4]

        FP_segment = GetFeatures(x=x, y=y, z=z, dt=dt, azimuth=azi, polar=po, step_angle=step_angle)

        FP_segment[np.isnan(FP_segment)] = 0
        FP_segment = np.expand_dims(FP_segment, axis=[0, 1])
        FP_segment_repeat = np.repeat(FP_segment, len(window), axis=1)
        timeseries[center, start:ends, :] = FP_segment_repeat
        del FP_segment, FP_segment_repeat, window

    # timeseries_clean = np.zeros((1, length, num_feature))
    # for col in range(len(timeseries)):
    #     b = timeseries[:, col, :][timeseries[:, col, :] != -10].reshape(-1, num_feature)
    #     timeseries_clean[0, col, :] = np.mean(b, axis=0)

    timeseries[timeseries == -10] = np.nan
    timeseries_clean = np.nanmean(timeseries, axis=0, keepdims=True)

    return timeseries_clean


def process_trajectory(idx, data, dt, window_ratio, step_angle, savepath):
    """
    Extracting multiple features from trajectories using the rolling window method.

    :param idx:
    :param data:
    :param dt: The time interval between neighboring points.
    :param window_ratio: Ratio of sliding window size to track length.
    :param step_angle: Whether the input data is a step angle.
    :param savepath: The path to save the extracted features.
    """
    try:
        feature = Rolling_extract_feature(data, dt, window_ratio, step_angle)
        np.save(os.path.join(savepath, f'{idx:06}.npy'), feature)
        return feature
    except Exception as e:
        with open(os.path.join(savepath, 'error_log.txt'), 'a') as f:
            f.write(f'Error in: {idx}\n')
        return None


if __name__ == '__main__':
    step_angle = False
    window_ratio = 0.1
    dt = 0.02

    # path = r'D:\TrajSeg-Cls\TrajSEG-CLS_V3\CLS\Var_LD_500\SNR05_4-5D'
    # for mode in ['test']:
    #     data = scipy.io.loadmat(os.path.join(path, f'varLD_{mode}.mat'))
    #     savepath = os.path.join(path, mode)
    #     if not os.path.exists(savepath):
    #         os.mkdir(savepath)
    #
    #     X, Y = data['data'].squeeze(), data['label'].squeeze()
    #     # features = Extract_features(savepath=savepath, traj=X, dt=dt, window_ratio=window_ratio, step_angle=step_angle)
    #
    #     nums = X.shape[0]
    #     features = Parallel(n_jobs=-1)(
    #         delayed(process_trajectory)(i, X[i], dt, window_ratio, step_angle) for i in range(nums))
    #     # TODO: Remember to modify the step_angle parameter according to the actual data.
    #
    #     # Save features
    #     scipy.io.savemat(os.path.join(savepath, f'Roll_{mode}_feature.mat'), {'data': features, 'label': Y})

    paths = [
        r'E:\20240707\新建文件夹\20250104\Endocytosis_NEW',
        r'E:\20240707\新建文件夹\20250104\QiPan_NEW',
        r'E:\20240707\新建文件夹\20250104\YanYu_NEW'
    ]

    for path in paths:
        data = scipy.io.loadmat(os.path.join(path, 'aug_data_resample.mat'))
        savepath = path

        X, Y = data['data'].squeeze(), data['label'].squeeze()

        nums = X.shape[0]
        features = Parallel(n_jobs=-1)(
            delayed(process_trajectory)(i, X[i], dt, window_ratio, step_angle, savepath) for i in range(nums))
        # TODO: Remember to modify the step_angle parameter according to the actual data.

        # Save features
        scipy.io.savemat(os.path.join(savepath, f'Roll_feature.mat'), {'data': features, 'label': Y})
