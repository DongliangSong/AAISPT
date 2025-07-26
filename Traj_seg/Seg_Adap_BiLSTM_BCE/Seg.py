# -*- coding: utf-8 -*-
# @Time    : 2025/5/26 15:39
# @Author  : Dongliang


import os
from datetime import datetime

import numpy as np
import pandas as pd
import scipy
import torch
from scipy.signal import find_peaks

from AAISPT.Traj_seg.Seg_Adap_BiLSTM_BCE.Post_processing import merge_change_points, calculate_features, plot_trace
from AAISPT.Traj_seg.Seg_Adap_BiLSTM_BCE.main import Seg_Adap_BiLSTM_BCE
from AAISPT.Traj_seg.Seg_Adap_BiLSTM_BCE.Data_Preprocessing import preprocess_tracks
from AAISPT.read_files import load_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def direct_segmentation(model, traj, prominence=0.0001):
    """
    Performs segmentation on a trajectory using a given model to identify change points.

    :param model: Seg_BiLSTM_BCE model used for predicting probabilities from the input trajectory.
    :param traj: Input trajectory data.
    :param prominence: Minimum prominence for peak detection. Defaults to 0.0001.
    :return: A tuple containing:
            - loc: Indices of detected peaks in the trajectory.
            - prob: Probability array output by the model.
    """
    model.to(device)
    torch.no_grad()
    traj = torch.tensor(traj, dtype=torch.float32).unsqueeze(0).to(device)
    prob = model(traj)
    prob = prob.squeeze().detach().cpu().numpy()

    # Define the region of interest (ROI) by excluding 10 points from start and end
    length = len(prob)
    x = np.arange(length)
    x_min, x_max = 10, length - 10
    mask = (x >= x_min) & (x <= x_max)
    y_roi = prob[mask]

    # Find peaks
    peak, _ = find_peaks(y_roi, prominence=prominence)
    loc = peak + 10
    return loc, prob


def sliding_window_segmentation(model, traj, window_size, step_size, prominence=0.0001):
    """
    Arbitrary length trajectory segmentation by sliding window method.

    :param model: Seg_BiLSTM_BCE model used for predicting probabilities from the input trajectory.
    :param traj: Input trajectory data.
    :param window_size: Length of sliding window.
    :param step_size: Step length of sliding window.
    :param prominence: Minimum prominence for peak detection. Defaults to 0.0001.
    :return: A tuple containing:
            - seg_points: Indices of detected peaks in the trajectory.
            - probs: Probability array output by the model.
    """
    n = len(traj)
    seg_points = []
    model.to(device)
    torch.no_grad()
    probs = []
    for i in range(0, n - window_size + 1, step_size):
        win = traj[i:i + window_size, :]
        window = torch.tensor(win, dtype=torch.float32).unsqueeze(0).to(device)
        prob = model(window)
        prob = prob.squeeze().detach().cpu().numpy()
        probs.append(prob)

        mask = (np.arange(len(prob)) >= 10) & (np.arange(len(prob)) < window_size - 10)
        peak, _ = find_peaks(prob[mask], prominence=prominence)
        loc = peak + 10
        # seg1 = win[:loc, :]
        # seg2 = win[loc:, :]
        #
        # # Determine if there is a significant difference between seg1 and seg2.
        # # Perform the Wilcoxon test
        # statistic, p_value = ranksums(seg1, seg2)
        # print(f"statistic: {statistic}")
        # print(f"P-value: {p_value}")
        #
        # if np.sum(p_value < conf_level) >= 1:
        #     print("There is a significant difference")
        #     seg_points.append(i + loc)
        # else:
        #     print("No significant difference")

        seg_points.extend(i + loc)
    return seg_points, probs


if __name__ == '__main__':
    data_process = 'Normalization'
    mode = 'Directed segmention'
    # mode = 'Rolling window segmentation'

    dt = 0.001  # Time interval between adjacent frames
    # Use sliding window for segmentation? Set scrolling parameters. Otherwise, skip.
    window_size = 80
    step_size = 20

    # Post_processing parameters
    prominence = 0.0001
    threshold = 3000  # Length threshold for merging change points.

    # Set model hyperparameters (Fixed)
    hidden_size = 128
    num_layers = 3
    max_dim = 5
    d_model = 128
    out_dim = 1

    # Load well-trained model
    model_path = r'../model'
    model = Seg_Adap_BiLSTM_BCE(max_dim=max_dim, d_model=d_model, hidden_size=hidden_size, num_layers=num_layers,
                                out_dim=out_dim)

    model.load_state_dict(torch.load(os.path.join(model_path, 'seg_model.pth')))
    torch.no_grad()

    # Load data
    data_path = r'..\data\Exp_data_for_example'
    savepath = os.path.join(data_path, 'result')
    if not os.path.exists(savepath):
        os.mkdir(savepath)

    test = load_data(os.path.join(data_path, '210604 TR0111 Fig3A_D TrajSeg 001.csv'))
    # xyz = test[:, :-2]
    # pa = test[:, -2:]
    # pa[:, [0, 1]] = pa[:, [1, 0]]
    # new = np.concatenate((np.diff(xyz, axis=0), np.abs(np.diff(pa, axis=0))), axis=1)
    # new = np.expand_dims(new, axis=0)
    # data = preprocess_tracks(new, use_xyz_diff=[])

    xyz = test[:,:3]
    new = np.diff(xyz, axis=0)
    new = np.expand_dims(new, axis=0)
    data = preprocess_tracks(new, use_xyz_diff=[])
    data = data.squeeze()

    # Sliding window for trajectory segmentation.
    if mode == 'Directed segmention':
        loc, prob = direct_segmentation(model, data, prominence=prominence)
    else:
        loc, prob = sliding_window_segmentation(model, data, window_size, step_size, prominence=prominence)

    loc.sort()
    print('loc: ', loc)
    CPs = merge_change_points(loc, threshold=threshold)  # threshold: Length threshold for merging change points.
    print('Change points: ', CPs)
    scipy.io.savemat(os.path.join(savepath, 'Adap_CPs.mat'), {'CPs': CPs, 'prob': prob})

    # Visualization of results
    if xyz.shape[-1] == 2:
        dis = np.sqrt((xyz[:, 0] - xyz[0, 0]) ** 2 + (xyz[:, 1] - xyz[0, 1]) ** 2)
    else:
        dis = np.sqrt((xyz[:, 0] - xyz[0, 0]) ** 2 + (xyz[:, 1] - xyz[0, 1]) ** 2 + (xyz[:, 2] - xyz[0, 2]) ** 2)

    if 'ap' in globals():
        print("ap exists")
    else:
        ap = None
        print("my_var was not defined. Set to None.")

    plot_trace(CPs, xyz, dis, ap=ap, savepath=savepath)
    calculate_features(dt=dt, cp=CPs, xy=xyz, savepath=savepath)

    # Save results
    params = {
        'window_size': window_size,
        'step_size': step_size,
    }
    # Get current date and time
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    file_name = os.path.join(savepath, 'Predict_params.txt')
    name = os.path.split(model_path)[-1]
    with open(file_name, 'a') as f:
        f.write('==========' + f'{current_time}' + '=========\n')
        f.write('Model path: ' + f'{model_path}\n')
        f.write('Data preprocess: ' + f'{data_process}\n')
        f.write(name + 'Predict parameters:\n')
        for key, value in params.items():
            f.write(f"{key} = {value}\n")

    # # Load test dataset
    # data_path = r'D:\TrajSeg-Cls\endoysis\16_01092020_2 good'
    # savepath = data_path
    #
    # if 'endoysis' in data_path or 'Exp Demo' in data_path:
    #     test = pd.read_excel(os.path.join(data_path, 'xyap.xlsx')).values
    #     xyzap = test[:, 2:]
    #
    #     # Data preprocessing
    #     xyz = np.diff(xyzap[:, :-2], axis=0)
    #     ap = np.abs(np.diff(xyzap[:, -2:], axis=0))
    #     new = np.concatenate((xyz, ap), axis=1)
    #
    # elif 'QiPan' in data_path:
    #     test = pd.read_excel(os.path.join(data_path, 'S1-S2_Tracks_160.xlsx')).values
    #     xy = test[:, 1:]
    #     new = np.diff(xy, axis=0)
    #
    # elif 'YanYu' in data_path or 'Janus' in data_path:
    #     file = os.path.join(data_path, '590-101517_S1_BW1_T10_10_Single_track.xlsx')
    #     test = pd.read_excel(file).values
    #
    #     if 'Single_track' in file:
    #         xy = test[:, 1:]
    #         new = np.diff(xy, axis=0)
    #     else:
    #         if (test[:, 6] == 90).all() or (test[:, 8] == 90).all():
    #             xy = test[:, 1:3]
    #             new = np.diff(xy, axis=0)
    #         else:
    #             xy = test[:, 1:3]
    #             ap = test[:, 7:9]
    #             xy = np.diff(xy, axis=0)
    #             ap = np.abs(np.diff(ap, axis=0))
    #             new = np.concatenate((xy, ap), axis=1)
    #
    # new = np.expand_dims(new, axis=0)
    # data = preprocess_tracks(new, use_xyz_diff=[])
    # data = data.squeeze()
