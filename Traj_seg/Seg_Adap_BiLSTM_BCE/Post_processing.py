# -*- coding: utf-8 -*-
# @Time    : 2024/7/7 18:34
# @Author  : Dongliang

import os
from datetime import datetime

import numpy as np
import pandas as pd
import scipy
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit


def merge_change_points(loc, threshold):
    """
    Merge change points by retaining points separated by at least threshold distance.

    :param loc: List of change point indices or positions.
    :param threshold: Minimum distance threshold for retaining points.
    :return: Filtered list of change points.
    """

    N = len(loc)
    filtered_points = [loc[0]]  # Retain the first point
    last_kept = loc[0]

    for i in range(1, N):
        if loc[i] - last_kept >= threshold:
            filtered_points.append(loc[i])
            last_kept = loc[i]  # Update the last retained point

    return filtered_points


def merge_close_numbers(arr, threshold=20):
    """
    Merge numbers in the array that are within the threshold.

    :param arr: List of switching point locations.
    :param threshold: The maximum distance between two switching points to be merged.
    :return: A list of merged switching points.
    """
    arr = np.sort(arr)
    # merged = []
    # i = 0
    # while i < len(arr):
    #     merged.append(arr[i])
    #     while i + 1 < len(arr) and abs(arr[i + 1] - arr[i]) < threshold:
    #         i += 1
    #     i += 1
    # return merged

    result = []
    group = [arr[0]]

    for i in range(1, len(arr)):
        if arr[i] - arr[i - 1] <= threshold:
            group.append(arr[i])
        else:
            result.append(round(sum(group) / len(group)))
            group = [arr[i]]

    if group:
        result.append(round(sum(group) / len(group)))

    return result


def cal_net_velocity(dt, trajectory):
    """
    Calculate displacement speed (net speed): distance between start and end points / total time.

    :param dt: Time interval per step.
    :param trajectory: Trajectory with shape (N, D) (2D or 3D).
    :return: Displacement velocity.
    """

    diff_ = trajectory[-1] - trajectory[0]
    net_distance = np.linalg.norm(diff_)
    total_time = dt * (trajectory.shape[0] - 1)
    return net_distance / total_time


def cal_average_speed(dt, trajectory):
    """
    Calculate average speed: total distance traveled / total time.

    :param dt: Time interval per step.
    :param trajectory: Trajectory with shape (N, D) (2D or 3D).
    :return: Average speed.
    """

    diff_ = np.diff(trajectory, axis=0)
    step_lengths = np.linalg.norm(diff_, axis=1)
    total_distance = np.sum(step_lengths)
    total_time = dt * (trajectory.shape[0] - 1)
    return total_distance, total_distance / total_time


def batch_velocity_analysis(dt, trajectory):
    """
    Perform batch velocity analysis on multiple trajectories, calculating average speed and net velocity.

    :param dt: Time interval per step.
    :param trajectory: Collection of trajectories, each with shape (N, D) (2D or 3D).
    :return: tuple: Two numpy arrays containing:
            - avg_speeds (np.ndarray): Average speeds for each trajectory.
            - net_velocity (np.ndarray): Net (displacement) velocity for each trajectory.
    """

    avg_speeds = []
    total_distances = []
    net_velocity = []
    for traj in trajectory:
        if traj.shape[0] < 2:
            avg_speeds.append(np.nan)
            net_velocity.append(np.nan)
            continue
        step_size, avg_speed = cal_average_speed(dt, traj)
        total_distances.append(step_size)
        avg_speeds.append(avg_speed)
        net_velocity.append(cal_net_velocity(dt, traj))
    return np.array(total_distances), np.array(avg_speeds), np.array(net_velocity)


def SquareDist(x0, x1, y0, y1, z0=None, z1=None):
    """
    Computes the squared distance between the two points (x0,y0,z0) and (x1,y1,z1).
    """
    if z0 is None and z1 is None:
        z0, z1 = np.zeros_like(x0), np.zeros_like(x1)

    return (x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2


def MSD(x, y, z=None, frac=0.2):
    """
    Computes the mean squared displacement for a trajectory (x,y) up to
    frac*len(x) of the trajectory.

    :param x: The x-coordinate of the trajectory.
    :param y: The y-coordinate of the trajectory.
    :param z: The z-coordinate of the trajectory.
    :param frac: float in [0,1]. Fraction of trajectory duration to compute msd.

    """
    if z is None:
        z = np.zeros_like(x)

    N = int(len(x) * frac)
    msd = []

    for lag in range(1, N):
        msd.append(
            np.mean(
                [
                    SquareDist(x[j], x[j + lag], y[j], y[j + lag], z[j], z[j + lag])
                    for j in range(len(x) - lag)
                ]
            )
        )
    return np.array(msd)


def Scalings(msds, dt, dim):
    """
    Fit mean squared displacement to a power-law function.
    :param msds: Mean squared displacements.
    :param dt: Time interval between adjacent frames.
    :param dim: Dimension of the trajectory.
    :return: tuple of length 3
        The first index is the fitted generalized diffusion constant,
        the second is the scaling exponent alpha, and the final is the p-value for the fit.

    """

    def power(x, D, alpha, offset=0):
        return 2 * dim * D * (x) ** alpha + offset

    # Perform the curve fitting
    t = np.arange(1, len(msds) + 1) * dt

    popt, pcov = curve_fit(power, t, msds,
                           p0=[msds[0] / (2 * dim * dt), 1],
                           bounds=[[0.0000001, 0.], [np.inf, 10]],
                           method='trf')
    r = msds - power(np.arange(1, len(msds) + 1) * dt, *popt)

    popt, pcov = curve_fit(power, t, msds,
                           sigma=np.repeat(np.std(r, ddof=1), len(msds)),
                           p0=[msds[0] / (2 * dim * dt), 1],
                           bounds=[[0.0000001, 0.], [np.inf, 10]],
                           method='trf')
    return popt[0], popt[1]


def calculate_column_widths(data):
    """
    Dynamically calculates the maximum width of each column string in the list.

    :param data: List of strings to be written.
    """

    column_widths = [max(len(str(cell)) for cell in column) for column in zip(*data)]
    return column_widths


def write_table_to_txt(filename, data):
    """
    Format each column left-justified when writing to the file.

    :param filename: Name of the file to be saved.
    :param data: List of strings to be written.
    """
    column_widths = calculate_column_widths(data)
    with open(filename, "w", encoding="utf-8") as file:
        for row in data:
            formatted_row = "  ".join(f"{cell:<{column_width}}" for cell, column_width in zip(row, column_widths))
            file.write(formatted_row + "\n")


def plot_trace(cp, xy, dis, ap, savepath):
    """
    Plots a trajectory with change points.

    :param cp: A list of switching point for the trajectory.
    :param xy: The xy coordinates of the trajectory.
    :param dis: The displacement of the trajectory.
    :param ap: The azimuth and polar angles of the particles.
    :param savepath: The path to save the results.
    """

    # Aggregating the trajectories of individual segments
    seg = []
    seg.append(xy[:cp[0]])
    for i in range(len(cp) - 1):
        seg.append(xy[cp[i]:cp[i + 1] + 1])
    seg.append(xy[cp[-1]:])

    # Plot
    # XY trajectory
    # colors = plt.cm.get_cmap('viridis', len(cp) + 1)
    colors = ['r', 'g', 'b', 'y', 'c', 'm'] * (len(cp) + 1)

    cp = np.concatenate(([0], cp, [len(xy) - 1]))
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 2, 1)
    for i in range(len(cp) - 1):
        plt.plot(xy[cp[i]:cp[i + 1] + 1, 0], xy[cp[i]:cp[i + 1] + 1, 1], color=colors[i])
        # plt.plot(xy[cp[i], 0], xy[cp[i], 1], 'r*')
    plt.title('XY Trajectory')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.gca().set_aspect('equal', adjustable='box')

    # Displacement
    plt.subplot(2, 2, 2)
    for i in range(len(cp) - 1):
        plt.plot(np.arange(cp[i], cp[i + 1] + 1), dis[cp[i]:cp[i + 1] + 1], color=colors[i])
        # plt.plot(cp[i], dis[cp[i]], 'r*')
    plt.title('Displacement')
    plt.xlabel('Frames')
    plt.ylabel('Displacement')

    # Calculate and plot azimuth and polar angles
    if ap is not None:
        # Calculate step azimuth and polar angles
        mean_step_azi, mean_step_po, std_step_azi, std_step_po = [], [], [], []
        for i in range(len(cp) - 1):
            step_po = np.abs(np.diff(ap[cp[i]:cp[i + 1] + 1, 0]))
            step_azi = np.abs(np.diff(ap[cp[i]:cp[i + 1] + 1, 1]))
            mean_step_azi.append(np.mean(step_azi))
            mean_step_po.append(np.mean(step_po))
            std_step_azi.append(np.std(step_azi))
            std_step_po.append(np.std(step_po))

        # Save the step angle
        step_angle_path = os.path.join(savepath, 'step_angle.txt')
        data = [['id', 'mean_step_azi', 'std_step_azi', 'mean_step_po', 'std_step_po']]
        k = 0
        for i, j, m, n in zip(mean_step_azi, std_step_azi, mean_step_po, std_step_po):
            data.append([f'segment{k}: ', f'{i:.2f}', f'{j:.2f}', f'{m:.2f}', f'{n:.2f}'])
            k += 1
        write_table_to_txt(step_angle_path, data=data)

        # Plot polar angle
        plt.subplot(2, 2, 3)
        for i in range(len(cp) - 1):
            plt.plot(np.arange(cp[i], cp[i + 1] + 1), ap[cp[i]:cp[i + 1] + 1, 0], color=colors[i])
        plt.title('Polar Angle')
        plt.xlabel('Frames')
        plt.ylabel('Polar angle')

        # Plot azimuth angle
        plt.subplot(2, 2, 4)
        for i in range(len(cp) - 1):
            plt.plot(np.arange(cp[i], cp[i + 1] + 1), ap[cp[i]:cp[i + 1] + 1, 1], color=colors[i])
        plt.title('Azimuth Angle')
        plt.xlabel('Frames')
        plt.ylabel('Azimuth angle')

    plt.tight_layout()
    plt.savefig(os.path.join(savepath, 'Adap_seg.png'), dpi=600)
    plt.show()


def calculate_features(dt, cp, xy, savepath):
    """
    Calculate average velocity, diffusion coefficient and alpha for each segment.

    :param dt: The time interval between adjacent frames.
    :param cp: A list of switching point for the trajectory.
    :param xy: The xy coordinates of the trajectory.
    :param savepath: The path to save the results.
    """
    # Aggregating the trajectories of individual segments
    seg = []
    seg.append(xy[:cp[0]])
    for i in range(len(cp) - 1):
        seg.append(xy[cp[i]:cp[i + 1]])
    seg.append(xy[cp[-1]:])

    total_distances, avg_speeds, net_velocities = batch_velocity_analysis(dt=dt, trajectory=seg)
    print(f'total_distances : {total_distances}')
    print(f'Average Velocity : {avg_speeds}\n')
    print(f'Net speed : {net_velocities}\n')

    # Save the average speed and net velocity
    with open(os.path.join(savepath, 'total_distance.txt'), 'w') as f:
        for i, total_distance in enumerate(total_distances):
            f.write(f'segment {i} : {total_distance}\n')

    with open(os.path.join(savepath, 'avg_vel.txt'), 'w') as f:
        for i, avg_speed in enumerate(avg_speeds):
            f.write(f'segment {i} : {avg_speed}\n')

    with open(os.path.join(savepath, 'net_vel.txt'), 'w') as f:
        for i, net_velocity in enumerate(net_velocities):
            f.write(f'segment {i} : {net_velocity}\n')

    Diffusion_coeff, alphas = [], []
    for i in seg:
        x, y = i[:, 0], i[:, 1]
        if i.shape[1] == 2:
            z = None
        else:
            z = i[:, 2]

        msds = MSD(x=x, y=y, z=z, frac=0.2)
        D, alpha = Scalings(msds=msds, dt=dt, dim=i.shape[1])
        Diffusion_coeff.append(D)
        alphas.append(alpha)
        print(f'Diffusion Velocity and alpha : {D}, {alpha}')

    # Save the Diffusion Velocity
    diff_vel_path = os.path.join(savepath, 'diff_vel.txt')
    with open(diff_vel_path, 'w') as f:
        for i, (D, alpha) in enumerate(zip(Diffusion_coeff, alphas)):
            f.write(f'segment {i} : {D}, {alpha}\n')


if __name__ == '__main__':
    # Example Usage

    dt = 0.001
    threshold = 20

    # Load data
    path = r'C:\Users\songn\Desktop'
    savepath = path
    if not os.path.exists(savepath):
        os.mkdir(savepath)

    # Load change points
    arr = scipy.io.loadmat(os.path.join(savepath, 'Adap_CPs.mat'))['CPs']
    cp = arr.squeeze()
    cp = merge_close_numbers(cp, threshold=threshold)
    print(f'Merged array : {cp}')
    scipy.io.savemat(os.path.join(savepath, 'Merged_CPs.mat'), {'merged_cp': cp})

    # Save merged change points
    # Get current date and time
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with open(os.path.join(savepath, 'merged_CP.txt'), 'a') as f:
        f.write(f'========== {current_time} =========\n')
        f.write(f'Merged CPs : {cp}\n')

    test = pd.read_csv(os.path.join(path, '210414 TR0006 Fig3E_G TrajSeg 001.csv')).values
    # xyzap = test[:,1:6]
    # xy = xyzap[:,:-2]
    # ap = xyzap[:, -2:]
    xy = test[:, :3]
    ap = None

    if xy.shape[1] == 2:
        dis = np.sqrt((xy[:, 0] - xy[0, 0]) ** 2 + (xy[:, 1] - xy[0, 1]) ** 2)
    elif xy.shape[1] == 3:
        dis = np.sqrt((xy[:, 0] - xy[0, 0]) ** 2 + (xy[:, 1] - xy[0, 1]) ** 2 + (xy[:, 2] - xy[0, 2]) ** 2)

    plot_trace(cp, xy, dis, ap=ap, savepath=savepath)
    calculate_features(dt=dt, cp=cp, xy=xy, savepath=savepath)

    params = {'threshold': threshold}
    file_name = os.path.join(savepath, f'Predict_params.txt')
    name = os.path.split(path)[-1]
    with open(file_name, 'a') as f:
        f.write(f'========== {current_time} ==========\n')
        f.write(f'{name}  Predict parameters:\n')
        for key, value in params.items():
            f.write(f'{key} = {value}\n')
