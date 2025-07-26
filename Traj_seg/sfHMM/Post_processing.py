# -*- coding: utf-8 -*-
# @Time    : 2024/4/16 10:18
# @Author  : Dongliang

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def change_point_detection(segN, thresholds):
    """
    Performs detection of change points in the trajectory.

    :param segN: Results after initial segmentation.
    :param thresholds: The minimum distance between two neighboring change points.
    :return: A list containing change points.
    """

    mN = segN.shape[0]
    fit_data = segN[:, 4]  # Assuming the 5th column of segN contains the data to fit.
    cp = []
    for i in range(mN - 2):
        if fit_data[i + 1] != fit_data[i]:
            cp.append(i + 1)

    # Further filtering
    ind = []
    for i in range(len(cp) - 1):
        if abs(cp[i + 1] - cp[i]) < thresholds:
            ind.append(i + 1)

    cp = [point for idx, point in enumerate(cp) if idx not in ind]

    return cp


def calculate_average_velocity(dt, trajectory):
    """
    Calculate the average speed of each segment.

    :param dt: Time interval between two consecutive points.
    :param trajectory: List of particle trajectories.
    :return: The average speed of a particle's motion.
    """
    vel = []
    for traj in trajectory:
        dif = np.sum(np.sqrt((traj[1:, 0] - traj[:-1, 0]) ** 2 + (traj[1:, 1] - traj[:-1, 1]) ** 2))
        vel.append(dif / (dt * (len(traj) - 1)))
    return vel


def SquareDist(x0, x1, y0, y1, z0=None, z1=None):
    """
    Computes the squared distance between the two points (x0,y0,z0) and (x1,y1,z1).
    """
    if z0 is None and z1 is None:
        z0, z1 = np.zeros_like(x0), np.zeros_like(x1)

    return (x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2


def MSD(x, y, z=None, frac: float = 1):
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


def Scalings(msds, dt, dim=3):
    """
    Fit mean squared displacement to a power-law function.

    :param msds: Mean squared displacements.
    :param dt: Time interval between adjacent frames.
    :param dim: Dimension of the trajectory. Default is 3.
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


if __name__ == '__main__':
    # Set the parameters for the analysis
    dt = 0.02
    thresholds = 50    # Sets the length of the shortest track segment.
    diff_cp = 20       # Distance threshold between neighboring switching points

    # Loading data
    root = r'D:\TrajSeg-Cls\Exp Demo\19 01072020_2 perfect\5900-6689'
    current_dir = root
    xyzap_path = os.path.join(root, 'xyzap.xlsx')

    if not os.path.exists(xyzap_path):
        print("Error: File not found.")
        exit()

    xyzap = pd.read_excel(xyzap_path).values
    xy = xyzap[:, 1:3]
    ap = xyzap[:, 4:6]
    dis = np.sqrt((xy[:, 0] - xy[0, 0]) ** 2 + (xy[:, 1] - xy[0, 1]) ** 2)

    # Importing sfHMM segmentation results
    seg_dir = os.path.join(root, '')
    segA_path = os.path.join(seg_dir, 'sfHMM_azi.txt')
    segP_path = os.path.join(seg_dir, 'sfHMM_po.txt')
    segD_path = os.path.join(seg_dir, 'sfHMM_dis.txt')

    if os.path.exists(segA_path):
        segA = np.loadtxt(segA_path, skiprows=1, delimiter=',')
        cpA = change_point_detection(segN=segA, thresholds=thresholds)
    else:
        cpA = []

    if os.path.exists(segP_path):
        segP = np.loadtxt(segP_path, skiprows=1, delimiter=',')
        cpP = change_point_detection(segN=segP, thresholds=thresholds)
    else:
        cpP = []

    if os.path.exists(segD_path):
        segD = np.loadtxt(segD_path, skiprows=1, delimiter=',')
        cpD = change_point_detection(segN=segD, thresholds=thresholds)
    else:
        cpD = []

    # Merge switching points
    cp = np.unique(np.sort(np.concatenate((cpA, cpP, cpD))))

    # Merge neighboring switching points
    final_cp = []
    i = 0
    while i < len(cp):
        j = i + 1
        while j < len(cp) and cp[j] - cp[i] <= diff_cp:
            j += 1
        final_cp.append(int(np.floor(np.mean(cp[i:j]))))
        i = j

    # Save the final change points
    final_cp_path = os.path.join(current_dir, 'final_cp.txt')
    with open(final_cp_path, 'w') as f:
        for cp_point in final_cp:
            f.write(str(cp_point) + '\n')
            print(cp_point)

    cp = np.array(final_cp)

    # # Aggregating the trajectories of individual segments
    # seg = []
    # seg.append(xy[:cp[0], :2])
    # for i in range(len(cp) - 1):
    #     seg.append(xy[cp[i]:cp[i + 1] + 1, :2])
    # seg.append(xy[cp[-1]:, :2])
    #
    # avg_velocity = calculate_average_velocity(dt=dt, trajectory=seg)
    # print("Average Velocity:", avg_velocity)
    #
    # # Save the Average Velocity
    # avg_vel_path = os.path.join(current_dir, 'avg_vel.txt')
    # with open(avg_vel_path, 'w') as f:
    #     for velocity in avg_velocity:
    #         f.write(str(velocity) + '\n')
    #
    # diff_vels = []
    # for i in seg:
    #     x = i[:, 0]
    #     y = i[:, 1]
    #     msds = MSD(x=x, y=y, frac=0.5)
    #     diff_vel = Scalings(msds=msds, dt=dt, dim=2)
    #     diff_vels.append(diff_vel)
    #     print("Diffusion Velocity and alpha :", diff_vel)
    #
    # # Save the Diffusion Velocity
    # diff_vel_path = os.path.join(current_dir, 'diff_vel.txt')
    # with open(diff_vel_path, 'w') as f:
    #     for diff_vel in diff_vels:
    #         f.write(str(diff_vel) + '\n')

    # Plot
    colors = ['r', 'g', 'b', 'k', 'y', 'm', 'c'] * 10

    plt.figure(figsize=(10, 8))

    # XY trajectory
    plt.subplot(2, 2, 1)
    plt.plot(xy[:cp[0], 0], xy[:cp[0], 1], 'b*')
    for i in range(len(cp) - 1):
        plt.plot(xy[cp[i]:cp[i + 1] + 1, 0], xy[cp[i]:cp[i + 1] + 1, 1], color=colors[i])
    plt.plot(xy[cp[-1]:, 0], xy[cp[-1]:, 1], color=colors[-1])
    plt.title('XY Trajectory')

    # Displacement
    plt.subplot(2, 2, 2)
    plt.plot(np.arange(cp[0]), dis[:cp[0]], 'b*')
    for i in range(len(cp) - 1):
        plt.plot(np.arange(cp[i], cp[i + 1] + 1), dis[cp[i]:cp[i + 1] + 1], color=colors[i])
    plt.plot(np.arange(cp[-1], len(dis)), dis[cp[-1]:], color=colors[-1])
    plt.title('Displacement')

    # Azimuth angle
    if len(ap) > 0:
        plt.subplot(2, 2, 3)
        plt.plot(np.arange(cp[0]), ap[:cp[0], 0], 'b*')
        for i in range(len(cp) - 1):
            plt.plot(np.arange(cp[i], cp[i + 1] + 1), ap[cp[i]:cp[i + 1] + 1, 0], color=colors[i])
        plt.plot(np.arange(cp[-1], len(ap)), ap[cp[-1]:, 0], color=colors[-1])
        plt.title('Azimuth Angle')

    # Polar angle
    if len(ap) > 0:
        plt.subplot(2, 2, 4)
        plt.plot(np.arange(cp[0]), ap[:cp[0], 1], 'b*')
        for i in range(len(cp) - 1):
            plt.plot(np.arange(cp[i], cp[i + 1] + 1), ap[cp[i]:cp[i + 1] + 1, 1], color=colors[i])
        plt.plot(np.arange(cp[-1], len(ap)), ap[cp[-1]:, 1], color=colors[-1])
        plt.title('Polar Angle')

        plt.tight_layout()
        plt.savefig(os.path.join(current_dir, 'seg.png'), dpi=600)
        plt.show()
