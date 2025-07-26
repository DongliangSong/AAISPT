# -*- coding: utf-8 -*-
# @Time    : 2024/3/1 15:32
# @Author  : Dongliang

"""
    Library for generation of diffusional features
"""

import os
from itertools import combinations

import numpy as np
import scipy
from joblib import Parallel, delayed
from numba import njit
from scipy.optimize import curve_fit
from scipy.spatial import ConvexHull
from scipy.special import gamma
from scipy.stats import kurtosis, kstest, normaltest, skew


@njit
def SquareDist(x0, x1, y0, y1, z0, z1):
    """
    Computes the squared distance between the two points (x0,y0,z0) and (x1,y1,z1).
    """
    return (x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2


@njit
def QuadDist(x0, x1, y0, y1, z0=None, z1=None):
    """
    Computes the four-norm (x1-x0)**4+(y1-y0)**4+(z1-z0)**4.
    """
    return (x1 - x0) ** 4 + (y1 - y0) ** 4 + (z1 - z0) ** 4


@njit
def Displacement(x, y, z):
    """
    Calculate the displacement of the trajectory.

    :param x: The x-coordinate of the trajectory.
    :param y: The y-coordinate of the trajectory.
    :param z: The z-coordinate of the trajectory.
    """
    return np.sqrt((x - x[0]) ** 2 + (y - y[0]) ** 2 + (z - z[0]) ** 2)


def GetMax(x, y, z=None):
    """
    Computes the maximum squared distance between all points in the (x, y, z) set.

    :param x: The x-coordinate of the trajectory.
    :param y: The y-coordinate of the trajectory.
    :param z: The z-coordinate of the trajectory.
    """

    A = np.array([x, y, z]).T

    def square_distance(x, y):
        return sum([(xi - yi) ** 2 for xi, yi in zip(x, y)])

    max_square_distance = 0
    for pair in combinations(A, 2):
        if square_distance(*pair) > max_square_distance:
            max_square_distance = square_distance(*pair)
    return max_square_distance


def MSD(x, y, z, frac: float = 1):
    """
    Computes the mean squared displacement for a trajectory (x,y) up to
    frac*len(x) of the trajectory.

    :param x: The x-coordinate of the trajectory.
    :param y: The y-coordinate of the trajectory.
    :param z: The z-coordinate of the trajectory.
    :param frac: float in [0,1]. Fraction of trajectory duration to compute msd up to.

    """
    N = int(len(x) * frac) if len(x) > 20 else len(x)
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


def calculate_covariance(x, y, z=None):
    """
    Calculates the covariance matrix for the given data.

    :param x: The x-coordinate of the trajectory.
    :param y: The y-coordinate of the trajectory.
    :param z: The z-coordinate of the trajectory.
    """
    if z is None:
        return np.cov(x, y)
    else:
        return np.array([[np.cov(x, x)[0, 1], np.cov(x, y)[0, 1], np.cov(x, z)[0, 1]],
                         [np.cov(y, x)[0, 1], np.cov(y, y)[0, 1], np.cov(y, z)[0, 1]],
                         [np.cov(z, x)[0, 1], np.cov(z, y)[0, 1], np.cov(z, z)[0, 1]]])


def Asymmetry(gyration_tensor):
    """
    Calculates the asymmetry of a trajectory. The asymmetry can be used to detect directed motion.

    :param gyration_tensor: Covariance matrix for 2D or 3D trajectory data.
    """

    eigenvalues, eigenvectors = np.linalg.eig(gyration_tensor)
    lambda1 = eigenvalues[0]
    lambda2 = eigenvalues[1]
    a = -1 * np.log(1 - (lambda1 - lambda2) ** 2 / (2 * (lambda1 + lambda2) ** 2))
    return a


@njit
def Efficiency(x, y, z):
    """
    The efficiency of a trajectory is a measure of the linearity of the trajectory.
    It is defined as the logarithm of the ratio of the squared end-to-end distance
    to the sum of the squared distances.

    :param x: The x-coordinate of the trajectory.
    :param y: The y-coordinate of the trajectory.
    :param z: The z-coordinate of the trajectory.

    """
    N = len(x)
    top = SquareDist(x[0], x[-1], y[0], y[-1], z[0], z[-1])
    bottom = sum(
        [SquareDist(x[i], x[i + 1], y[i], y[i + 1], z[i], z[i + 1]) for i in range(0, N - 1)])
    return np.log(np.sqrt(top / ((N - 1) * bottom)))


def Stretched_exponential(t, A, tau, beta, C):
    """
    Define a stretched exponential function.
    """
    return A * np.exp(-(t / tau) ** beta) + C


def Relaxation_time(data):
    """
    Fit the stretched exponential function to the auto-correlation curve.
    """
    # Compute the auto-correlation function of a time series.
    n = len(data)
    mean = np.mean(data)
    centered_data = data - mean
    autocorr = np.correlate(centered_data, centered_data, mode='full') / (np.std(data) ** 2 * n)
    acf = autocorr[n - 1:]

    # Fitting
    x = np.linspace(1, len(acf), len(acf))
    popt, pcov = curve_fit(Stretched_exponential, x, acf, p0=(1, 10, 1.5, 0), maxfev=100000)
    A, tau, beta, C = popt

    return tau / beta * gamma(1 / beta)


@njit
def FractalDim(x, y, z, max_square_distance):
    """
    Fractal dimension is a measure of the space-filling capacity of a trajectory.

    Computes the fractal dimension using the estimator suggested by Katz & George
    in Fractals and the analysis of growth paths, 1985.

    :param x: The x-coordinate of the trajectory.
    :param y: The y-coordinate of the trajectory.
    :param z: The z-coordinate of the trajectory.
    :param max_square_distance: Maximum squared pair-wise distance for the point in the trajectory.
    """

    N = len(x)
    upper = np.log(N)

    L = sum(
        [
            np.sqrt(SquareDist(x[i], x[i + 1], y[i], y[i + 1], z[i], z[i + 1]))
            for i in range(0, N - 1)
        ]
    )

    d = np.sqrt(max_square_distance)
    lower = np.log(N * d * L ** (-1))
    return upper / lower


@njit
def Step_length(x, y, z):
    """
    Calculate the step length of a trajectory.

    :param x: The x-coordinate of the trajectory.
    :param y: The y-coordinate of the trajectory.
    :param z: The z-coordinate of the trajectory.
    """

    N = len(x)
    SL = np.sqrt((x[1:] - x[:N - 1]) ** 2 + (y[1:] - y[:N - 1]) ** 2 + (z[1:] - z[:N - 1]) ** 2)
    return SL


@njit
def Max_excursion(x, y, z, SL=None):
    """
    The maximal excursion of the particle, normalised to its total displacement (range of movement).
    It should detect relatively long jumps (in comparison to the overall displacement).

    :param x: The x-coordinate of the trajectory.
    :param y: The y-coordinate of the trajectory.
    :param z: The z-coordinate of the trajectory.
    :param SL: The step length of the trajectory.
    """

    if SL is None:
        SL = Step_length(x, y, z)

    N = len(x)
    max_dis = max(SL)
    total_dis = np.sqrt((x[N - 1] - x[0]) ** 2 + (y[N - 1] - y[0]) ** 2 + (z[N - 1] - z[0]) ** 2)
    return max_dis / total_dis


def Gaussianity(x, y, z, msds=None):
    """
    Calculate the Gaussianity of a trajectory.

    :param x: The x-coordinate of the trajectory.
    :param y: The y-coordinate of the trajectory.
    :param z: The z-coordinate of the trajectory.
    :param msds: Mean squared displacements for the trajectory.
    """

    gn = []
    for lag in range(1, len(msds)):
        r4 = np.mean(
            [QuadDist(x[j], x[j + lag], y[j], y[j + lag], z[j], z[j + lag]) for j in range(len(x) - lag)]
        )
        gn.append(r4 / (2 * msds[lag] ** 2))
        # gn.append(2 * r4 / (3 * msds[lag] ** 2))
    return np.mean(gn)


def Kurtosis_xyz(x, y, z=None, gyration_tensor=None):
    """
    Calculate the kurtosis of a trajectory.
    Kurtosis measures the asymmetry and peakedness of the distribution of points within a trajectory

    :param x: The x-coordinate of the trajectory.
    :param y: The y-coordinate of the trajectory.
    :param z: The z-coordinate of the trajectory.
    :param gyration_tensor: The gyration tensor of the trajectory.
    """

    # val, vec = np.linalg.eig(np.cov(x, y))
    # dominant = vec[:, np.argsort(val)][:, -1]
    # return kurtosis([np.dot(dominant, v) for v in np.array([x, y]).T], fisher=False)

    # eigenvalues, eigenvectors = LA.eig(get_tensor(x, y))
    #
    # N = len(x)
    # index = np.where(eigenvalues == max(eigenvalues))[0][0]
    # dominant_eigenvector = eigenvectors[index]
    # a_prod_b = np.array([sum(np.array([x[i], y[i]]) * dominant_eigenvector) for i in range(N)])
    # K = 1 / N * sum((a_prod_b - np.mean(a_prod_b)) ** 4 / np.std(a_prod_b) ** 4) - 3
    # return K
    if gyration_tensor is None:
        gyration_tensor = calculate_covariance(x, y, z)

    from scipy.stats import kurtosis

    if z is not None:
        val, vec = np.linalg.eig(gyration_tensor)
        dominant = vec[:, np.argsort(val)][:, -1]
        K = kurtosis([np.dot(dominant, v) for v in np.array([x, y, z]).T], fisher=False)
    else:
        val, vec = np.linalg.eig(gyration_tensor)
        dominant = vec[:, np.argsort(val)][:, -1]
        K = kurtosis([np.dot(dominant, v) for v in np.array([x, y]).T], fisher=False)
    return K


@njit
def Straightness(x, y, z, SL=None):
    """
    Straightness is a measure of the average direction change between subsequent steps.

    :param x: The x-coordinate of the trajectory.
    :param y: The y-coordinate of the trajectory.
    :param z: The z-coordinate of the trajectory.
    :param SL: The step length of the trajectory.
    """

    if SL is None:
        SL = Step_length(x, y, z)

    N = len(x)

    upper = np.sqrt((x[N - 1] - x[0]) ** 2 + (y[N - 1] - y[0]) ** 2 + (z[N - 1] - z[0]) ** 2)
    lower = sum(SL)
    return upper / lower


def Trappedness(x, maxpair, msds):
    """
    Calculate the trappedness of a trajectory.

    Trappedness is the probability that a diffusing particle with the diffusion coefficient D
    and traced for a time interval t is trapped in a bounded region with radius r0.

    :param x: x-coordinates for the trajectory.
    :param maxpair: Maximum squared pair-wise distance for the poinst in the trajectory.
    :param msds: Mean squared displacements.
    """

    r0 = np.sqrt(maxpair) / 2
    D = msds[1] - msds[0]

    return 1 - np.exp(0.2048 - 0.25117 * (D * len(x)) / r0 ** 2)


def MSDratio(msds):
    """
    Calculate the MSD ratio of a trajectory.

    :param msds: Mean squared displacements of a trajectory.
    """

    return np.mean(
        [msds[i] / msds[i + 1] - (i) / (i + 1) for i in range(len(msds) - 1)]
    )


def D_Agostino_peason(data):
    """
    Verify that the data follows a normal distribution.
    """

    stat, p = normaltest(data)  # stat: s^2 + k^2
    return stat, p


def Max_std(x, y, z=None, step=1):
    """
    Calculate the Maximum standard deviation of a trajectory.

    :param x: The x-coordinate of the trajectory.
    :param y: The y-coordinate of the trajectory.
    :param z: The z-coordinate of the trajectory.
    :param step: The length of window.
    """
    delta_x = np.std(x)
    delta_y = np.std(y)

    std_x = []
    std_y = []
    thr = 1e-10

    for j in range(len(x) - step):
        std_x.append(np.std(x[j: j + step]))
        std_y.append(np.std(y[j: j + step]))

    std_x = [i if i > thr else 1 for i in std_x]
    std_y = [i if i > thr else 1 for i in std_y]

    # Calculate the MAM
    MAM_x = max(std_x) / min(std_x)
    MAM_y = max(std_y) / min(std_y)

    # Calculate the MAC
    MAC_X = max(abs(np.diff(std_x, axis=0))) / delta_x
    MAC_Y = max(abs(np.diff(std_y, axis=0))) / delta_y

    if z is not None:
        delta_z = np.std(z)
        std_z = []
        for j in range(len(x) - step):
            std_z.append(np.std(z[j: j + step]))

        std_z = [i if i > thr else 1 for i in std_z]

        MAM_z = max(std_z) / min(std_z)
        MAC_Z = max(abs(np.diff(std_z, axis=0))) / delta_z

        return MAM_x, MAM_y, MAM_z, MAC_X, MAC_Y, MAC_Z
    else:
        return MAM_x, MAM_y, MAC_X, MAC_Y


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
    r = msds - power(np.arange(1, len(msds) + 1) * dt, popt[0], popt[1])

    popt, pcov = curve_fit(power, t, msds,
                           sigma=np.repeat(np.std(r, ddof=1), len(msds)),
                           p0=[msds[0] / (2 * dim * dt), 1],
                           bounds=[[0.0000001, 0.], [np.inf, 10]],
                           method='trf')
    return popt[0], popt[1]


def dotproduct_traces(trace):
    """
    Calculate the dot product of adjacent steps.
    """
    vecs = trace[1:] - trace[:-1]
    dots = np.dot(vecs[:-1], vecs[1:].T).diagonal()
    return dots


def volume_area_measure(x, y, z=None):
    """Calculate the area of the hull of a trace.
    In 2D, this returns area; in 3D, this returns volume.

    :param x: The x-coordinate of the trajectory.
    :param y: The y-coordinate of the trajectory.
    :param z: The z-coordinate of the trajectory.
    """
    if z is None:
        trace = np.array([x, y]).T
    else:
        trace = np.array([x, y, z]).T
    hull = ConvexHull(trace)
    return hull.volume


def Calculate_step_angle(angle):
    """
    Calculate the step angle of the azimuth and polar angle.
    :param angle: the azimuth angle and polar angle.
    """

    step_angle = np.row_stack((np.zeros((1, 2)), np.abs(np.diff(angle, axis=0))))
    return step_angle


def Angular_velocity_ACF(angle, k=None):
    """
    Calculate the angle auto-correlation function.

    :param angle: Time series data, such as step azimuth angle and polar angle.
    :param k: time lags.
    """

    # subtract the mean value
    if k is None:
        k = len(angle) - 1

    ts = np.deg2rad(angle)
    x = np.array(ts) - np.mean(ts)

    coef = np.zeros(k + 1)

    # Calculate the sequence auto-covariance
    coef[0] = x.dot(x) / len(x)

    # Calculate the i order auto-covariance
    for i in range(1, k + 1):
        coef[i] = x[:-i].dot(x[i:]) / len(x[:-i])

    # Returns the auto-correlation coefficient
    return coef / coef[0]


def Skewness(data):
    """
    Calculate the skewness of the trajectory.
    :param data: step-length, displacement, azimuth or polar angle.
    """
    return skew(data)


def Kurtosis(data):
    """
    Calculate the kurtosis of the trajectory.
    :param data: step-length, displacement, azimuth or polar angle.
    """
    return kurtosis(data)


def Variation_coeff(data):
    """
    Calculate the coefficient of variation for the trajectory.
    :param data: step-length, displacement, azimuth or polar angle.
    """
    return np.std(data) / np.mean(data)


@njit
def Empirical_velocity_autocorr(x, y, z=None):
    """
     Compute the empirical velocity auto-correlation function.
     :param x: The x-coordinate of the trajectory.
     :param y: The y-coordinate of the trajectory.
     :param z: The z-coordinate of the trajectory.
    """

    N = len(x) - 1
    s = np.zeros(N - 1)

    for i in range(0, N - 1):
        V1 = np.array([x[i + 2] - x[i + 1], y[i + 2] - y[i + 1], z[i + 2] - z[i + 1]])
        V2 = np.array([x[i + 1] - x[i], y[i + 1] - y[i], z[i + 1] - z[i]])
        s[i] = np.dot(V1, V2)
    return np.sum(s) / (N - 1)


def Empirical_angle_autocorr(step_angle):
    """
    Compute the empirical angle auto-correlation function.
    :param step_angle: Calculated step azimuth and polar angle.
    """
    N = len(step_angle) - 1
    s = np.zeros(N - 1)

    for i in range(0, N - 1):
        s[i] = np.dot(step_angle[i + 2], step_angle[i + 1])
    return np.sum(s) / (N - 1)


def KS_statistic(data):
    """
    Compute the Kolmogorov-Smirnov test statistic for the trajectory.
    :param data: The coordinates or the displacement of the trajectory,
    """
    ks_statistic, _ = kstest(data, 'norm')
    return ks_statistic


def MSAD(angle, frac=1):
    """
    Calculate the mean square angular displacement in different directions.

    :param angle: The azimuth or polar angle of the trajectory.
    :param frac: The fraction of the trajectory to use for the analysis.
    """

    def AngleSquareDist(x, y):
        return (x - y) ** 2

    N = int(len(angle) * frac)
    msad = []

    for lag in range(1, N):
        msad.append(
            np.mean(
                [
                    AngleSquareDist(angle[j], angle[j + lag])
                    for j in range(len(angle) - lag)
                ]
            )
        )
    return np.array(msad)


def calculate_rotational_diffusion_coefficient(data, dt):
    """
    Calculate the rotational diffusion coefficient.

    :param data: The azimuth or polar angle of the trajectory.
    :param dt: The time interval between two neighboring points.
    :return: Rotational diffusion coefficient and alpha.
    """

    def power(x, D, alpha, offset=0):
        return 2 * D * (x) ** alpha + offset

    msad = MSAD(data)

    # Perform the curve fitting
    t = np.arange(1, len(msad) + 1) * dt

    popt, pcov = curve_fit(power, t, msad,
                           p0=[msad[0] / (2 * dt), 1],
                           bounds=[[0.0000001, 0.], [np.inf, 10]],
                           method='trf')
    return popt[0], popt[1]


def rolling_corrcoef(seq1, seq2, window_size):
    """
    Rolling method to calculate the rotational correlation coefficient.
    For DIC imaging, calculations were performed using bright and dark intensities;
    for dark field imaging, calculations were performed using azimuth and polar angle.

    :param seq1: Bright intensity (DIC) or azimuth angle (dark field).
    :param seq2: Dark intensity (DIC) or polar angle (dark field).
    :param window_size: The size of the rolling window.
    :return: A list of rotational correlation coefficients.
    """

    n = len(seq1)
    corr = []

    for i in range(n):
        start = max(0, i - window_size // 2)
        end = min(n, i + window_size // 2 + 1)
        x_window = seq1[start:end]
        y_window = seq2[start:end]
        if len(x_window) > 1:
            corr.append(np.corrcoef(x_window, y_window)[0, 1])
        else:
            raise ValueError("Window size is 0 or 1.")

    return corr


def GetFeatures(x, y, z=None, dt=0.02, azimuth=None, polar=None, step_angle=False):
    """
    Compute the diffusional fingerprint for a trajectory.

    :param x: The x-coordinate of the trajectory.
    :param y: The y-coordinate of the trajectory.
    :param z: The z-coordinate of the trajectory.
    :param dt: Time interval between adjacent frames.
    :param azimuth: The azimuth angle of particle motion.
    :param polar: The polar angle of particle motion.
    :param step_angle: Whether to use the step angle method for extracting features.

    :return: Extracted all featured fingerprints.
    """
    volume_area = volume_area_measure(x, y, z)
    if z is None:
        z = np.zeros_like(x)
        dim = 2
    else:
        dim = 3

    SL = Step_length(x, y, z)
    msds = MSD(x, y, z, frac=0.5)
    maxpair = GetMax(x, y, z)
    D, alpha = Scalings(msds, dt, dim=dim)
    dis = Displacement(x, y, z)
    gyration_tensor = calculate_covariance(x, y, z)

    if z is None:
        dot = dotproduct_traces(np.array([x, y]).T)
    else:
        dot = dotproduct_traces(np.array([x, y, z]).T)

    shared_features = [alpha,
                       D,
                       np.mean(msds),
                       Asymmetry(gyration_tensor),
                       Efficiency(x, y, z),
                       FractalDim(x, y, z, maxpair),
                       Max_excursion(x, y, z, SL=SL),
                       Gaussianity(x, y, z, msds),
                       Kurtosis_xyz(x, y, z, gyration_tensor),
                       Straightness(x, y, z, SL=SL),
                       Trappedness(x, maxpair, msds),
                       MSDratio(msds),
                       # len(x),
                       Empirical_velocity_autocorr(x, y, z),
                       # (Relaxation_time(x) + Relaxation_time(y) + Relaxation_time(z)) / 3,
                       # *Max_std(x, y, z, step=max_std_step),

                       np.sum(SL),
                       np.mean(SL),
                       np.std(SL),
                       # Relaxation_time(SL),
                       *D_Agostino_peason(SL),
                       Skewness(SL),
                       Kurtosis(SL),
                       Variation_coeff(SL),

                       np.nanmean(dot),
                       np.nanmean(np.sign(dot[1:]) == np.sign(dot[:-1])),
                       np.nanmean(np.sign(dot[1:]) > 0),

                       np.mean(dis),
                       np.std(dis),
                       # Relaxation_time(dis),
                       *D_Agostino_peason(dis),
                       Skewness(dis),
                       Kurtosis(dis),
                       Variation_coeff(dis),
                       volume_area
                       ]

    if azimuth is not None and polar is not None:
        if step_angle:  # The input angle is the step angle.
            step_azimuth, step_polar = azimuth, polar

        else:  # The input angle is not the step angle.
            angle = np.stack([azimuth, polar], axis=1)
            step_angles = Calculate_step_angle(angle)
            step_azimuth, step_polar = step_angles[:, 0], step_angles[:, 1]

        return np.array(
            [
                *shared_features,
                # *calculate_rotational_diffusion_coefficient(step_azimuth, dt),
                # *calculate_rotational_diffusion_coefficient(step_polar, dt),
                # Empirical_angle_autocorr(step_azimuth),
                # Empirical_angle_autocorr(step_polar),
                np.mean(step_azimuth),
                np.mean(step_polar),
                np.std(step_azimuth),
                np.std(step_polar),
                # Relaxation_time(step_azimuth),
                # Relaxation_time(step_polar),
                *D_Agostino_peason(step_azimuth),
                *D_Agostino_peason(step_polar),
                Skewness(step_azimuth),
                Skewness(step_polar),
                Kurtosis(step_azimuth),
                Kurtosis(step_polar),
                Variation_coeff(step_azimuth),
                Variation_coeff(step_polar),
            ])
    else:
        return np.array([*shared_features])


def process_trajectory(idx, data, dt, step_angle, savepath):
    """

    :param idx:
    :param data:
    :param dt:
    :param step_angle:
    :param savepath:
    :return:
    """
    try:
        dim = data.shape[-1]
        x, y = data[:, 0], data[:, 1]

        if dim == 3 or dim == 5:
            z = data[:, 2]
        else:
            z = None

        if dim == 4:
            po, azi = data[:, 2], data[:, 3]
        elif dim == 5:
            po, azi = data[:, 3], data[:, 4]
        else:
            po, azi = None, None

        feature = GetFeatures(x=x, y=y, z=z, dt=dt, azimuth=azi, polar=po, step_angle=step_angle)
        np.save(os.path.join(savepath, f'{idx:06}.npy'), feature)
        return feature

    except Exception as e:
        with open(os.path.join(savepath, 'error_log.txt'), 'a') as f:
            f.write(f'Error in: {idx}\n')
        return None


if __name__ == '__main__':
    print()
    # step_angle = True
    # dt = 0.02
    #
    # path = r'D:\TrajSeg-Cls\TrajSEG-CLS_V3\CLS\Var_LD_500\SNR05_4-5D'
    # for mode in ['train', 'val', 'test']:
    #     savepath = os.path.join(path, 'Feature', mode)
    #     if not os.path.exists(savepath):
    #         os.mkdir(savepath)
    #
    #     data = scipy.io.loadmat(os.path.join(path, f'varLD_{mode}.mat'))
    #     X, Y = data['data'].squeeze(), data['label'].squeeze()
    #     nums = X.shape[0]
    #
    #     features = Parallel(n_jobs=-1)(
    #         delayed(process_trajectory)(i, X[i], dt, step_angle, savepath) for i in range(nums))
    #     # TODO: Remember to modify the step_angle parameter in Get_features according to the actual data.

    # path = r'D:\TrajSeg-Cls\TrajSEG-CLS_V3\CLS\Endocytosis_NEW\5D'
    # data = scipy.io.loadmat(os.path.join(path, 'resample_aug.mat'))['data'].squeeze()[0]
    #
    # x, y, z, azi, po = data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4]
    # feature_origin = GetFeatures(x=x, y=y, z=z, dt=dt, max_std_step=19, azimuth=azi, polar=po, step_angle=False)
    #
    # x, y, z, azi, po = data[:, 0] / 1000, data[:, 1] / 1000, data[:, 2] / 1000, data[:, 3], data[:, 4]
    # feature_1000 = GetFeatures(x=x, y=y, z=z, dt=dt, max_std_step=19, azimuth=azi, polar=po, step_angle=False)
    #
    # x, y, z, azi, po = data[:, 0] / 100, data[:, 1] / 100, data[:, 2] / 100, data[:, 3], data[:, 4]
    # feature_100 = GetFeatures(x=x, y=y, z=z, dt=dt, max_std_step=19, azimuth=azi, polar=po, step_angle=False)
    #
    # x, y, z, azi, po = data[:, 0] / 10, data[:, 1] / 10, data[:, 2] / 10, data[:, 3], data[:, 4]
    # feature_10 = GetFeatures(x=x, y=y, z=z, dt=dt, max_std_step=19, azimuth=azi, polar=po, step_angle=False)
    #
    # scipy.io.savemat(os.path.join(r'C:\Users\songn\Desktop','feature_compare.mat'),
    #                  {'origin':feature_origin,
    #                   'feature_1000': feature_1000,
    #                   'feature_100': feature_100,
    #                   'feature_10': feature_10
    #                   }
    #                  )