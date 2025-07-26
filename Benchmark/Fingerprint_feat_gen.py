# -*- coding: utf-8 -*-
# @Time    : 2024/3/1 15:32
# @Author  : Dongliang


from AAISPT.Feature_extraction.Fingerprint_feat_gen import *


def gen_AnDi_feature(x, y, z=None, dt=0.02):
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
    return GetFeatures(x, y, z, dt)


def process_trajectory(idx, data, dt, savepath):
    """
    Process a trajectory dataset to generate features and save them to a file.

    :param idx: Integer identifier for the trajectory.
    :param data: NumPy array containing trajectory data with shape (n, dim), where n is the number of points
                 and dim is the dimensionality (2 for 2D, 3 for 3D).
    :param dt: Time step between consecutive points in the trajectory.
    :param savepath: Directory path where the output feature file will be saved.
    :return: Generated feature array if successful, None if an error occurs.
    """
    try:
        dim = data.shape[-1]
        x, y = data[:, 0], data[:, 1]

        if dim == 3:
            z = data[:, 2]
        else:
            z = None

        feature = gen_AnDi_feature(x=x, y=y, z=z, dt=dt)
        np.save(os.path.join(savepath, f'{idx:06}.npy'), feature)
        return feature

    except Exception as e:
        with open(os.path.join(savepath, 'error_log.txt'), 'a') as f:
            f.write(f'Error in: {idx}\n')
        return None

