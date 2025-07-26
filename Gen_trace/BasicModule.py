# -*- coding: utf-8 -*-
# @Time    : 2024/5/4 20:41
# @Author  : Dongliang


import numpy as np
import scipy.stats as stats


def gen_truncnorm(min_val, max_val, mean, std, num):
    """
    Generate Truncated Normal distribution.

    :param min_val: Lower bound of the truncated distribution.
    :param max_val: Upper bound of the truncated distribution
    :param mean: Mean of the truncated distribution.
    :param std: Standard deviation of the truncated distribution.
    :param num: The number of random numbers to generate.
    """
    dist = stats.truncnorm((min_val - mean) / std, (max_val - mean) / std, loc=mean, scale=std)
    values = dist.rvs(num)
    return values


class gen_trace():
    """Generate a single type of diffusion trajectory."""
    def gen_polarpara(self, min_val, max_val, total_mean, total_std, num_trace, lamda_std):
        """
        Generate step polar angle simulation parameters. (Biophys J. 2021,120(8),1378-1386)

        :param min_val: Minimum value of the step polar angle.
        :param max_val: Maximum value of the step polar angle.
        :param total_mean: Overall mean of the step polar angle.
        :param total_std: Overall standard deviation of the step polar angle.
        :param num_trace: Number of simulated trajectories.
        :param lamda_std: Calibration parameters.
        :return: Generated simulation parameters.
        """
        step_polar_mean = gen_truncnorm(min_val, max_val, total_mean, total_std, num_trace)
        step_polar_std = np.random.rand(num_trace) + lamda_std
        return step_polar_mean, step_polar_std

    def gen_azimuthpara(self, min_val, max_val, total_mean, total_std, num_trace, lamda_std):
        """
        Generate step azimuth angle simulation parameters. (Biophys J. 2021,120(8),1378-1386)

        :param min_val: Minimum value of the step azimuth angle.
        :param max_val: Maximum value of the step azimuth angle.
        :param total_mean: Overall mean of the step azimuth angle.
        :param total_std: Overall standard deviation of the step azimuth angle.
        :param num_trace: Number of simulated trajectories.
        :param lamda_std: Calibration parameters.
        :return:  Generated simulation parameters.
        """
        step_azimuth_mean = gen_truncnorm(min_val, max_val, total_mean, total_std, num_trace)
        step_azimuth_std = np.random.rand(num_trace) + lamda_std
        return step_azimuth_mean, step_azimuth_std

    def gen_steppolar(self, num_trace, ini_frames, min_val, max_val, length, step_polar_mean, step_polar_std):
        """
        Generate step polar angle simulation trajectories.

        :param num_trace: Number of simulated trajectories.
        :param ini_frames: The initial value of the trajectory length.
        :param min_val: Minimum value of the step polar angle.
        :param max_val: Maximum value of the step polar angle.
        :param length: The length of the trajectory.
        :param step_polar_mean: Simulated mean parameters for step polar angle.
        :param step_polar_std: Simulated standard deviation parameters for step polar angle.
        :return: Generate a simulated step polar angle dataset.
        """
        final_step_polars = np.zeros((num_trace, ini_frames), dtype='float32')
        for i in range(num_trace):
            step_polar = gen_truncnorm(min_val, max_val, step_polar_mean, step_polar_std, length)
            if step_polar.shape[0] == ini_frames:
                final_step_polars[i] = step_polar
            else:
                c = [0] * np.ones((ini_frames - step_polar.shape[0]), dtype='float32')
                final_step_polars[i] = np.hstack((c, step_polar))
        return final_step_polars

    def gen_stepazimuth(self, num_trace, ini_frames, min_val, max_val, length, step_azimuth_mean, step_azimuth_std):
        """
        Generate step azimuth angle simulation trajectories.

        :param num_trace: Number of simulated trajectories.
        :param ini_frames: The initial value of the trajectory length.
        :param min_val: Minimum value of the step azimuth angle.
        :param max_val: Maximum value of the step azimuth angle.
        :param length: The length of the trajectory.
        :param step_azimuth_mean: Simulated mean parameters for step azimuth angle.
        :param step_azimuth_std: Simulated standard deviation parameters for step azimuth angle.
        :return: Generate a simulated step azimuth angle dataset.
        """
        final_step_azimuths = np.zeros((num_trace, ini_frames), dtype='float32')
        for i in range(num_trace):
            step_azimuth = gen_truncnorm(min_val, max_val, step_azimuth_mean, step_azimuth_std, length)
            if step_azimuth.shape[0] == ini_frames:
                final_step_azimuths[i] = step_azimuth
            else:
                c = [0] * np.ones((ini_frames - step_azimuth.shape[0]), dtype='float32')
                final_step_azimuths[i] = np.hstack((c, step_azimuth))
        return final_step_azimuths


def gen_ND_diff(
        step_polar_meanND,
        step_polar_stdND,
        step_azimuth_meanND,
        step_azimuth_stdND,
        num_perclass,
        trace_len,
        NsND
):
    """
    Generate Normal Diffusion rotation information
    :param step_polar_meanND: Simulated mean parameters for step polar angle.
    :param step_polar_stdND: Simulated standard deviation parameters for step polar angle.
    :param step_azimuth_meanND: Simulated mean parameters for step azimuth angle.
    :param step_azimuth_stdND: Simulated standard deviation parameters for step azimuth angle.
    :param num_perclass: The number of trajectories of each type.
    :param trace_len: The length of the trajectory.
    :param NsND: An ndarray containing num_perclass elements, where each element stores the length of each trajectory.
    :return: Generate a Normal Diffusion step angle dataset.
    """

    ND = gen_trace()
    s_polar_meanND, s_polar_stdND = ND.gen_polarpara(
        min_val=0,
        max_val=35,
        total_mean=step_polar_meanND,
        total_std=step_polar_stdND,
        num_trace=num_perclass,
        lamda_std=3
    )

    s_azimuth_meanND, s_azimuth_stdND = ND.gen_azimuthpara(
        min_val=0,
        max_val=120,
        total_mean=step_azimuth_meanND,
        total_std=step_azimuth_stdND,
        num_trace=num_perclass,
        lamda_std=5
    )

    step_polars = np.zeros((num_perclass, trace_len), dtype='float32')
    step_azimuths = np.zeros((num_perclass, trace_len), dtype='float32')
    for index, i in enumerate(NsND):
        step_polars[index] = ND.gen_steppolar(
            num_trace=1,
            ini_frames=trace_len,
            min_val=0,
            max_val=90,
            length=int(i),
            step_polar_mean=s_polar_meanND[index],
            step_polar_std=s_polar_stdND[index],
        )

        step_azimuths[index] = ND.gen_stepazimuth(
            num_trace=1,
            ini_frames=trace_len,
            min_val=0,
            max_val=180,
            length=int(i),
            step_azimuth_mean=s_azimuth_meanND[index],
            step_azimuth_std=s_azimuth_stdND[index],
        )
    print('Normal diffusion rotation has been completed!')
    return step_polars, step_azimuths


def gen_TA_diff(
        step_polar_meanTA,
        step_polar_stdTA,
        step_azimuth_meanTA,
        step_azimuth_stdTA,
        num_perclass,
        trace_len,
        NsTA
):
    """
    Generate Tight Attachment diffusion rotation information.
    :param step_polar_meanTA: Simulated mean parameters for step polar angle.
    :param step_polar_stdTA: Simulated standard deviation parameters for step polar angle.
    :param step_azimuth_meanTA: Simulated mean parameters for step azimuth angle.
    :param step_azimuth_stdTA: Simulated standard deviation parameters for step azimuth angle.
    :param num_perclass: The number of trajectories of each type.
    :param trace_len: The length of the trajectory.
    :param NsTA: An ndarray containing num_perclass elements, where each element stores the length of each trajectory.
    :return: Generate a Tight Attachment step angle dataset.
    """

    TA = gen_trace()
    s_polar_meanTA, s_polar_stdTA = TA.gen_polarpara(
        min_val=0,
        max_val=20,
        total_mean=step_polar_meanTA,
        total_std=step_polar_stdTA,
        num_trace=num_perclass,
        lamda_std=3
    )

    s_azimuth_meanTA, s_azimuth_stdTA = TA.gen_azimuthpara(
        min_val=0,
        max_val=30,
        total_mean=step_azimuth_meanTA,
        total_std=step_azimuth_stdTA,
        num_trace=num_perclass,
        lamda_std=3
    )

    step_polars = np.zeros((num_perclass, trace_len), dtype='float32')
    step_azimuths = np.zeros((num_perclass, trace_len), dtype='float32')
    for index, i in enumerate(NsTA):
        step_polars[index] = TA.gen_steppolar(
            num_trace=1,
            ini_frames=trace_len,
            min_val=0,
            max_val=90,
            length=int(i),
            step_polar_mean=s_polar_meanTA[index],
            step_polar_std=s_polar_stdTA[index],
        )
        step_azimuths[index] = TA.gen_stepazimuth(
            num_trace=1,
            ini_frames=trace_len,
            min_val=0,
            max_val=180,
            length=int(i),
            step_azimuth_mean=s_azimuth_meanTA[index],
            step_azimuth_std=s_azimuth_stdTA[index],
        )
    step_polars -= 2
    step_azimuths -= 2
    print('Tight attachment rotation has been completed!')
    return step_polars, step_azimuths


def gen_TR_diff(
        step_polar_meanTR,
        step_polar_stdTR,
        step_azimuth_meanTR,
        step_azimuth_stdTR,
        num_perclass,
        trace_len,
        NsTR
):
    """
    Generate Tethered Rotation information.
    :param step_polar_meanTR: Simulated mean parameters for step polar angle.
    :param step_polar_stdTR: Simulated standard deviation parameters for step polar angle.
    :param step_azimuth_meanTR: Simulated mean parameters for step azimuth angle.
    :param step_azimuth_stdTR: Simulated standard deviation parameters for step azimuth angle.
    :param num_perclass: The number of trajectories of each type.
    :param trace_len: The length of the trajectory.
    :param NsTR: An ndarray containing num_perclass elements, where each element stores the length of each trajectory.
    :return: Generate a Tethered Rotation step angle dataset.
    """

    TR = gen_trace()
    s_polar_meanTR, s_polar_stdTR = TR.gen_polarpara(
        min_val=0,
        max_val=20,
        total_mean=step_polar_meanTR,
        total_std=step_polar_stdTR,
        num_trace=num_perclass,
        lamda_std=4
    )
    s_azimuth_meanTR, s_azimuth_stdTR = TR.gen_azimuthpara(
        min_val=0,
        max_val=30,
        total_mean=step_azimuth_meanTR,
        total_std=step_azimuth_stdTR,
        num_trace=num_perclass,
        lamda_std=4
    )
    step_polars = np.zeros((num_perclass, trace_len), dtype='float32')
    step_azimuths = np.zeros((num_perclass, trace_len), dtype='float32')
    for index, i in enumerate(NsTR):
        step_polars[index] = TR.gen_steppolar(
            num_trace=1,
            ini_frames=trace_len,
            min_val=0,
            max_val=90,
            length=int(i),
            step_polar_mean=s_polar_meanTR[index],
            step_polar_std=s_polar_stdTR[index],
        )
        step_azimuths[index] = TR.gen_stepazimuth(
            num_trace=1,
            ini_frames=trace_len,
            min_val=0,
            max_val=180,
            length=int(i),
            step_azimuth_mean=s_azimuth_meanTR[index],
            step_azimuth_std=s_azimuth_stdTR[index],
        )
    step_polars -= 2.5
    step_azimuths -= 2.5
    print('Tethered Rotation rotation has been completed!')
    return step_polars, step_azimuths


def gen_DM_diff(
        step_polar_meanDM,
        step_polar_stdDM,
        step_azimuth_meanDM,
        step_azimuth_stdDM,
        num_perclass,
        trace_len,
        NsDM
):
    """
    Generate Directed Motion rotation information.
    :param step_polar_meanDM: Simulated mean parameters for step polar angle.
    :param step_polar_stdDM: Simulated standard deviation parameters for step polar angle.
    :param step_azimuth_meanDM: Simulated mean parameters for step azimuth angle.
    :param step_azimuth_stdDM: Simulated standard deviation parameters for step azimuth angle.
    :param num_perclass: The number of trajectories of each type.
    :param trace_len: The length of the trajectory.
    :param NsDM: An ndarray containing num_perclass elements, where each element stores the length of each trajectory.
    :return: Generate a Directed Motion step angle dataset.
    """

    DM = gen_trace()
    s_polar_meanDM, s_polar_stdDM = DM.gen_polarpara(
        min_val=0,
        max_val=10,
        total_mean=step_polar_meanDM,
        total_std=step_polar_stdDM,
        num_trace=num_perclass,
        lamda_std=1.5
    )
    s_azimuth_meanDM, s_azimuth_stdDM = DM.gen_azimuthpara(
        min_val=0,
        max_val=10,
        total_mean=step_azimuth_meanDM,
        total_std=step_azimuth_stdDM,
        num_trace=num_perclass,
        lamda_std=2
    )
    step_polars = np.zeros((num_perclass, trace_len), dtype='float32')
    step_azimuths = np.zeros((num_perclass, trace_len), dtype='float32')
    for index, i in enumerate(NsDM):
        step_polars[index] = DM.gen_steppolar(
            num_trace=1,
            ini_frames=trace_len,
            min_val=0,
            max_val=90,
            length=int(i),
            step_polar_mean=s_polar_meanDM[index],
            step_polar_std=s_polar_stdDM[index],
        )
        step_azimuths[index] = DM.gen_stepazimuth(
            num_trace=1,
            ini_frames=trace_len,
            min_val=0,
            max_val=90,
            length=int(i),
            step_azimuth_mean=s_azimuth_meanDM[index],
            step_azimuth_std=s_azimuth_stdDM[index],
        )
    print('Directed diffusion rotation has been completed!')
    return step_polars, step_azimuths


def gen_DMR_diff(
        step_polar_meanDMR,
        step_polar_stdDMR,
        step_azimuth_meanDMR,
        step_azimuth_stdDMR,
        num_perclass,
        trace_len,
        NsDMR
):
    """
    Generate Directed Motion and fast rotation information
    :param step_polar_meanDMR: Simulated mean parameters for step polar angle.
    :param step_polar_stdDMR: Simulated standard deviation parameters for step polar angle.
    :param step_azimuth_meanDMR: Simulated mean parameters for step azimuth angle.
    :param step_azimuth_stdDMR: Simulated standard deviation parameters for step azimuth angle.
    :param num_perclass: The number of trajectories of each type.
    :param trace_len: The length of the trajectory.
    :param NsDMR: An ndarray containing num_perclass elements, where each element stores the length of each trajectory.
    :return: Generate a Directed Motion and fast rotation step angle dataset.
    """

    DMR = gen_trace()
    s_polar_meanDMR, s_polar_stdDMR = DMR.gen_polarpara(
        min_val=0,
        max_val=30,
        total_mean=step_polar_meanDMR,
        total_std=step_polar_stdDMR,
        num_trace=num_perclass,
        lamda_std=3
    )
    s_azimuth_meanDMR, s_azimuth_stdDMR = DMR.gen_azimuthpara(
        min_val=0,
        max_val=30,
        total_mean=step_azimuth_meanDMR,
        total_std=step_azimuth_stdDMR,
        num_trace=num_perclass,
        lamda_std=3
    )
    DMRstep_polars = np.zeros((num_perclass, trace_len), dtype='float32')
    DMRstep_azimuths = np.zeros((num_perclass, trace_len), dtype='float32')
    for index, i in enumerate(NsDMR):
        DMRstep_polars[index] = DMR.gen_steppolar(
            num_trace=1,
            ini_frames=trace_len,
            min_val=0,
            max_val=90,
            length=i,
            step_polar_mean=s_polar_meanDMR[index],
            step_polar_std=s_polar_stdDMR[index],
        )
        DMRstep_azimuths[index] = DMR.gen_stepazimuth(
            num_trace=1,
            ini_frames=trace_len,
            min_val=0,
            max_val=180,
            length=i,
            step_azimuth_mean=s_azimuth_meanDMR[index],
            step_azimuth_std=s_azimuth_stdDMR[index],
        )
    print('Directed diffusion and fast rotation has been completed!')
    return DMRstep_polars, DMRstep_azimuths
