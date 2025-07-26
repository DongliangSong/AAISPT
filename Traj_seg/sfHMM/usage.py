# -*- coding: utf-8 -*-
# @Time    : 2023/7/20 16:50
# @Author  : Dongliang

import os
import time
from glob import glob

import matplotlib.pyplot as plt
import numpy as np

from AISPT.Traj_seg.sfHMM import sfHMM1

plt.style.use("default")


def process_single_trajectory(path: str, psf: float = 0.1, n_init: int = 5, plot: bool = False):
    # Load data
    df = np.loadtxt(path)

    # Initialize sfHMM object
    sf = sfHMM1(df, psf=psf, name="test")

    # Step finding
    sf.step_finding()

    # Denoising
    sf.denoising()

    # GMM fitting
    sf.gmmfit(n_init=n_init)

    # HMM fitting
    sf.hmmfit()

    # Save results
    filename = os.path.split(path)[-1]
    savename = os.path.join(os.path.dirname(path), 'sfHMM_' + filename)
    sf.save(savename)

    # Plot if enabled
    if plot:
        sf.plot()


def process_batch_trajectories(paths: list, psf: float = 0.03, n_init: int = 5, plot: bool = False):
    for path in paths:
        process_single_trajectory(path, psf=psf, n_init=n_init, plot=plot)


if __name__ == '__main__':
    mode = "single"
    start = time.time()
    if mode == "single":
        path = r'C:\Users\songn\Desktop\dis.txt'
        process_single_trajectory(path)
    else:
        print("Batch usage")
        paths = glob(r'D:\TrajSeg-Cls\Exp Demo\19 01072020_2 perfect\6700-8200\sfHMM\*')
        process_batch_trajectories(paths)

    end = time.time()
    print('A total of {} s.'.format(end - start))
