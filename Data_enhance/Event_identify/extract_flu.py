# -*- coding: utf-8 -*-
# @Time    : 2024/12/17 16:37
# @Author  : Dongliang

import glob
import os
import re

import numpy as np
import pandas as pd

# Load fluorescence data
path = r'D:\TrajSeg-Cls\endoysis\Aug_SPT_FLU\cla_dyn_twist_fission.xlsx'
data = pd.read_excel(path, header=None).to_numpy()
data = data[2:, :]
t = data[:, 0]
flu = data[:, 1:]  # fluorescence intensity

# Load group name
with open(r'D:\TrajSeg-Cls\endoysis\Aug_SPT_FLU\name.txt', 'r') as f:
    name = [line.strip() for line in f.readlines()]

# Load time points
time = pd.read_excel(path, sheet_name='twist-fission').to_numpy()

# Get xyzap information for each group
roots = []
root_dir = r'D:\TrajSeg-Cls\endoysis'
for folder in os.listdir(root_dir):
    folder_path = os.path.join(root_dir, folder)
    if os.path.isdir(folder_path):
        for i in range(len(name)):
            str1 = re.sub(r'[-]', '_', name[i])  # regex replace
            if str1 in folder:
                roots.append(folder_path)

# Initialize trace storage
trace = []
trace_time = []

# Loop over each group (assuming that `roots` and `name` have matching lengths)
for i in range(len(name)):
    # Get fluorescence time
    cla_time = t[~np.isnan(flu[:, i * 2].astype('float32'))]

    # Find corresponding SPT file
    filename = glob.glob(os.path.join(roots[i], '*.xlsx'))
    for j in filename:
        df = pd.read_excel(j)
        if 't' in df.columns:
            spt_time = [df['t'].min(), df['t'].max()]
            final_file = df.to_numpy()

    # Determine start and end time
    print(i)
    if cla_time.shape[0] != 0:
        if min(cla_time) > spt_time[0]:
            start = min(cla_time)
        else:
            start = spt_time[0]
    else:
        start = spt_time[0]

    if time[i, -1] + 2 < spt_time[1]:  # DNM fission add 2 seconds
        ends = time[i, -1] + 2
    else:
        ends = spt_time[1]

    trace_time.append([start, ends])

    # Extract SPT data based on time range
    tolerance = 1e-10
    tt = final_file[:, 0]
    spt_start = np.where(np.abs(tt - start) < tolerance)[0][0]
    spt_end = np.where(np.abs(tt - ends) < tolerance)[0][0]
    xyzap = final_file[spt_start:spt_end + 1, :]

    # Extract fluorescence data based on the same time range
    flu_start = np.where(np.abs(t - start) < tolerance)[0][0]
    flu_end = np.where(np.abs(t - ends) < tolerance)[0][0]
    dnm = flu[flu_start:flu_end + 1, i * 2]
    trace.append(np.column_stack((xyzap, dnm)))

print('Done!')
