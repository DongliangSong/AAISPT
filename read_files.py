# -*- coding: utf-8 -*-
# @Time    : 2025/7/24 21:05
# @Author  : Dongliang

from pathlib import Path
import pandas as pd
import numpy as np
import scipy
import json
import pickle
import h5py
import yaml

def load_data(filepath):
    path = Path(filepath)
    suffix = path.suffix.lower()

    if suffix in ['.csv', '.tsv']:
        return pd.read_csv(path, sep='\t' if suffix == '.tsv' else ',').values
    elif suffix == '.xlsx':
        return pd.read_excel(path).values

    # TODO: Continue adding input data in other formats.
    # elif suffix == '.json':
    #     return json.load(open(path, 'r', encoding='utf-8'))
    # elif suffix == '.mat':
    #     return scipy.io.loadmat(path)
    # elif suffix == '.h5':
    #     return h5py.File(path, 'r')
    # elif suffix == '.npy':
    #     return np.load(path)
    # elif suffix == '.npz':
    #     return dict(np.load(path))
    # elif suffix == '.pkl':
    #     return pickle.load(open(path, 'rb'))
    # elif suffix in ['.yaml', '.yml']:
    #     return yaml.safe_load(open(path, 'r', encoding='utf-8'))
    # elif suffix == '.txt':
    #     with open(path, 'r', encoding='utf-8') as f:
    #         return f.readlines()
    else:
        raise ValueError(f"Unsupported file type: {suffix}")
