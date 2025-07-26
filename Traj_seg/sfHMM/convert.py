# -*- coding: utf-8 -*-
# @Time    : 2023/7/20 16:17
# @Author  : Dongliang

import os
import sys

# Defining Path Variables
base_path = r"D:\PythonCode\TrajSEG-CLS\sfHMM-main\sample_data"
path1 = os.path.join(base_path, '')
sys.path.append(path1)

# Get the names of all files in the directory
files = os.listdir(base_path)
print('files', files)

# Iterate over files and rename them
for filename in files:
    name, ext = os.path.splitext(filename)
    if ext == ".dat":
        newname = name + ".txt"
        old_path = os.path.join(base_path, filename)
        new_path = os.path.join(base_path, newname)
        os.rename(old_path, new_path)
