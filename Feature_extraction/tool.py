# -*- coding: utf-8 -*-
# @Time    : 2024/12/29 18:01
# @Author  : Dongliang

import glob
import os

import numpy as np
import scipy


# path = 'D:\TrajSeg-Cls\TrajSEG-CLS_V3\CLS\Var_LD_500\SNR05_4-5D'

# for mode in [ 'train','val','test']:
#     Y = scipy.io.loadmat(os.path.join(path, f'varLD_{mode}.mat'))['label'].squeeze()
#
#     dirs = os.path.join(path,'Feature', mode)
#     features = []
#     for i in glob.glob(os.path.join(dirs, '*.npy')):
#         data = np.load(i).squeeze()
#         _, idx = np.unique(data, axis=0, return_index=True)
#         UNIQUE = data[np.sort(idx)]
#         features.append(UNIQUE)
#
#     scipy.io.savemat(os.path.join(dirs,f'Roll_{mode}_feature.mat'), {'data': features, 'label': Y})


# for mode in ['train', 'val', 'test']:
#     Y = scipy.io.loadmat(os.path.join(path, f'varLD_{mode}.mat'))['label']
#     dirs = os.path.join(path, 'Feature', mode)
#     files = glob.glob(os.path.join(dirs, '*.npy'))
#     features = [np.load(i) for i in files]
#
#     scipy.io.savemat(os.path.join(dirs, f'{mode}_feature.mat'), {'data': features, 'label': Y})


def check_missing_files(folder_path, prefix='', suffix=''):
    """
    Checks that file names in a folder are in order and displays missing file names.
    :param folder_path: Folder Path.
    :param prefix: File name prefix (e.g. ‘file_’).
    :param suffix: File name suffix (e.g. ‘.npy’).
    """

    # Gets the names of all files in the folder.
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    # Filters files that match prefixes and suffixes.
    files = [f for f in files if f.startswith(prefix) and f.endswith(suffix)]

    # Extracts the numeric portion of the filename and sorts it.
    file_numbers = sorted(int(f[len(prefix):-len(suffix)]) for f in files)

    # Find the missing document number.
    missing_files = []
    for num in range(file_numbers[0], file_numbers[-1] + 1):
        if num not in file_numbers:
            missing_files.append(f"{prefix}{num}{suffix}")

    if missing_files:
        print("Missing files:")
        for missing_file in missing_files:
            print(missing_file)
    else:
        print("All documents are in order and not missing.")


if __name__ == '__main__':
    # folder_path = r"D:\TrajSeg-Cls\endoysis\Aug_SPT_FLU\Roll_Feature\features"
    # check_missing_files(folder_path, prefix='', suffix='.npy')

    # path = r'D:\TrajSeg-Cls\TrajSEG-CLS_V3\CLS\Andi\3D'

    # for mode in ['train', 'val', 'test']:
    #     Y = scipy.io.loadmat(os.path.join(path, f'raw{mode}.mat'))['label'].transpose()
    #     dirs = os.path.join(path, 'Feature', mode)
    #     files = glob.glob(os.path.join(dirs, '*.npy'))
    #     features = [np.load(i) for i in files]
    #
    #     idx = []
    #     with open(os.path.join(dirs, 'error_log.txt')) as f:
    #         for line in f:
    #             idx.append(int(line.split(':')[-1]))
    #
    #     id = np.array(idx)
    #     new_Y = np.delete(Y, id, axis=0)
    #
    #     scipy.io.savemat(os.path.join(dirs, f'{mode}_feature.mat'), {'data': features, 'label': new_Y})


    # for mode in ['train', 'val', 'test']:
    #     Y = scipy.io.loadmat(os.path.join(path, f'raw{mode}.mat'))['label'].transpose()
    #     dirs = os.path.join(path, 'Roll_Feature', mode)
    #     files = glob.glob(os.path.join(dirs, '*.npy'))
    #     features = [np.load(i).squeeze() for i in files]
    #
    #     idx = []
    #     with open(os.path.join(dirs, 'error_log.txt')) as f:
    #         for line in f:
    #             idx.append(int(line.split('：')[-1]))
    #
    #     id = np.array(idx)
    #     new_Y = np.delete(Y, id, axis=0)
    #
    #     scipy.io.savemat(os.path.join(dirs, f'{mode}_feature.mat'), {'data': features, 'label': new_Y})
    #

    path = r'D:\TrajSeg-Cls\endoysis\Aug_SPT_FLU'

    # Y = scipy.io.loadmat(os.path.join(path, 'aug_data.mat'))['label'].transpose()
    Y = scipy.io.loadmat(os.path.join(path,'aug_data.mat'))['label'].squeeze(axis=2)
    dirs = os.path.join(path, r'Roll_Feature\features')
    # dirs = path
    files = glob.glob(os.path.join(dirs, '*.npy'))
    features = [np.load(i).squeeze() for i in files]

    idx = []
    with open(os.path.join(dirs, 'error_log.txt')) as f:
        for line in f:
            idx.append(int(line.split(':')[-1]))

    id = np.array(idx)
    new_Y = np.delete(Y, id, axis=0)

    scipy.io.savemat(os.path.join(dirs, f'Roll_feature.mat'), {'data': features, 'label': new_Y})
