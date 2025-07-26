# -*- coding: utf-8 -*-
# @Time    : 2024/6/11 15:32
# @Author  : Dongliang

from AAISPT.Data_enhance.Exp_data_enhance import *

path = glob.glob('D:\TrajSeg-Cls\endoysis\Enhanced\Transform\*.xlsx')
savepath = r'D:\TrajSeg-Cls\TrajSEG-CLS_V3\CLS\Endocytosis_NEW\small'

if not os.path.exists(savepath):
    os.mkdir(savepath)

# Experimental dataset enhancement
expand_nums = 2

# Load dataset and labels
dataset = []
label = []
for i in path:
    data = pd.read_excel(i, dtype=np.float32).values[:, 2:]
    dataset.append(data)

    if 'stage1' in i:
        label.append(0)
    elif 'stage2' in i:
        label.append(1)
    elif 'stage3' in i:
        label.append(2)
    elif 'stage4' in i:
        label.append(3)

aug_methods = {
    'Multi_shift': Multi_shift,
    'Multi_interpolate': Multi_interpolate,
    'Multi_addnoise': Multi_addnoise,
    'Multi_windowslice': Multi_windowslice,
    'Multi_reverse': Multi_reverse
}

sing_params = {
    'Multi_shift': {'min_shift_ratio': 0.1, 'max_shift_ratio': 0.6, 'const': 0.05},
    'Multi_interpolate': {'factor': 3, 'const': 1},
    'Multi_addnoise': {'min_level_1': 0.001, 'max_level_1': 0.002, 'min_level_2': 0.01, 'max_level_2': 0.05},
    'Multi_windowslice': {'max_ratio': 0.7},
    'Multi_reverse': {'const': 0}
}

# Set parameters for data augmentation
comb_params = {
    'Multi_shift': {'min_shift_ratio': 0.1, 'max_shift_ratio': 0.6, 'const': 0.05},
    'Multi_interpolate': {'factor': 3, 'const': 1},
    'Multi_addnoise': {'min_level_1': 0.001, 'max_level_1': 0.002, 'min_level_2': 0.01, 'max_level_2': 0.05},
    'Multi_windowslice': {'max_ratio': 0.7},
    'Multi_reverse': {'const': 0}
}

# Single method for data enhancement
print('Single method for data enhancement...')
single_augs = single_methods(
    data=dataset,
    aug_methods=aug_methods,
    nums=expand_nums,
    params=sing_params)

single_augs_data = []
num_methods = len(aug_methods)
for i, single in enumerate(single_augs):
    if i < num_methods - 1:
        for j in range(len(single)):
            single_augs_data.extend(single[j])
    else:
        single_augs_data.extend(single)

single_augs_label = label * (expand_nums * (num_methods - 1) + 1)
print('single_augs_data:', len(single_augs_data), 'single_augs_label:', len(single_augs_label))

# Combine methods for data enhancement
print('Combine methods for data enhancement...')
combinations = combine_methods(
    data=dataset,
    aug_methods=aug_methods,
    nums=expand_nums,
    params=comb_params
)
comb_augs_data = []
for i, comb in enumerate(combinations):
    for j in range(len(comb)):
        comb_augs_data.extend(comb[j])

comb_augs_label = label * expand_nums * (num_methods * (num_methods - 1))
print('comb_augs_data:', len(comb_augs_data), 'comb_augs_label:', len(comb_augs_label))

print('Save data...')
merge_augs = single_augs_data + comb_augs_data
merge_label = single_augs_label + comb_augs_label
scipy.io.savemat(os.path.join(savepath, 'aug_data.mat'), {'data': merge_augs, 'label': merge_label})

# Parameters are written to the log and saved.
current_date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
file_name = os.path.join(savepath, f'params_log_{current_date}.txt')

with open(file_name, 'w', encoding='utf-8') as file:
    file.write('Single method params\n')
    for key, value in sing_params.items():
        file.write(f'{key}: {value}\n')
    file.write('\n')
    file.write('Combination method params\n')
    for key, value in comb_params.items():
        file.write(f'{key}: {value}\n')

    file.write('\n')
    file.write('expand_nums: {}\n'.format(expand_nums))
    file.write('total_nums: {}\n'.format(len(merge_augs)))

print('Done!')
