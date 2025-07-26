# -*- coding: utf-8 -*-
# @Time    : 2024/6/28 21:35
# @Author  : Dongliang

path = r'D:\TrajSeg-Cls\TrajSEG-CLS_V3\CLS\Endocytosis_NEW\5D\Feature_based'


def config(mode, path):
    if mode == 'Sim':
        if '2D' in path or '3D' in path:
            num_class = 3
            label_name = ['ND', 'CD', 'DM']

        elif '4D' in path or '5D' in path:
            num_class = 5
            label_name = ['ND', 'TA', 'TR', 'DMR', 'DM']

    elif mode == 'Exp':
        if 'YanYu' in path or 'Janus' in path:
            num_class = 3
            label_name = ['Circling', 'Confined', 'Rocking']

        elif 'QiPan' in path or 'Phase' in path:
            num_class = 3
            label_name = ['Semi-fluidic', 'Transition', 'Non-fluidic']

        elif 'endocytosis' in path.lower():
            num_class = 4
            label_name = ['EI', 'CCP_F', 'CCP_M', 'VR']
            # EI: endocytosis initial
            # CCP_F: CCP formation
            # CCP_M: CCP maturity
            # VR: Vesicle release

    return num_class, label_name


xyz_feature = [
    'alpha',
    'D',
    'Mean_msds',
    'Asymmetry',
    'Efficiency',
    'FractalDim',
    'Max_excursion',
    'Gaussianity',
    'Kurtosis_xy',
    'Straightness',
    'Trappedness',
    'MSDratio',
    # 'Length',
    'Empirical_velcor',
    # 'MAM_x',
    # 'MAM_y',
    # 'MAM_z',
    # 'MAC_X',
    # 'MAC_Y',
    # 'MAC_Z',
    'Sum_SL',
    'Mean_SL',
    'Std_SL',
    'Stat_SL',
    'p_SL',
    'Skewness_SL',
    'Kurtosis_SL',
    'Var_coeff_SL',
    'Mean_dot',
    'Mean_signdot',
    'Mean_signdot_p',
    'Mean_dis',
    'Std_dis',
    'Stat_dis',
    'p_dis',
    'Skewness_dis',
    'Kurtosis_dis',
    'Var_coeff_dis',
    'Volume_area'
]

rotation_feature = [
    'Mean_Δφ',
    'Mean_Δθ',
    'Std_Δφ',
    'Std_Δθ',
    'Stat_Δφ',
    'p_Δφ',
    'Stat_Δθ',
    'p_Δθ',
    'Skewness_Δφ',
    'Skewness_Δθ',
    'Kurtosis_Δφ',
    'Kurtosis_Δθ',
    'Var_coeff_Δφ',
    'Var_coeff_Δθ'
]


def config_feature(path):
    if 'QiPan' in path or '2D' in path:
        dim = 2
        feature_names = xyz_feature
    elif 'kevin' in path or '3D' in path:
        dim = 3
        feature_names = xyz_feature
    elif 'YanYu' in path or '4D' in path:
        dim = 4
        feature_names = xyz_feature + rotation_feature
    elif 'endocytosis' in path.lower() or '5D' in path:
        dim = 5
        feature_names = xyz_feature + rotation_feature

    return dim, feature_names
