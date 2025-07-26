# -*- coding: utf-8 -*-
# @Time    : 2025/2/21 13:17
# @Author  : Dongliang

import os

import scipy
import torch

from AAISPT.Traj_seg.Seg_Adap_BiLSTM_BCE.main import Seg_Adap_BiLSTM_BCE

# Fixed parameters
max_dim = 5
d_model = 128
hidden_size = 128
num_layers = 3
out_dim = 1

# Create and load model
model = Seg_Adap_BiLSTM_BCE(
    max_dim=max_dim,
    d_model=d_model,
    hidden_size=hidden_size,
    num_layers=num_layers,
    out_dim=out_dim,
)

# Load model
model_path = r'..\model'
model.load_state_dict(torch.load(os.path.join(model_path, 'seg_model.pth')))

# Load dataset
path = r'..\data\5D'
save_path = os.path.join(path, 'Visualization')
if not os.path.exists(save_path):
    os.mkdir(save_path)

test = scipy.io.loadmat(os.path.join(path, 'norm_test.mat'))
testx = test['data'][0]  # 0 denotes the first trajectory.
test_X = torch.unsqueeze(torch.from_numpy(testx), dim=0)

layer_outputs = {}


# Hook function to capture the output of each layer
def save_output(name):
    def hook(module, input, output):
        layer_outputs[name] = output

    return hook


# Register hooks
model.conv_block.register_forward_hook(save_output("conv"))
model.lstm.register_forward_hook(save_output("bilstm"))
model.fc.register_forward_hook(save_output("fc"))

model.eval()
with torch.no_grad():
    prob = model(test_X)

# Visualize the output of each layer
feature_vec = {}
for layer_name, output in layer_outputs.items():
    if layer_name == "conv":
        feature_vec[layer_name] = output.squeeze(0)
    elif layer_name == "bilstm":
        feature_vec[layer_name] = output[0].squeeze(0)
    elif layer_name == "fc":
        feature_vec[layer_name] = output.squeeze(0)

    feature_vec['prob'] = prob

scipy.io.savemat(os.path.join(save_path, 'Feature_vector.mat'), feature_vec)

# if layer_name == "conv":
#     output = output.squeeze(0)
#     num_channels = output.shape[0]
#     fig,axes = plt.subplots(1,num_channels,figsize=(20,5))
#
#     for i in range(num_channels):
#         axes[i].plot(output[i].detach().numpy())
#         axes[i].set_title(f'Channel {i+1}')
#         axes[i].set_xlabel('Timesteps')
#         axes[i].set_ylabel('Value')
#         axes[i].grid(True)
#     plt.show()
# elif layer_name == "bilstm":
#     output = output[0].squeeze(0)
#     seq_len, num_features = output.shape
#     plt.figure(figsize=(10,5))
#     for i in range(num_features):
#         plt.plot(output[0, i].detach().numpy(),label=f"Feature {i}")
#
#     plt.title(f"BiLSTM: {layer_name}")
#     plt.legend()
#     plt.show()
