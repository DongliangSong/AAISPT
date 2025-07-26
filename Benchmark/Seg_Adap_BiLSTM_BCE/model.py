# -*- coding: utf-8 -*-
# @Time    : 2024/8/10 16:33
# @Author  : Dongliang

from torch import nn


class AdaptiveConvModule(nn.Module):
    def __init__(self, max_dim, target_dim=64):
        super(AdaptiveConvModule, self).__init__()

        # Create corresponding 1x1 convolutional layers for each input dimension.
        self.dim_conv = nn.ModuleDict({
            f'dim_{d}': nn.Conv1d(d, target_dim, kernel_size=1)
            for d in range(2, max_dim + 1)
        })

        self.target_dim = target_dim
        self.norm = nn.LayerNorm(target_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        batch_size, seq_len, input_dim = x.size()
        conv = self.dim_conv[f'dim_{input_dim}']
        x = x.transpose(1, 2)
        x = conv(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.activation(x)
        return x


class Seg_Adap_BiLSTM_BCE(nn.Module):
    def __init__(self, max_dim, d_model, hidden_size, num_layers, out_dim):
        super(Seg_Adap_BiLSTM_BCE, self).__init__()
        self.conv_block = AdaptiveConvModule(max_dim=max_dim, target_dim=d_model)
        self.lstm = nn.LSTM(d_model, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, out_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv_block(x)
        lstm_out, _ = self.lstm(x)
        fc_out = self.fc(lstm_out)
        outputs = self.sigmoid(fc_out)
        return outputs
