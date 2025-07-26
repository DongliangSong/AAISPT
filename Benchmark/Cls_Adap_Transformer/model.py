# -*- coding: utf-8 -*-
# @Time    : 2025/7/25 17:39
# @Author  : Dongliang

import numpy as np
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


# Define a multitasking Transformer model
class MultiTaskTransformer(nn.Module):
    def __init__(self, feature_dims, num_classes_list, d_model, num_layers, num_heads, hidden_dim, max_seq_len=1000):
        """
        Initializes a multitask Transformer model for trajectory classification.

        :param feature_dims: List of input feature dimensions for each task (e.g., [32, 46]).
        :param num_classes_list: List of number of classes for each task (e.g., [3, 5]).
        :param d_model: Dimension of the Transformer model.
        :param num_layers: Number of Transformer encoder layers.
        :param num_heads: Number of attention heads.
        :param hidden_dim: Dimension of the feedforward network in Transformer.
        :param max_seq_len: Maximum sequence length for positional encoding.
        """

        super().__init__()
        self.num_tasks = len(feature_dims)
        self.d_model = d_model

        # Projection layers for each task
        self.projs = nn.ModuleList([nn.Linear(dim, d_model) for dim in feature_dims])

        # Transformer encoder
        self.encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=hidden_dim)
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        # Classification heads for each task
        self.fcs = nn.ModuleList([nn.Linear(d_model, num_classes) for num_classes in num_classes_list])
        self.logits = nn.Parameter(torch.zeros(self.num_tasks))

        # 正弦位置编码
        self.positional_encoding = self.create_positional_encoding(max_seq_len, d_model)

    def create_positional_encoding(self, max_seq_len, d_model):
        """
        Generates positional encoding for the Transformer model.

        :param max_seq_len: Maximum sequence length.
        :param d_model:
        """
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_seq_len, d_model)
        return pe

    def get_weights(self):
        """
        Computes task weights using softmax on logits.
        """
        return torch.softmax(self.logits, dim=0)

    def forward(self, x, mask, task_id, return_attention=False):
        """
        Performs a forward pass through the multitask Transformer.

        :param x: Input tensor of shape (batch_size, seq_len, feature_dim).
        :param mask: Boolean mask of shape (batch_size, seq_len), True for padded positions.
        :param task_id: Integer index of the task to process.
        :param return_attention: If True, returns attention weights from the last encoder layer.
        :return:
            - logits: Classification logits of shape (batch_size, num_classes).
            - attn_weights (optional): Attention weights of shape (batch_size, num_heads, seq_len, seq_len) if return_attention is True.
        """
        batch_size, seq_len, _ = x.size()
        # Project input to d_model dimension
        x = self.projs[task_id](x)  # (batch, seq_len, d_model)

        # Add positional encoding
        positional_encoding = self.positional_encoding[:, :seq_len, :].to(x.device)
        x = x + positional_encoding  # (batch, seq_len, d_model)
        x = x.transpose(0, 1)

        # Process through Transformer encoder
        if return_attention:
            # Process all layers except the last one
            for layer in self.transformer_encoder.layers[:-1]:
                x = layer(x, src_key_padding_mask=mask)

            # Last layer: extract attention weights
            attn_output, attn_weights = self.transformer_encoder.layers[-1].self_attn(
                x, x, x, key_padding_mask=mask, need_weights=True, average_attn_weights=False
            )
            x = self.transformer_encoder.layers[-1](x, src_key_padding_mask=mask)
        else:
            x = self.transformer_encoder(x, src_key_padding_mask=mask)

        x = x.transpose(0, 1)
        mask_expanded = mask.unsqueeze(-1)
        valid_mask = (~mask_expanded).float()
        x = (x * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1e-9)

        # Classification
        logits = self.fcs[task_id](x)  # [batch_size, num_classes]

        if return_attention:
            return logits, attn_weights
        return logits