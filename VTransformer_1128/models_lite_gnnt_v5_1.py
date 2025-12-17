# -*- coding: utf-8 -*-
"""
LiteTransformer V5.1: Transformer 单流（去除 GNN）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LiteTransformerStream(nn.Module):
    """
    轻量 Transformer 流 (ALBERT式权重共享)。

    输入: 滤波后原始EEG (B, 1, C, T)
    输出: (B, embed_dim)
    """

    def __init__(self, n_channels=22, signal_length=1000, embed_dim=16, t_heads=2, t_layers=2, t_dropout=0.5):
        super().__init__()

        self.embed_dim = embed_dim

        # === 特征提取 ===
        self.spatial_conv = nn.Conv1d(
            n_channels, n_channels,
            kernel_size=64,
            padding=32,
            groups=n_channels,
            bias=False,
        )
        self.channel_mixer = nn.Conv1d(
            n_channels, embed_dim,
            kernel_size=1,
            bias=False,
        )
        self.norm1 = nn.BatchNorm1d(embed_dim)

        # === 时间降采样 ===
        self.pool1 = nn.AvgPool1d(kernel_size=4, stride=4)

        # === Patch 切片 ===
        self.patch_embed_depthwise = nn.Conv1d(
            embed_dim, embed_dim,
            kernel_size=10,
            stride=5,
            groups=embed_dim,
            bias=False,
        )
        self.patch_embed_pointwise = nn.Conv1d(
            embed_dim, embed_dim,
            kernel_size=1,
            bias=False,
        )
        self.norm2 = nn.BatchNorm1d(embed_dim)

        # === 卷积式位置编码 ===
        self.pos_enc = nn.Conv1d(
            embed_dim, embed_dim,
            kernel_size=7,
            padding=3,
            groups=embed_dim,
            bias=False,
        )
        self.norm3 = nn.BatchNorm1d(embed_dim)

        # === Transformer（权重共享）===
        self.shared_encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=t_heads,
            dim_feedforward=embed_dim * 2,
            dropout=t_dropout,
            activation='relu',
            batch_first=True,
        )
        self.num_shared_layers = t_layers

        self.dropout = nn.Dropout(t_dropout)

    def forward(self, x):
        """
        Args:
            x: (B, 1, C, T) 滤波后原始EEG
        Returns:
            temporal_features: (B, embed_dim)
        """
        x = x.squeeze(1)  # (B, C, T)

        x = self.spatial_conv(x)
        x = self.channel_mixer(x)
        x = self.norm1(x)
        x = F.elu(x)
        x = self.dropout(x)

        x = self.pool1(x)

        x = self.patch_embed_depthwise(x)
        x = self.patch_embed_pointwise(x)
        x = self.norm2(x)
        x = F.elu(x)
        x = self.dropout(x)

        x_pos = self.pos_enc(x)
        x = x + x_pos
        x = self.norm3(x)
        x = F.elu(x)

        x = x.permute(0, 2, 1)  # (B, T, C)

        temporal_features = x
        for _ in range(self.num_shared_layers):
            temporal_features = self.shared_encoder_layer(temporal_features)

        temporal_features = temporal_features.mean(dim=1)  # (B, embed_dim)
        return temporal_features


class LiteGNNT_V3_6(nn.Module):
    """兼容旧入口命名的 Transformer 单流包装器。"""

    def __init__(self, n_channels=22, n_classes=4,
                 signal_length=1000, t_dim=16, t_heads=2, t_layers=2,
                 dropout=0.25, **kwargs):
        super().__init__()
        self.temporal_stream = LiteTransformerStream(
            n_channels=n_channels,
            signal_length=signal_length,
            embed_dim=t_dim,
            t_heads=t_heads,
            t_layers=t_layers,
            t_dropout=dropout,
        )
        self.classifier = nn.Linear(t_dim, n_classes)

    def forward(self, X_raw, *args, **kwargs):
        temporal_feat = self.temporal_stream(X_raw)
        logits = self.classifier(temporal_feat)
        return logits

    # 兼容接口（不再使用，可返回0）
    def adj_regularization(self, lambda_reg):
        return 0.0

    def channel_regularization(self, lambda_reg):
        return 0.0
