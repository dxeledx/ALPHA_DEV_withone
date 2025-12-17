from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SubsetRobustVTransformer(nn.Module):
    """
    A lightweight Transformer classifier adapted from VTransformer_1128 (LiteTransformer V5.1),
    made robust for EEG channel subsets via:
      - explicit channel mask gating (selected vs missing channels),
      - optional channel-dropout during training,
      - GroupNorm (per-sample) instead of BatchNorm (more stable under subset shifts).

    Input:
      - x: [B, C, T] (or [B, 1, C, T]) where C == n_channels (default 22)
    Output:
      - logits: [B, n_outputs]
    """

    def __init__(
        self,
        *,
        n_channels: int = 22,
        n_outputs: int = 4,
        embed_dim: int = 16,
        t_heads: int = 2,
        t_layers: int = 2,
        dropout: float = 0.25,
        channel_drop_p: float = 0.0,
        sel_mask: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.n_channels = int(n_channels)
        self.n_outputs = int(n_outputs)
        self.embed_dim = int(embed_dim)
        self.channel_drop_p = float(channel_drop_p)
        if not (0.0 <= self.channel_drop_p < 1.0):
            raise ValueError("channel_drop_p must be in [0,1)")

        if sel_mask is None:
            sel_mask_t = torch.ones((self.n_channels,), dtype=torch.float32)
        else:
            sel_mask_t = torch.as_tensor(sel_mask, dtype=torch.float32).flatten()
            if int(sel_mask_t.numel()) != int(self.n_channels):
                raise ValueError(f"sel_mask must have length {self.n_channels}, got {int(sel_mask_t.numel())}")
            sel_mask_t = (sel_mask_t > 0.0).to(dtype=torch.float32)
        self.register_buffer("_base_mask", sel_mask_t.view(1, self.n_channels, 1), persistent=False)

        # === Per-channel temporal conv (depthwise over channel axis) ===
        self.spatial_conv = nn.Conv1d(
            self.n_channels,
            self.n_channels,
            kernel_size=64,
            padding=32,
            groups=self.n_channels,
            bias=False,
        )

        # === Channel mixing (1x1 conv) ===
        self.channel_mixer = nn.Conv1d(
            self.n_channels,
            self.embed_dim,
            kernel_size=1,
            bias=False,
        )
        self.norm1 = nn.GroupNorm(num_groups=self.embed_dim, num_channels=self.embed_dim)

        # === Temporal downsampling ===
        self.pool1 = nn.AvgPool1d(kernel_size=4, stride=4)

        # === Patch embed ===
        self.patch_embed_depthwise = nn.Conv1d(
            self.embed_dim,
            self.embed_dim,
            kernel_size=10,
            stride=5,
            groups=self.embed_dim,
            bias=False,
        )
        self.patch_embed_pointwise = nn.Conv1d(
            self.embed_dim,
            self.embed_dim,
            kernel_size=1,
            bias=False,
        )
        self.norm2 = nn.GroupNorm(num_groups=self.embed_dim, num_channels=self.embed_dim)

        # === Convolutional positional encoding ===
        self.pos_enc = nn.Conv1d(
            self.embed_dim,
            self.embed_dim,
            kernel_size=7,
            padding=3,
            groups=self.embed_dim,
            bias=False,
        )
        self.norm3 = nn.GroupNorm(num_groups=self.embed_dim, num_channels=self.embed_dim)

        # === Transformer (ALBERT-style weight sharing) ===
        self.shared_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=int(t_heads),
            dim_feedforward=self.embed_dim * 2,
            dropout=float(dropout),
            activation="relu",
            batch_first=True,
        )
        self.num_shared_layers = int(t_layers)

        self.dropout = nn.Dropout(float(dropout))
        self.classifier = nn.Linear(self.embed_dim, self.n_outputs)

    def _sample_effective_mask(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns an effective mask in {0,1} with shape [B, C, 1]:
          - base subset mask is always applied
          - optional channel-dropout applied on top during training
        Also applies a sqrt scaling to keep feature energy stable under dropout.
        """
        base_mask = self._base_mask.to(dtype=x.dtype, device=x.device)  # [1,C,1]
        if (not self.training) or self.channel_drop_p <= 0.0:
            return base_mask.expand(x.shape[0], -1, -1)

        bsz, c, _t = x.shape
        keep = (torch.rand((bsz, c), device=x.device) > float(self.channel_drop_p)).to(dtype=x.dtype).view(bsz, c, 1)
        eff = base_mask.expand(bsz, -1, -1) * keep

        # ensure at least one active channel per sample
        active = eff.sum(dim=1, keepdim=True)  # [B,1,1]
        dead = (active[:, 0, 0] <= 0).nonzero(as_tuple=False).view(-1)
        if int(dead.numel()) > 0:
            allowed = (base_mask[0, :, 0] > 0).to(dtype=torch.float32)
            if float(allowed.sum().item()) > 0:
                for i in dead.tolist():
                    ch = int(torch.multinomial(allowed, 1).item())
                    eff[i, ch, 0] = 1.0
                active = eff.sum(dim=1, keepdim=True)

        # inverted-dropout style scaling (keep expected energy close to base subset)
        base_count = base_mask.sum(dim=1, keepdim=True).clamp(min=1.0)  # [1,1,1]
        scale = torch.sqrt(base_count / active.clamp(min=1.0))
        return eff * scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            # accept [B, 1, C, T] for compatibility with some pipelines
            if int(x.shape[1]) != 1:
                raise ValueError(f"Expected x shape [B,1,C,T], got {tuple(x.shape)}")
            x = x.squeeze(1)

        if x.ndim != 3:
            raise ValueError(f"Expected x shape [B,C,T], got {tuple(x.shape)}")
        if int(x.shape[1]) != int(self.n_channels):
            raise ValueError(f"Expected C={self.n_channels}, got {int(x.shape[1])}")

        # subset mask gating (and optional channel-dropout)
        eff_mask = self._sample_effective_mask(x)
        x = x * eff_mask

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

        x = x + self.pos_enc(x)
        x = self.norm3(x)
        x = F.elu(x)

        # [B, embed_dim, T'] -> [B, T', embed_dim]
        x = x.permute(0, 2, 1)

        for _ in range(int(self.num_shared_layers)):
            x = self.shared_encoder_layer(x)

        feat = x.mean(dim=1)
        feat = self.dropout(feat)
        return self.classifier(feat)
