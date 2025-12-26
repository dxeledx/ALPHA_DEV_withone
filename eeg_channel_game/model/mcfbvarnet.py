from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


def _pick_gn_groups(n_channels: int, preferred: int) -> int:
    n_channels = int(n_channels)
    preferred = int(preferred)
    if n_channels % preferred == 0:
        return preferred
    for g in range(min(preferred, n_channels), 0, -1):
        if n_channels % g == 0:
            return g
    return 1


@dataclass(frozen=True)
class MCFBVarNetConfig:
    n_chans: int = 22
    n_bands: int = 4
    n_classes: int = 4

    # temporal stem
    F: int = 16
    k0: int = 25

    # multiscale temporal depthwise
    k1: int = 15
    d1: int = 1
    d2: int = 2
    F2: int = 16

    # spatial depthwise
    D: int = 2
    pool_t: int = 8

    # regularization
    dropout_conv: float = 0.25
    dropout_fc: float = 0.5

    # mask conditioning (FiLM)
    mask_hidden: int = 64
    use_film: bool = True

    # head
    fc_hidden: int = 256


class _FBBranch(nn.Module):
    def __init__(self, cfg: MCFBVarNetConfig):
        super().__init__()
        pad0 = int(cfg.k0 // 2)
        pad1 = int((cfg.k1 // 2) * cfg.d1)
        pad2 = int((cfg.k1 // 2) * cfg.d2)

        g1 = _pick_gn_groups(cfg.F, preferred=max(1, cfg.F // 4))
        g2 = _pick_gn_groups(cfg.F2, preferred=max(1, cfg.F2 // 4))
        g3 = _pick_gn_groups(cfg.F2 * cfg.D, preferred=4)

        self.temporal = nn.Sequential(
            nn.Conv2d(1, cfg.F, kernel_size=(1, cfg.k0), padding=(0, pad0), bias=False),
            nn.GroupNorm(num_groups=g1, num_channels=cfg.F),
            nn.ELU(),
        )

        self.dw_t1 = nn.Conv2d(
            cfg.F,
            cfg.F,
            kernel_size=(1, cfg.k1),
            padding=(0, pad1),
            dilation=(1, cfg.d1),
            groups=cfg.F,
            bias=False,
        )
        self.dw_t2 = nn.Conv2d(
            cfg.F,
            cfg.F,
            kernel_size=(1, cfg.k1),
            padding=(0, pad2),
            dilation=(1, cfg.d2),
            groups=cfg.F,
            bias=False,
        )
        self.pw_mix = nn.Sequential(
            nn.Conv2d(2 * cfg.F, cfg.F2, kernel_size=(1, 1), bias=False),
            nn.GroupNorm(num_groups=g2, num_channels=cfg.F2),
            nn.ELU(),
        )

        # depthwise spatial filter across channel axis
        self.spatial = nn.Sequential(
            nn.Conv2d(cfg.F2, cfg.F2 * cfg.D, kernel_size=(cfg.n_chans, 1), groups=cfg.F2, bias=False),
            nn.GroupNorm(num_groups=g3, num_channels=cfg.F2 * cfg.D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, cfg.pool_t), stride=(1, cfg.pool_t)),
            nn.Dropout(p=float(cfg.dropout_conv)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        if x.ndim != 3:
            raise ValueError(f"Expected x=[B,C,T], got {tuple(x.shape)}")
        x = x.unsqueeze(1)  # [B, 1, C, T]
        x = self.temporal(x)  # [B, F, C, T]
        a = self.dw_t1(x)
        b = self.dw_t2(x)
        x = torch.cat([a, b], dim=1)  # [B, 2F, C, T]
        x = self.pw_mix(x)  # [B, F2, C, T]
        x = self.spatial(x)  # [B, F2*D, 1, T']
        x = x.squeeze(2)  # [B, F2*D, T']
        var = x.var(dim=-1, unbiased=False)
        return torch.log(var + 1e-6)  # [B, F2*D]


class MCFBVarNet(nn.Module):
    """
    Mask-conditioned FilterBank VarianceNet (MC-FBVarNet).

    Expected inputs:
      - x_fb: [B, n_bands, C, T]
      - mask: [B, C] in {0,1}
    """

    def __init__(self, cfg: MCFBVarNetConfig):
        super().__init__()
        self.cfg = cfg
        self.branches = nn.ModuleList([_FBBranch(cfg) for _ in range(int(cfg.n_bands))])

        d_branch = int(cfg.F2 * cfg.D)
        if cfg.use_film:
            self.mask_mlp = nn.Sequential(
                nn.Linear(int(cfg.n_chans), int(cfg.mask_hidden)),
                nn.ELU(),
                nn.Linear(int(cfg.mask_hidden), int(2 * cfg.n_bands * d_branch)),
            )
        else:
            self.mask_mlp = None

        self.head = nn.Sequential(
            nn.Linear(int(cfg.n_bands * d_branch), int(cfg.fc_hidden)),
            nn.ELU(),
            nn.Dropout(p=float(cfg.dropout_fc)),
            nn.Linear(int(cfg.fc_hidden), int(cfg.n_classes)),
        )

    def forward(self, x_fb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if x_fb.ndim != 4:
            raise ValueError(f"Expected x_fb=[B,BANDS,C,T], got {tuple(x_fb.shape)}")
        if mask.ndim != 2:
            raise ValueError(f"Expected mask=[B,C], got {tuple(mask.shape)}")

        bsz, n_bands, n_chans, _t = x_fb.shape
        if int(n_bands) != int(self.cfg.n_bands):
            raise ValueError(f"Expected n_bands={self.cfg.n_bands}, got {int(n_bands)}")
        if int(n_chans) != int(self.cfg.n_chans):
            raise ValueError(f"Expected n_chans={self.cfg.n_chans}, got {int(n_chans)}")

        # hard gating (unselected channels are zeroed)
        gate = mask.to(dtype=x_fb.dtype)[:, None, :, None]  # [B,1,C,1]
        x_fb = x_fb * gate

        feats = []
        for bi, branch in enumerate(self.branches):
            xb = x_fb[:, bi, :, :]  # [B, C, T]
            feats.append(branch(xb))

        f = torch.stack(feats, dim=1)  # [B, BANDS, d_branch]
        if self.mask_mlp is not None:
            film = self.mask_mlp(mask.to(dtype=x_fb.dtype))  # [B, 2*BANDS*d_branch]
            d_branch = f.shape[-1]
            film = film.view(bsz, int(self.cfg.n_bands), 2 * d_branch)
            gamma, beta = film.chunk(2, dim=-1)
            f = gamma * f + beta

        f = f.reshape(bsz, -1)
        return self.head(f)

