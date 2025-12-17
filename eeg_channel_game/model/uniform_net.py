from __future__ import annotations

import torch
from torch import nn


class UniformPolicyValueNet(nn.Module):
    """
    A minimal policy/value network that produces a uniform prior over valid actions
    and a constant value=0.

    Useful as an ablation baseline for MCTS ("planning only" without learned priors/values).
    """

    def __init__(self, *, n_tokens: int = 24, n_actions: int = 23):
        super().__init__()
        self.n_tokens = int(n_tokens)
        self.n_actions = int(n_actions)

    def forward(self, tokens: torch.Tensor, action_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if tokens.ndim != 3 or tokens.shape[1] != self.n_tokens:
            raise ValueError(f"Expected tokens [B,{self.n_tokens},D], got {tuple(tokens.shape)}")
        bsz = int(tokens.shape[0])
        logits = torch.zeros((bsz, self.n_actions), device=tokens.device, dtype=tokens.dtype)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, -1e9)
        value = torch.zeros((bsz,), device=tokens.device, dtype=tokens.dtype)
        return logits, value

