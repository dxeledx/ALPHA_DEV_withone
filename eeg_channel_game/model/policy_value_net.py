from __future__ import annotations

import torch
from torch import nn


class PolicyValueNet(nn.Module):
    def __init__(
        self,
        *,
        d_in: int = 64,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        n_actions: int = 23,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_tokens = 24
        self.in_proj = nn.Linear(d_in, d_model)
        self.pos_emb = nn.Parameter(torch.zeros((1, self.n_tokens, d_model), dtype=torch.float32))
        # 3 token types: 0=CLS, 1=channel, 2=CTX
        self.type_emb = nn.Embedding(3, d_model)
        self.emb_drop = nn.Dropout(dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.tr = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.policy_head = nn.Linear(d_model, n_actions)
        self.value_head = nn.Linear(d_model, 1)

    def forward(self, tokens: torch.Tensor, action_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        tokens: float32 [B, 24, d_in]
        action_mask: bool [B, 23] or None
        returns: logits [B, 23], value [B]
        """
        if tokens.ndim != 3 or tokens.shape[1] != self.n_tokens:
            raise ValueError(f"Expected tokens [B,{self.n_tokens},D], got {tuple(tokens.shape)}")
        x = self.in_proj(tokens)

        # positional + token-type embeddings
        type_ids = torch.empty((self.n_tokens,), device=x.device, dtype=torch.long)
        type_ids[0] = 0
        type_ids[1:23] = 1
        type_ids[23] = 2
        x = x + self.pos_emb[:, : self.n_tokens, :].to(device=x.device, dtype=x.dtype)
        x = x + self.type_emb(type_ids)[None, :, :].to(dtype=x.dtype)
        x = self.emb_drop(x)

        h = self.tr(x)
        cls = self.norm(h[:, 0, :])

        logits = self.policy_head(cls)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, -1e9)
        value = self.value_head(cls).squeeze(-1)
        return logits, value
