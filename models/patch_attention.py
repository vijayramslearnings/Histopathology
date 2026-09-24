"""Spatial patch attention over the fused scale tokens."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class PatchAttention(nn.Module):
    """Gate local tokens before MIL pooling."""

    def __init__(self, feature_dim: int = 512) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.Tanh(),
            nn.Linear(feature_dim // 2, 1),
        )

    def forward(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        scores = self.gate(tokens).squeeze(-1)
        weights = torch.sigmoid(scores)
        return tokens * (1.0 + weights.unsqueeze(-1)), weights