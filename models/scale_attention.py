"""Attention over the available image scales."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ScaleAttention(nn.Module):
    """Learn a normalized importance weight for each magnification branch."""

    def __init__(self, feature_dim: int = 512) -> None:
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.Tanh(),
            nn.Linear(feature_dim // 2, 1),
        )

    def forward(self, scale_features: Tensor) -> tuple[Tensor, Tensor]:
        """Return weighted context and ``[batch, scales]`` attention weights."""
        scores = self.scorer(scale_features).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(scale_features * weights.unsqueeze(-1), dim=1)
        return context, weights