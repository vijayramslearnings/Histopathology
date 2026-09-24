"""Multiple-instance learning pooling."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class MILPooling(nn.Module):
    """Gated attention pooling for image/patch instances."""

    def __init__(self, feature_dim: int = 512) -> None:
        super().__init__()
        hidden = max(32, feature_dim // 2)
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, instances: Tensor) -> tuple[Tensor, Tensor]:
        scores = self.attention(instances).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(instances * weights.unsqueeze(-1), dim=1)
        return pooled, weights