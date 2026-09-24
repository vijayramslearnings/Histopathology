"""Cross-scale feature fusion."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class CrossScaleFusion(nn.Module):
    """Project each scale into a shared space and exchange scale context."""

    def __init__(self, input_dim: int = 2048, fusion_dim: int = 512) -> None:
        super().__init__()
        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, fusion_dim),
                    nn.LayerNorm(fusion_dim),
                    nn.GELU(),
                )
                for _ in range(3)
            ]
        )
        self.context = nn.MultiheadAttention(
            embed_dim=fusion_dim, num_heads=8, batch_first=True, dropout=0.1
        )
        self.norm = nn.LayerNorm(fusion_dim)

    def forward(self, scale_tokens: list[Tensor]) -> Tensor:
        if len(scale_tokens) != len(self.projections):
            raise ValueError(
                f"Expected {len(self.projections)} scales, received {len(scale_tokens)}."
            )
        projected = [
            projection(tokens)
            for projection, tokens in zip(self.projections, scale_tokens)
        ]
        tokens = torch.cat(projected, dim=1)
        attended, _ = self.context(tokens, tokens, tokens, need_weights=False)
        return self.norm(tokens + attended)