"""Proposed multi-scale attention network."""

from __future__ import annotations

from typing import Any, Optional

import torch
from torch import Tensor, nn

from .backbone import HierarchicalSubtypeHeads, ResNet50Backbone, make_classifier
from .cross_scale_fusion import CrossScaleFusion
from .mil import MILPooling
from .patch_attention import PatchAttention
from .scale_attention import ScaleAttention


class MultiScaleAttentionNetwork(nn.Module):
    """Three-branch multi-scale network with optional ablation switches.

    When the source dataset does not contain real WSI magnification levels,
    the three input branches are context-scale approximations. The model does
    not label them as true 5x/10x/20x magnifications.
    """

    scale_names = ("5x", "10x", "20x")

    def __init__(
        self,
        cancer_classes: int = 3,
        subtype_classes: int | dict[str, int] = 0,
        pretrained: bool = True,
        fusion_dim: int = 512,
        hidden_dim: int = 512,
        dropout: float = 0.3,
        use_scale_attention: bool = True,
        use_patch_attention: bool = True,
        use_mil: bool = True,
        shared_backbone: bool = False,
        patch_grid: int = 4,
        subtype_specs: Optional[dict[str, int]] = None,
    ) -> None:
        super().__init__()
        self.patch_grid = patch_grid
        self.use_scale_attention = use_scale_attention
        self.use_patch_attention = use_patch_attention
        self.use_mil = use_mil
        self.shared_backbone = shared_backbone

        if shared_backbone:
            self.shared = ResNet50Backbone(pretrained=pretrained)
            self.backbones = None
        else:
            self.shared = None
            self.backbones = nn.ModuleDict(
                {
                    scale: ResNet50Backbone(pretrained=pretrained)
                    for scale in self.scale_names
                }
            )
        self.fusion = CrossScaleFusion(2048, fusion_dim)
        self.scale_attention = ScaleAttention(fusion_dim)
        self.patch_attention = PatchAttention(fusion_dim)
        self.mil_pooling = MILPooling(fusion_dim)
        self.cancer_head = make_classifier(fusion_dim, cancer_classes, hidden_dim, dropout)

        if isinstance(subtype_classes, dict):
            specs = subtype_classes
        elif subtype_specs:
            specs = subtype_specs
        elif isinstance(subtype_classes, int) and subtype_classes > 0:
            specs = {"BRCA": 3, "LUAD": 2, "GBM": 2}
        else:
            specs = {}

        self.subtype_heads = (
            HierarchicalSubtypeHeads(fusion_dim, specs, hidden_dim, dropout)
            if specs
            else None
        )
        self.subtype_head = (
            make_classifier(fusion_dim, subtype_classes, hidden_dim, dropout)
            if isinstance(subtype_classes, int) and subtype_classes > 0
            else None
        )

    def _backbone(self, scale: str) -> ResNet50Backbone:
        if self.shared is not None:
            return self.shared
        assert self.backbones is not None
        return self.backbones[scale]

    def forward(
        self,
        x: Tensor | dict[str, Tensor],
        target_cancer: Optional[str] = None,
    ) -> dict[str, Any]:
        if isinstance(x, dict):
            x = torch.stack([x[scale] for scale in self.scale_names], dim=1)
        if x.ndim != 5 or x.shape[1] != 3:
            raise ValueError("Multi-scale input must have shape [batch, 3, channels, height, width].")

        scale_tokens = [
            self._backbone(scale).feature_tokens(x[:, index], self.patch_grid)
            for index, scale in enumerate(self.scale_names)
        ]
        fused_tokens = self.fusion(scale_tokens)
        scale_features = fused_tokens.reshape(
            x.shape[0], len(self.scale_names), self.patch_grid * self.patch_grid, -1
        ).mean(dim=2)

        if self.use_scale_attention:
            _, scale_weights = self.scale_attention(scale_features)
        else:
            scale_weights = torch.full(
                (x.shape[0], 3), 1.0 / 3.0, device=x.device, dtype=x.dtype
            )

        tokens_per_scale = self.patch_grid * self.patch_grid
        token_weights = scale_weights.repeat_interleave(tokens_per_scale, dim=1)
        fused_tokens = fused_tokens * token_weights.unsqueeze(-1)

        if self.use_patch_attention:
            refined_tokens, patch_weights = self.patch_attention(fused_tokens)
        else:
            refined_tokens = fused_tokens
            patch_weights = torch.ones(
                fused_tokens.shape[:2], device=x.device, dtype=x.dtype
            )

        if self.use_mil:
            pooled, mil_weights = self.mil_pooling(refined_tokens)
        else:
            pooled = refined_tokens.mean(dim=1)
            mil_weights = torch.full(
                refined_tokens.shape[:2],
                1.0 / refined_tokens.shape[1],
                device=x.device,
                dtype=x.dtype,
            )

        subtype_dict = (
            self.subtype_heads(pooled, target_cancer)
            if self.subtype_heads is not None
            else {}
        )
        subtype_logits = (
            self.subtype_head(pooled) if self.subtype_head is not None else None
        )

        return {
            "cancer_logits": self.cancer_head(pooled),
            "subtype_logits": subtype_logits,
            "subtype_dict": subtype_dict,
            "features": pooled,
            "scale_weights": scale_weights,
            "patch_weights": patch_weights,
            "mil_weights": mil_weights,
            "feature_maps": scale_tokens,
        }

    def target_layer(self, scale: str = "20x") -> nn.Module:
        return self._backbone(scale).model[-1][-1].conv3