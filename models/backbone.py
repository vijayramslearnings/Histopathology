"""ResNet feature extractors used by both research models."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn
from torchvision.models import ResNet50_Weights, resnet50


class ResNet50Backbone(nn.Module):
    """ResNet-50 without its classification head.

    The convolutional feature map is kept available for Grad-CAM.  Setting
    ``pretrained=True`` downloads the official ImageNet weights through
    torchvision when they are not already cached.
    """

    output_dim = 2048

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        network = resnet50(weights=weights)
        self.model = nn.Sequential(*list(network.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward_features(self, x: Tensor) -> Tensor:
        return self.model(x)

    def forward(self, x: Tensor) -> Tensor:
        features = self.forward_features(x)
        return torch.flatten(self.pool(features), 1)

    def feature_tokens(self, x: Tensor, grid_size: int = 4) -> Tensor:
        """Return spatial tokens with shape ``[batch, tokens, channels]``."""
        features = self.forward_features(x)
        pooled = nn.functional.adaptive_avg_pool2d(features, (grid_size, grid_size))
        return pooled.flatten(2).transpose(1, 2)


def make_classifier(
    input_dim: int,
    num_classes: int,
    hidden_dim: int = 512,
    dropout: float = 0.3,
) -> nn.Sequential:
    """Build the requested Dropout -> Linear -> ReLU -> Dropout -> Linear head."""
    return nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, num_classes),
    )


class HierarchicalSubtypeHeads(nn.Module):
    """Cancer-specific subtype classification heads."""

    def __init__(
        self,
        input_dim: int,
        subtype_specs: dict[str, Any],
        hidden_dim: int = 512,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        heads = {}
        for cancer, num_classes in subtype_specs.items():
            n_cls = len(num_classes) if isinstance(num_classes, dict) else int(num_classes)
            if n_cls > 0:
                heads[cancer] = make_classifier(input_dim, n_cls, hidden_dim, dropout)
        self.heads = nn.ModuleDict(heads)

    def forward(self, features: Tensor, target_cancer: Optional[str] = None) -> dict[str, Tensor]:
        if target_cancer and target_cancer in self.heads:
            return {target_cancer: self.heads[target_cancer](features)}
        return {cancer: head(features) for cancer, head in self.heads.items()}


class ResNetBaseline(nn.Module):
    """ImageNet ResNet-50 baseline with cancer and hierarchical subtype heads."""

    def __init__(
        self,
        cancer_classes: int = 3,
        subtype_classes: int | dict[str, int] = 0,
        pretrained: bool = True,
        hidden_dim: int = 512,
        dropout: float = 0.3,
        subtype_specs: Optional[dict[str, int]] = None,
    ) -> None:
        super().__init__()
        self.backbone = ResNet50Backbone(pretrained=pretrained)
        self.cancer_head = make_classifier(
            self.backbone.output_dim, cancer_classes, hidden_dim, dropout
        )
        if isinstance(subtype_classes, dict):
            specs = subtype_classes
        elif subtype_specs:
            specs = subtype_specs
        elif isinstance(subtype_classes, int) and subtype_classes > 0:
            specs = {"BRCA": 3, "LUAD": 2, "GBM": 2}
        else:
            specs = {}

        self.subtype_heads = (
            HierarchicalSubtypeHeads(self.backbone.output_dim, specs, hidden_dim, dropout)
            if specs
            else None
        )
        self.subtype_head = (
            make_classifier(self.backbone.output_dim, subtype_classes, hidden_dim, dropout)
            if isinstance(subtype_classes, int) and subtype_classes > 0
            else None
        )

    def forward(self, x: Tensor, target_cancer: Optional[str] = None) -> dict[str, Any]:
        features = self.backbone(x)
        cancer_logits = self.cancer_head(features)
        subtype_dict = (
            self.subtype_heads(features, target_cancer)
            if self.subtype_heads is not None
            else {}
        )
        subtype_logits = (
            self.subtype_head(features) if self.subtype_head is not None else None
        )
        return {
            "cancer_logits": cancer_logits,
            "subtype_logits": subtype_logits,
            "subtype_dict": subtype_dict,
            "features": features,
        }

    def target_layer(self) -> nn.Module:
        return self.backbone.model[-1][-1].conv3