from __future__ import annotations

import torch

from backend.models.backbone import ResNetBaseline
from backend.models.multi_scale_model import MultiScaleAttentionNetwork


torch.set_num_threads(1)


def test_baseline_contract_with_optional_subtype_head() -> None:
    model = ResNetBaseline(
        cancer_classes=3,
        subtype_classes=2,
        pretrained=False,
        hidden_dim=16,
        dropout=0.0,
    ).eval()
    with torch.no_grad():
        output = model(torch.randn(1, 3, 64, 64))
    assert output["cancer_logits"].shape == (1, 3)
    assert output["subtype_logits"].shape == (1, 2)
    assert output["features"].shape == (1, 2048)


def test_multiscale_contract_and_attention_shapes() -> None:
    model = MultiScaleAttentionNetwork(
        cancer_classes=3,
        subtype_classes=0,
        pretrained=False,
        fusion_dim=32,
        hidden_dim=16,
        dropout=0.0,
        shared_backbone=True,
        patch_grid=1,
    ).eval()
    with torch.no_grad():
        output = model(torch.randn(1, 3, 3, 64, 64))
    assert output["cancer_logits"].shape == (1, 3)
    assert output["subtype_logits"] is None
    assert output["scale_weights"].shape == (1, 3)
    assert output["patch_weights"].shape == (1, 3)
    assert output["mil_weights"].shape == (1, 3)
    assert torch.allclose(output["scale_weights"].sum(dim=1), torch.ones(1))
    assert torch.allclose(output["mil_weights"].sum(dim=1), torch.ones(1))