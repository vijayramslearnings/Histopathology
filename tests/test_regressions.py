"""Regression checks for data safety, model contracts, and checkpoints."""

from __future__ import annotations

import pandas as pd
import pytest
import torch

from backend.dataset.dataset import build_label_maps
from backend.dataset.splitting import make_patient_splits
from backend.models.backbone import ResNetBaseline
from backend.models.multi_scale_model import MultiScaleAttentionNetwork
from backend.prediction.inference import load_checkpoint
from backend.training.losses import MultiTaskLoss


def _metadata_with_splits() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "image_path": ["a.png", "b.png", "c.png"],
            "patient_id": ["patient-a", "patient-b", "patient-c"],
            "slide_id": ["slide-a", "slide-b", "slide-c"],
            "cancer_type": ["BRCA", "LUAD", "GBM"],
            "subtype": ["", "", ""],
            "split": ["train", "val", "test"],
        }
    )


def test_supplied_splits_reject_patient_overlap() -> None:
    metadata = _metadata_with_splits()
    metadata.loc[2, "patient_id"] = "patient-a"

    with pytest.raises(ValueError, match="Patient leakage detected"):
        make_patient_splits(metadata)


def test_missing_subtypes_disable_subtype_supervision() -> None:
    metadata = _metadata_with_splits().drop(columns="split")
    labels = build_label_maps(metadata)

    assert labels.subtype_to_index == {}
    assert labels.has_subtypes is False

    model = ResNetBaseline(
        cancer_classes=len(labels.cancer_to_index),
        subtype_classes=len(labels.subtype_to_index),
        pretrained=False,
        hidden_dim=8,
    ).eval()
    inputs = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        outputs = model(inputs)

    assert outputs["cancer_logits"].shape == (2, 3)
    assert outputs["subtype_logits"] is None

    losses = MultiTaskLoss()(
        outputs["cancer_logits"],
        torch.tensor([0, 1]),
        outputs["subtype_logits"],
        torch.full((2,), -1),
    )
    assert torch.isfinite(losses["total"])
    assert losses["subtype"].item() == 0.0


def test_baseline_returns_cancer_and_optional_subtype_logits() -> None:
    model = ResNetBaseline(
        cancer_classes=3,
        subtype_classes=2,
        pretrained=False,
        hidden_dim=8,
    ).eval()

    with torch.no_grad():
        outputs = model(torch.randn(2, 3, 64, 64))

    assert outputs["cancer_logits"].shape == (2, 3)
    assert outputs["subtype_logits"].shape == (2, 2)
    assert outputs["features"].shape == (2, 2048)


def test_multiscale_accepts_tensor_and_scale_mapping_inputs() -> None:
    model = MultiScaleAttentionNetwork(
        cancer_classes=3,
        subtype_classes=2,
        pretrained=False,
        fusion_dim=32,
        hidden_dim=8,
        shared_backbone=True,
        patch_grid=1,
    ).eval()
    inputs = torch.randn(1, 3, 3, 64, 64)

    with torch.no_grad():
        tensor_outputs = model(inputs)
        mapping_outputs = model(
            {
                scale: inputs[:, index]
                for index, scale in enumerate(model.scale_names)
            }
        )

    for outputs in (tensor_outputs, mapping_outputs):
        assert outputs["cancer_logits"].shape == (1, 3)
        assert outputs["subtype_logits"].shape == (1, 2)
        assert outputs["scale_weights"].shape == (1, 3)
        assert outputs["patch_weights"].shape == (1, 3)
        assert outputs["mil_weights"].shape == (1, 3)


def test_checkpoint_reload_preserves_class_mappings(tmp_path) -> None:
    model = ResNetBaseline(
        cancer_classes=2,
        subtype_classes=2,
        pretrained=False,
        hidden_dim=8,
        dropout=0.0,
    )
    checkpoint = {
        "model_state": model.state_dict(),
        "model_config": {
            "name": "baseline",
            "hidden_dim": 8,
            "dropout": 0.0,
        },
        "cancer_classes": ["LUAD", "BRCA"],
        "subtype_classes": ["subtype_b", "subtype_a"],
    }
    checkpoint_path = tmp_path / "model.pt"
    torch.save(checkpoint, checkpoint_path)

    reloaded_model, reloaded_checkpoint = load_checkpoint(checkpoint_path)

    assert reloaded_checkpoint["cancer_classes"] == ["LUAD", "BRCA"]
    assert reloaded_checkpoint["subtype_classes"] == ["subtype_b", "subtype_a"]
    assert reloaded_model.cancer_head[-1].out_features == 2
    assert reloaded_model.subtype_head is not None
    assert reloaded_model.subtype_head[-1].out_features == 2
    for name, parameter in model.state_dict().items():
        assert torch.equal(parameter, reloaded_model.state_dict()[name])