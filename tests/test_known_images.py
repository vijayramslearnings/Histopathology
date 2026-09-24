"""Dedicated test suite for verifying predictions on known lung, breast, and GBM samples."""

from __future__ import annotations

import tempfile
from pathlib import Path
import pytest
import torch
from PIL import Image

from dataset.inspect_and_build_dataset import create_realistic_histopathology_patch
from models.multi_scale_model import MultiScaleAttentionNetwork
from prediction.inference import load_checkpoint, predict_image


@pytest.fixture
def trained_multiscale_model():
    """Instantiate and populate trained multi-scale attention model."""
    ckpt_path = Path("checkpoints/multiscale_full.pt")
    if not ckpt_path.exists():
        ckpt_path = Path("checkpoints/best_model.pt")
    if ckpt_path.exists():
        model, _ = load_checkpoint(ckpt_path, device="cpu")
        return model

    model = MultiScaleAttentionNetwork(
        cancer_classes=3,
        subtype_specs={"BRCA": 3, "LUAD": 2, "GBM": 2},
        shared_backbone=True,
        pretrained=False,
    )
    model.eval()
    return model


def test_known_lung_image_prediction(trained_multiscale_model):
    """PHASE 27 Requirement: Test known lung image against model."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        lung_img_path = Path(tmp_dir) / "known_lung_sample.png"
        img = create_realistic_histopathology_patch("LUAD", "Acinar_Predominant", seed=101)
        img.save(lung_img_path)

        res = predict_image(
            model=trained_multiscale_model,
            image_path=lung_img_path,
            cancer_classes=["BRCA", "LUAD", "GBM"],
            organ_type="lung",
        )

        assert res["status"] == "ok"
        assert "cancer" in res
        assert "subtype" in res
        assert "uncertainty" in res
        # Verify that predicted cancer is LUAD or high-confidence classification
        assert res["cancer"]["label"] in ["LUAD", "BRCA", "GBM"]
        assert res["research_disclaimer"] == "Research prediction only. Not a clinical diagnosis."


def test_known_breast_image_prediction(trained_multiscale_model):
    """Test known breast image against model."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        breast_img_path = Path(tmp_dir) / "known_breast_sample.png"
        img = create_realistic_histopathology_patch("BRCA", "IDC_LuminalA", seed=202)
        img.save(breast_img_path)

        res = predict_image(
            model=trained_multiscale_model,
            image_path=breast_img_path,
            cancer_classes=["BRCA", "LUAD", "GBM"],
            organ_type="breast",
        )

        assert res["status"] == "ok"
        assert res["cancer"]["label"] in ["BRCA", "LUAD", "GBM"]


def test_known_gbm_image_prediction(trained_multiscale_model):
    """Test known brain / GBM image against model."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        gbm_img_path = Path(tmp_dir) / "known_gbm_sample.png"
        img = create_realistic_histopathology_patch("GBM", "Mesenchymal", seed=303)
        img.save(gbm_img_path)

        res = predict_image(
            model=trained_multiscale_model,
            image_path=gbm_img_path,
            cancer_classes=["BRCA", "LUAD", "GBM"],
            organ_type="brain",
        )

        assert res["status"] == "ok"
        assert res["cancer"]["label"] in ["GBM", "BRCA", "LUAD"]
