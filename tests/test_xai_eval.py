"""Unit tests for faithfulness and stability evaluations."""

from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np
import torch
from PIL import Image

from models.multi_scale_model import MultiScaleAttentionNetwork
from xai.faithfulness import evaluate_faithfulness
from xai.patch_attention import extract_patch_attentions
from xai.stability import (
    compute_pearson_correlation,
    compute_region_iou,
    compute_ssim,
    evaluate_stability,
)


def test_stability_metrics_math():
    img1 = np.ones((10, 10), dtype=np.float32)
    img2 = np.ones((10, 10), dtype=np.float32)

    ssim_val = compute_ssim(img1, img2)
    corr_val = compute_pearson_correlation(img1, img2)
    iou_val = compute_region_iou(img1, img2)

    assert abs(ssim_val - 1.0) < 1e-3
    assert abs(corr_val - 1.0) < 1e-3
    assert abs(iou_val - 1.0) < 1e-3


def test_faithfulness_and_stability_evaluation():
    with tempfile.TemporaryDirectory() as tmp_dir:
        img_path = Path(tmp_dir) / "sample.png"
        Image.new("RGB", (224, 224), color="red").save(img_path)

        model = MultiScaleAttentionNetwork(cancer_classes=3, pretrained=False).eval()
        dummy_weights = [i * 0.02 for i in range(48)]
        patches = extract_patch_attentions(dummy_weights)

        # 1. Faithfulness
        faith_res = evaluate_faithfulness(
            model=model,
            image_path=img_path,
            patches=patches,
            target_class_index=0,
            steps=3,
        )

        assert "deletion_auc" in faith_res
        assert "insertion_auc" in faith_res
        assert len(faith_res["deletion_curve"]) > 0

        # 2. Stability
        dummy_heatmap = np.random.rand(224, 224).astype(np.float32)
        stab_res = evaluate_stability(
            model=model,
            image_path=img_path,
            original_heatmap=dummy_heatmap,
            target_class_index=0,
        )

        assert "ssim" in stab_res
        assert "correlation" in stab_res
        assert "prediction_stability" in stab_res
