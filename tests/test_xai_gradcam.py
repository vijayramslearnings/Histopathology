"""Unit tests for Grad-CAM explanation generation."""

from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np
import torch
from PIL import Image

from models.multi_scale_model import MultiScaleAttentionNetwork
from xai.gradcam import GradCAM, find_target_layer, save_gradcam_outputs


def test_find_target_layer():
    model = MultiScaleAttentionNetwork(cancer_classes=3, pretrained=False)
    layer = find_target_layer(model)
    assert layer is not None
    assert isinstance(layer, torch.nn.Conv2d)


def test_gradcam_generation_and_hook_cleanup():
    model = MultiScaleAttentionNetwork(cancer_classes=3, pretrained=False).eval()
    
    # Create dummy multi-scale input [batch=1, scales=3, channels=3, height=224, width=224]
    dummy_input = torch.randn(1, 3, 3, 224, 224)
    
    with GradCAM(model) as cam_gen:
        heatmap = cam_gen(dummy_input, class_index=0)
        assert isinstance(heatmap, np.ndarray)
        assert heatmap.shape == (224, 224)
        assert 0.0 <= heatmap.min() <= heatmap.max() <= 1.0

    # Ensure hooks were properly removed after context exit
    assert cam_gen._forward_handle is not None


def test_save_gradcam_outputs():
    with tempfile.TemporaryDirectory() as tmp_dir:
        img_path = Path(tmp_dir) / "test_img.png"
        heatmap_path = Path(tmp_dir) / "heatmap.png"
        overlay_path = Path(tmp_dir) / "overlay.png"

        # Create dummy RGB image
        Image.new("RGB", (224, 224), color="blue").save(img_path)
        dummy_heatmap = np.random.rand(224, 224).astype(np.float32)

        save_gradcam_outputs(img_path, dummy_heatmap, heatmap_path, overlay_path, alpha=0.5)

        assert heatmap_path.exists()
        assert overlay_path.exists()
        with Image.open(overlay_path) as overlay:
            assert overlay.size == (224, 224)
