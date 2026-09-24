"""Unit tests for patch and scale attention modules."""

from __future__ import annotations

import tempfile
from pathlib import Path
from PIL import Image

from xai.patch_attention import (
    create_patch_visualization,
    extract_patch_attentions,
    rank_patch_importance,
)
from xai.scale_attention import process_scale_attention


def test_extract_and_rank_patch_attentions():
    dummy_weights = [i * 0.05 for i in range(48)]  # 3 scales * 4x4 grid = 48 patches
    patches = extract_patch_attentions(
        patch_weights=dummy_weights,
        scale_names=("5x", "10x", "20x"),
        grid_size=4,
        image_size=(224, 224),
    )
    assert len(patches) == 48
    assert patches[0]["attention_score"] == 0.0
    assert patches[-1]["attention_score"] == 1.0

    ranked = rank_patch_importance(patches)
    assert len(ranked["top_5"]) == 5
    assert len(ranked["top_10"]) == 10
    assert len(ranked["least_5"]) == 5
    assert ranked["top_5"][0]["attention_score"] >= ranked["top_5"][-1]["attention_score"]


test_patch_vis_path = None

def test_create_patch_visualization():
    with tempfile.TemporaryDirectory() as tmp_dir:
        img_path = Path(tmp_dir) / "input.png"
        out_path = Path(tmp_dir) / "patch_vis.png"
        Image.new("RGB", (224, 224), color="white").save(img_path)

        dummy_weights = [0.1] * 48
        patches = extract_patch_attentions(dummy_weights)
        res_path = create_patch_visualization(img_path, patches, out_path, top_k=5)

        assert res_path.exists()


def test_process_scale_attention():
    raw_weights = [0.21, 0.34, 0.45]
    res = process_scale_attention(raw_weights, scale_names=("5x", "10x", "20x"))
    
    assert res["dominant_scale"] == "20x"
    assert res["scale_attention"]["20x"] == 0.45
    assert "highest learned importance" in res["summary"]
