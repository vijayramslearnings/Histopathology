from __future__ import annotations

import itertools

import torch

from backend.models.multi_scale_model import MultiScaleAttentionNetwork
from backend.prediction.inference import load_checkpoint


torch.set_num_threads(1)


def test_multiscale_checkpoint_reload_preserves_each_ablation(tmp_path) -> None:
    for ablation in ("full", "no_scale_attention", "no_patch_attention", "no_mil"):
        switches = {
            "use_scale_attention": ablation != "no_scale_attention",
            "use_patch_attention": ablation != "no_patch_attention",
            "use_mil": ablation != "no_mil",
        }
        source = MultiScaleAttentionNetwork(
            cancer_classes=3,
            subtype_classes=0,
            pretrained=False,
            fusion_dim=32,
            hidden_dim=16,
            dropout=0.0,
            shared_backbone=True,
            patch_grid=1,
            **switches,
        ).eval()
        checkpoint_path = tmp_path / f"{ablation}.pt"
        torch.save(
            {
                "model_state": source.state_dict(),
                "model_config": {
                    "name": "multiscale",
                    "fusion_dim": 32,
                    "hidden_dim": 16,
                    "dropout": 0.0,
                    "shared_backbone": True,
                    "patch_grid": 1,
                    **switches,
                },
                "cancer_classes": ["BRCA", "LUAD", "GBM"],
                "subtype_classes": [],
            },
            checkpoint_path,
        )
        restored, checkpoint = load_checkpoint(checkpoint_path, "cpu")
        assert checkpoint["model_config"]["name"] == "multiscale"
        assert all(
            torch.equal(left, right)
            for left, right in zip(source.state_dict().values(), restored.state_dict().values())
        )
        with torch.no_grad():
            output = restored(torch.randn(1, 3, 3, 64, 64))
        assert output["cancer_logits"].shape == (1, 3)