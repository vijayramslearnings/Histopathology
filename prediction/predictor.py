"""High-level Predictor interface for single and batch pathology inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch

from .inference import load_checkpoint, predict_image


class HistopathologyPredictor:
    """Predictor wrapper for multi-scale and baseline PyTorch pathology models."""

    def __init__(
        self,
        checkpoint_path: Optional[str | Path] = None,
        model: Optional[torch.nn.Module] = None,
        checkpoint_meta: Optional[dict[str, Any]] = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = torch.device(device)
        if checkpoint_path is not None:
            self.model, self.checkpoint = load_checkpoint(checkpoint_path, device=self.device)
        elif model is not None and checkpoint_meta is not None:
            self.model = model.to(self.device).eval()
            self.checkpoint = checkpoint_meta
        else:
            raise ValueError("Either checkpoint_path or (model, checkpoint_meta) must be provided.")

        self.cancer_classes = self.checkpoint["cancer_classes"]
        self.subtype_classes = self.checkpoint.get("subtype_classes", [])
        self.image_size = self.checkpoint.get("image_size", 224)
        self.model_name = self.checkpoint.get("model_config", {}).get("name", "multiscale")
        self.scale_fractions = tuple(self.checkpoint.get("scale_fractions", (1.0, 0.75, 0.55)))

    def predict(
        self,
        image_path: str | Path,
        confidence_threshold: float = 0.60,
        organ_type: Optional[str] = "auto",
    ) -> dict[str, Any]:
        """Predict cancer status, cancer type, subtype, probabilities, and uncertainty metrics."""
        return predict_image(
            model=self.model,
            image_path=image_path,
            cancer_classes=self.cancer_classes,
            subtype_classes=self.subtype_classes,
            image_size=self.image_size,
            model_name=self.model_name,
            device=self.device,
            scale_fractions=self.scale_fractions,
            confidence_threshold=confidence_threshold,
            organ_type=organ_type,
        )
