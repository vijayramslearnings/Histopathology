"""Metrics and checkpoint evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    cancer_classes: list[str],
    subtype_classes: list[str],
) -> dict[str, Any]:
    model.eval()
    cancer_true: list[int] = []
    cancer_pred: list[int] = []
    subtype_true: list[int] = []
    subtype_pred: list[int] = []
    total_loss = 0.0
    batches = 0

    for batch in loader:
        inputs = batch["image"].to(device)
        outputs = model(inputs)
        cancer_targets = batch["cancer_label"].to(device)
        cancer_predictions = outputs["cancer_logits"].argmax(dim=1)
        cancer_true.extend(cancer_targets.cpu().tolist())
        cancer_pred.extend(cancer_predictions.cpu().tolist())
        if outputs["subtype_logits"] is not None:
            subtype_targets = batch["subtype_label"].to(device)
            subtype_predictions = outputs["subtype_logits"].argmax(dim=1)
            valid = subtype_targets >= 0
            subtype_true.extend(subtype_targets[valid].cpu().tolist())
            subtype_pred.extend(subtype_predictions[valid].cpu().tolist())
        batches += 1

    result: dict[str, Any] = {
        "samples": len(cancer_true),
        "batches": batches,
        "cancer": {
            "accuracy": float(accuracy_score(cancer_true, cancer_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(cancer_true, cancer_pred)),
            "macro_f1": float(f1_score(cancer_true, cancer_pred, average="macro", zero_division=0)),
            "confusion_matrix": confusion_matrix(
                cancer_true, cancer_pred, labels=list(range(len(cancer_classes)))
            ).tolist(),
            "classes": cancer_classes,
        },
    }
    if subtype_classes and subtype_true:
        result["subtype"] = {
            "samples": len(subtype_true),
            "accuracy": float(accuracy_score(subtype_true, subtype_pred)),
            "balanced_accuracy": float(
                balanced_accuracy_score(subtype_true, subtype_pred)
            ),
            "macro_f1": float(
                f1_score(subtype_true, subtype_pred, average="macro", zero_division=0)
            ),
            "confusion_matrix": confusion_matrix(
                subtype_true, subtype_pred, labels=list(range(len(subtype_classes)))
            ).tolist(),
            "classes": subtype_classes,
        }
    return result


def save_metrics(metrics: dict[str, Any], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")