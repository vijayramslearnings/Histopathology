"""Wrong-prediction error analysis, confusion matrix generator, and error artifact exporter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from xai.ood import compute_brier_score, compute_ece


def analyze_prediction_error(
    image_path: str | Path,
    actual_label: str,
    predicted_label: str,
    confidence: float,
    class_probabilities: Dict[str, float],
    gradcam_overlay_path: Optional[str | Path] = None,
    patch_vis_path: Optional[str | Path] = None,
    scale_attention: Optional[Dict[str, float]] = None,
    output_dir: str | Path = "outputs/errors",
) -> Dict[str, Any]:
    """Identify and archive misclassified samples for clinical/model error research."""
    is_correct = actual_label.strip().upper() == predicted_label.strip().upper()
    error_type = "correct" if is_correct else f"{actual_label}_to_{predicted_label}"

    record: Dict[str, Any] = {
        "image_path": str(image_path),
        "actual_label": actual_label,
        "predicted_label": predicted_label,
        "is_correct": is_correct,
        "error_type": error_type,
        "confidence": round(float(confidence), 4),
        "class_probabilities": class_probabilities,
        "scale_attention": scale_attention or {},
    }

    if not is_correct:
        folder_name = f"{actual_label.upper()}_to_{predicted_label.upper()}"
        save_folder = Path(output_dir) / folder_name
        save_folder.mkdir(parents=True, exist_ok=True)

        sample_name = Path(image_path).stem
        
        # Copy or save original image
        try:
            with Image.open(image_path) as img:
                img.save(save_folder / f"{sample_name}_original.png")
        except Exception:
            pass

        # Save metadata JSON
        (save_folder / f"{sample_name}_error_meta.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8"
        )
        record["error_archive_folder"] = str(save_folder)

    return record


def generate_evaluation_metrics_and_matrix(
    y_true: List[int],
    y_pred: List[int],
    y_probs: np.ndarray,
    class_names: List[str],
    output_dir: str | Path = "outputs/metrics",
) -> Dict[str, Any]:
    """Generate confusion matrix plot and classification metrics report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    micro_f1 = float(f1_score(y_true, y_pred, average="micro"))
    macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    try:
        roc_auc = float(roc_auc_score(y_true, y_probs, multi_class="ovr"))
    except Exception:
        roc_auc = 0.5

    ece = compute_ece(y_probs, np.array(y_true))
    brier = compute_brier_score(y_probs, np.array(y_true), num_classes=len(class_names))

    per_class_report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )

    metrics_dict: Dict[str, Any] = {
        "macro_f1": round(macro_f1, 4),
        "micro_f1": round(micro_f1, 4),
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "roc_auc": round(roc_auc, 4),
        "ece": round(ece, 4),
        "brier_score": round(brier, 4),
        "per_class": per_class_report,
        "confusion_matrix": cm.tolist(),
    }

    # Plot & save Confusion Matrix PNG
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title="Histopathology Cancer Classification Confusion Matrix",
        ylabel="Actual Class",
        xlabel="Predicted Class",
    )

    # Loop over data dimensions and create text annotations
    thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    cm_path = output_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=200)
    plt.close(fig)

    # Save classification report JSON
    (output_dir / "classification_report.json").write_text(
        json.dumps(metrics_dict, indent=2), encoding="utf-8"
    )

    metrics_dict["confusion_matrix_path"] = str(cm_path)
    return metrics_dict
