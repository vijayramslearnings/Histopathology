"""Quantitative explanation faithfulness evaluation (Deletion & Insertion AUC tests)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

from dataset.preprocessing import BaselinePreprocessor, build_preprocessor


def trapezoid_auc(y: list[float] | np.ndarray, x: list[float] | np.ndarray) -> float:
    """Calculate Area Under Curve (AUC) using trapezoidal rule (NumPy 2.0+ compatible)."""
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    elif hasattr(np, "trapz"):
        return float(np.trapz(y, x))
    else:
        from scipy.integrate import trapezoid
        return float(trapezoid(y, x))


def evaluate_faithfulness(
    model: torch.nn.Module,
    image_path: str | Path,
    patches: List[Dict[str, Any]],
    target_class_index: int,
    image_size: int = 224,
    scale_fractions: tuple[float, float, float] = (1.0, 0.75, 0.55),
    device: str | torch.device = "cpu",
    steps: int = 10,
) -> Dict[str, Any]:
    """Execute quantitative Deletion and Insertion tests to measure explanation faithfulness."""
    device = torch.device(device)
    preprocessor = build_preprocessor(image_size, train=False, scale_fractions=scale_fractions)
    
    with Image.open(image_path) as img:
        original_pil = img.convert("RGB")
        
    w, h = original_pil.size
    
    # Sort patches by attention score descending
    sorted_patches = sorted(patches, key=lambda p: p["attention_score"], reverse=True)
    n_patches = len(sorted_patches)
    if n_patches == 0:
        return {
            "deletion_auc": 0.5,
            "insertion_auc": 0.5,
            "deletion_curve": [],
            "insertion_curve": [],
            "summary": "Faithfulness test skipped due to zero patches.",
        }

    percentages = np.linspace(0.0, 1.0, num=min(steps, n_patches) + 1)
    
    deletion_confidences: List[float] = []
    insertion_confidences: List[float] = []
    deletion_curve: List[Dict[str, float]] = []
    insertion_curve: List[Dict[str, float]] = []

    model.eval()

    # Pre-create baseline image (solid zero/gray canvas)
    baseline_pil = Image.new("RGB", (w, h), color=(128, 128, 128))

    for p_val in percentages:
        num_to_mask = int(round(p_val * n_patches))
        
        # --- DELETION TEST ---
        # Mask out top `num_to_mask` patches in original image
        del_img = original_pil.copy()
        del_canvas = Image.new("RGB", (w, h), color=(128, 128, 128))
        for p in sorted_patches[:num_to_mask]:
            x1, y1 = p["x"], p["y"]
            x2, y2 = x1 + p["width"], y1 + p["height"]
            del_img.paste(del_canvas.crop((x1, y1, x2, y2)), (x1, y1))
            
        with torch.no_grad():
            del_tensor = preprocessor(del_img).unsqueeze(0).to(device)
            del_out = model(del_tensor)
            del_logits = del_out["cancer_logits"] if isinstance(del_out, dict) else del_out
            del_probs = torch.softmax(del_logits, dim=1)[0]
            del_conf = float(del_probs[target_class_index])
            
        deletion_confidences.append(del_conf)
        deletion_curve.append({
            "percentage_deleted": round(float(p_val * 100), 1),
            "confidence": round(del_conf, 4),
        })

        # --- INSERTION TEST ---
        # Start with baseline, add top `num_to_mask` patches progressively
        ins_img = baseline_pil.copy()
        for p in sorted_patches[:num_to_mask]:
            x1, y1 = p["x"], p["y"]
            x2, y2 = x1 + p["width"], y1 + p["height"]
            ins_img.paste(original_pil.crop((x1, y1, x2, y2)), (x1, y1))
            
        with torch.no_grad():
            ins_tensor = preprocessor(ins_img).unsqueeze(0).to(device)
            ins_out = model(ins_tensor)
            ins_logits = ins_out["cancer_logits"] if isinstance(ins_out, dict) else ins_out
            ins_probs = torch.softmax(ins_logits, dim=1)[0]
            ins_conf = float(ins_probs[target_class_index])
            
        insertion_confidences.append(ins_conf)
        insertion_curve.append({
            "percentage_inserted": round(float(p_val * 100), 1),
            "confidence": round(ins_conf, 4),
        })

    # Calculate Area Under Curve (AUC) using trapezoidal rule
    deletion_auc = trapezoid_auc(deletion_confidences, percentages)
    insertion_auc = trapezoid_auc(insertion_confidences, percentages)

    summary = (
        f"Faithfulness test results: Deletion AUC = {deletion_auc:.3f}, Insertion AUC = {insertion_auc:.3f}. "
        "A sharp confidence drop during region deletion indicates model reliance on highlighted features."
    )

    return {
        "deletion_auc": round(deletion_auc, 4),
        "insertion_auc": round(insertion_auc, 4),
        "deletion_curve": deletion_curve,
        "insertion_curve": insertion_curve,
        "summary": summary,
    }
