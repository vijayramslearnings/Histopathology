"""Robustness testing script evaluating model and explanation stability across perturbations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

from dataset.preprocessing import build_preprocessor
from xai.gradcam import GradCAM
from xai.stability import compute_region_iou, compute_ssim, generate_perturbations


def evaluate_dataset_robustness(
    model: torch.nn.Module,
    image_paths: List[str | Path],
    target_classes: List[int],
    image_size: int = 224,
    scale_fractions: tuple[float, float, float] = (1.0, 0.75, 0.55),
    output_path: str | Path = "outputs/metrics/robustness_report.json",
    device: str | torch.device = "cpu",
) -> Dict[str, Any]:
    """Execute robustness evaluation across a dataset of test slides."""
    device = torch.device(device)
    preprocessor = build_preprocessor(image_size, train=False, scale_fractions=scale_fractions)
    model = model.to(device).eval()

    sample_reports: List[Dict[str, Any]] = []
    total_samples = len(image_paths)

    for idx, (img_p, t_class) in enumerate(zip(image_paths, target_classes)):
        img_p = Path(img_p)
        if not img_p.exists():
            continue

        with Image.open(img_p) as img:
            orig_pil = img.convert("RGB")

        orig_tensor = preprocessor(orig_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            orig_out = model(orig_tensor)
            orig_logits = orig_out["cancer_logits"] if isinstance(orig_out, dict) else orig_out
            orig_probs = torch.softmax(orig_logits, dim=1)[0]
            orig_pred = int(orig_probs.argmax())
            orig_conf = float(orig_probs[orig_pred])

        cam_gen = GradCAM(model)
        try:
            orig_cam = cam_gen(orig_tensor, class_index=orig_pred)
        finally:
            cam_gen.close()

        perturbations = generate_perturbations(orig_pil)
        matches = 0
        ssim_scores: List[float] = []
        iou_scores: List[float] = []
        conf_deltas: List[float] = []

        for p_name, p_img in perturbations:
            p_tensor = preprocessor(p_img).unsqueeze(0).to(device)
            with torch.no_grad():
                p_out = model(p_tensor)
                p_logits = p_out["cancer_logits"] if isinstance(p_out, dict) else p_out
                p_probs = torch.softmax(p_logits, dim=1)[0]
                p_pred = int(p_probs.argmax())
                p_conf = float(p_probs[p_pred])

            if p_pred == orig_pred:
                matches += 1

            cam_gen2 = GradCAM(model)
            try:
                p_cam = cam_gen2(p_tensor, class_index=orig_pred)
            finally:
                cam_gen2.close()

            ssim_val = compute_ssim(orig_cam, p_cam)
            iou_val = compute_region_iou(orig_cam, p_cam)

            ssim_scores.append(ssim_val)
            iou_scores.append(iou_val)
            conf_deltas.append(abs(orig_conf - p_conf))

        sample_reports.append({
            "image_path": str(img_p),
            "original_pred": orig_pred,
            "original_conf": round(orig_conf, 4),
            "prediction_consistency": round(matches / len(perturbations), 4) if perturbations else 1.0,
            "mean_ssim": round(float(np.mean(ssim_scores)), 4) if ssim_scores else 1.0,
            "mean_iou": round(float(np.mean(iou_scores)), 4) if iou_scores else 1.0,
            "mean_conf_delta": round(float(np.mean(conf_deltas)), 4) if conf_deltas else 0.0,
        })

    avg_consistency = float(np.mean([s["prediction_consistency"] for s in sample_reports])) if sample_reports else 1.0
    avg_ssim = float(np.mean([s["mean_ssim"] for s in sample_reports])) if sample_reports else 1.0
    avg_iou = float(np.mean([s["mean_iou"] for s in sample_reports])) if sample_reports else 1.0

    report: Dict[str, Any] = {
        "total_samples": total_samples,
        "overall_prediction_consistency": round(avg_consistency, 4),
        "overall_heatmap_ssim": round(avg_ssim, 4),
        "overall_region_iou": round(avg_iou, 4),
        "sample_reports": sample_reports,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report
