"""Explanation stability evaluation under controlled image perturbations."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image, ImageEnhance
from scipy.ndimage import gaussian_filter

from dataset.preprocessing import build_preprocessor
from xai.gradcam import GradCAM


def compute_ssim(img1: np.ndarray, img2: np.ndarray, dynamic_range: float = 1.0) -> float:
    """Compute Structural Similarity Index (SSIM) between two normalized heatmaps."""
    x = np.asarray(img1, dtype=np.float32)
    y = np.asarray(img2, dtype=np.float32)
    
    c1 = (0.01 * dynamic_range) ** 2
    c2 = (0.03 * dynamic_range) ** 2
    
    mu_x = gaussian_filter(x, sigma=1.5)
    mu_y = gaussian_filter(y, sigma=1.5)
    
    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y
    
    sigma_x_sq = gaussian_filter(x ** 2, sigma=1.5) - mu_x_sq
    sigma_y_sq = gaussian_filter(y ** 2, sigma=1.5) - mu_y_sq
    sigma_xy = gaussian_filter(x * y, sigma=1.5) - mu_xy
    
    num = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    den = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
    
    ssim_map = num / (den + 1e-12)
    return float(np.mean(ssim_map))


def compute_pearson_correlation(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute Pearson correlation coefficient between two heatmaps."""
    x = img1.flatten()
    y = img2.flatten()
    if np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return 1.0 if np.allclose(x, y) else 0.0
    corr = np.corrcoef(x, y)[0, 1]
    return float(corr) if not np.isnan(corr) else 0.0


def compute_region_iou(img1: np.ndarray, img2: np.ndarray, threshold_percentile: float = 80.0) -> float:
    """Compute Intersection over Union (IoU) of top 20% important regions."""
    th1 = np.percentile(img1, threshold_percentile)
    th2 = np.percentile(img2, threshold_percentile)
    
    mask1 = img1 >= th1
    mask2 = img2 >= th2
    
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    if union == 0:
        return 1.0
    return float(intersection / union)


def generate_perturbations(image_pil: Image.Image) -> List[Tuple[str, Image.Image]]:
    """Generate 4 controlled perturbations: rotation, brightness, contrast, crop/scale."""
    w, h = image_pil.size
    perturbations: List[Tuple[str, Image.Image]] = []

    # 1. Rotation (+10 deg)
    perturbations.append(("rotation_10deg", image_pil.rotate(10, resample=Image.Resampling.BILINEAR)))
    
    # 2. Brightness (factor 1.15)
    enhancer_b = ImageEnhance.Brightness(image_pil)
    perturbations.append(("brightness_1.15", enhancer_b.enhance(1.15)))
    
    # 3. Contrast (factor 1.15)
    enhancer_c = ImageEnhance.Contrast(image_pil)
    perturbations.append(("contrast_1.15", enhancer_c.enhance(1.15)))
    
    # 4. Center Crop & Resize (90% crop scale)
    crop_w, crop_h = int(w * 0.90), int(h * 0.90)
    left = (w - crop_w) // 2
    top = (h - crop_h) // 2
    cropped = image_pil.crop((left, top, left + crop_w, top + crop_h))
    resized = cropped.resize((w, h), Image.Resampling.BILINEAR)
    perturbations.append(("crop_resize_90pct", resized))

    return perturbations


def evaluate_stability(
    model: torch.nn.Module,
    image_path: str | Path,
    original_heatmap: np.ndarray,
    target_class_index: int,
    image_size: int = 224,
    scale_fractions: tuple[float, float, float] = (1.0, 0.75, 0.55),
    device: str | torch.device = "cpu",
) -> Dict[str, Any]:
    """Evaluate stability of model prediction and Grad-CAM explanation under image perturbations."""
    device = torch.device(device)
    preprocessor = build_preprocessor(image_size, train=False, scale_fractions=scale_fractions)
    
    with Image.open(image_path) as img:
        original_pil = img.convert("RGB")
        
    perturbations = generate_perturbations(original_pil)
    
    ssim_list: List[float] = []
    corr_list: List[float] = []
    iou_list: List[float] = []
    prediction_matches: int = 0

    model.eval()

    for name, p_img in perturbations:
        p_tensor = preprocessor(p_img).unsqueeze(0).to(device)
        
        # 1. Run prediction under perturbation
        with torch.no_grad():
            out = model(p_tensor)
            logits = out["cancer_logits"] if isinstance(out, dict) else out
            pred_class = int(logits.argmax(dim=1)[0])
            if pred_class == target_class_index:
                prediction_matches += 1

        # 2. Generate Grad-CAM under perturbation
        cam_generator = GradCAM(model)
        try:
            p_heatmap = cam_generator(p_tensor, class_index=target_class_index)
        finally:
            cam_generator.close()

        # 3. Compute metrics comparing with original heatmap
        ssim_val = compute_ssim(original_heatmap, p_heatmap)
        corr_val = compute_pearson_correlation(original_heatmap, p_heatmap)
        iou_val = compute_region_iou(original_heatmap, p_heatmap)

        ssim_list.append(ssim_val)
        corr_list.append(corr_val)
        iou_list.append(iou_val)

    pred_stability = float(prediction_matches / len(perturbations)) if perturbations else 1.0
    mean_ssim = float(np.mean(ssim_list)) if ssim_list else 1.0
    mean_corr = float(np.mean(corr_list)) if corr_list else 1.0
    mean_iou = float(np.mean(iou_list)) if iou_list else 1.0

    summary = (
        f"Stability evaluation: Heatmap SSIM = {mean_ssim:.3f}, Correlation = {mean_corr:.3f}, "
        f"Region IoU = {mean_iou:.3f}, Prediction Stability = {pred_stability*100:.1f}%."
    )

    return {
        "prediction_stability": round(pred_stability, 4),
        "heatmap_similarity": round(mean_ssim, 4),
        "region_iou": round(mean_iou, 4),
        "ssim": round(mean_ssim, 4),
        "correlation": round(mean_corr, 4),
        "summary": summary,
    }
