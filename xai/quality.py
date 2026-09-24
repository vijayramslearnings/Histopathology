"""Histopathology input image quality checker and tissue content detector."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image, ImageStat
from scipy.ndimage import laplace


def calculate_tissue_percentage(image_rgb: Image.Image, threshold: float = 220.0) -> float:
    """Calculate percentage of image area containing stained tissue vs white background."""
    img_array = np.asarray(image_rgb.convert("L"), dtype=np.float32)
    tissue_pixels = np.sum(img_array < threshold)
    total_pixels = img_array.size
    if total_pixels == 0:
        return 0.0
    return float((tissue_pixels / total_pixels) * 100.0)


def calculate_blur_score(image_rgb: Image.Image) -> float:
    """Compute variance of Laplacian to measure image sharpness/blur."""
    gray_array = np.asarray(image_rgb.convert("L"), dtype=np.float32)
    lap_var = float(np.var(laplace(gray_array)))
    return lap_var


def validate_image_quality(
    image_path: str | Path | Image.Image,
    min_resolution: tuple[int, int] = (100, 100),
    min_tissue_percent: float = 15.0,
    max_brightness: float = 248.0,
    min_brightness: float = 8.0,
    min_blur_score: float = 5.0,
) -> Dict[str, Any]:
    """Perform input image quality inspection before deep learning prediction."""
    warnings: List[str] = []
    
    # 1. Open Image
    try:
        if isinstance(image_path, (str, Path)):
            img = Image.open(image_path).convert("RGB")
        else:
            img = image_path.convert("RGB")
    except Exception as err:
        return {
            "quality": "poor",
            "is_acceptable": False,
            "warnings": [f"Corrupted or unreadable image file: {str(err)}"],
            "tissue_percentage": 0.0,
            "blur_score": 0.0,
        }

    w, h = img.size
    
    # 2. Minimum resolution check
    if w < min_resolution[0] or h < min_resolution[1]:
        warnings.append(f"Low resolution: {w}x{h} px is below minimum {min_resolution[0]}x{min_resolution[1]} px.")

    # 3. Brightness check
    stat = ImageStat.Stat(img.convert("L"))
    mean_brightness = float(stat.mean[0])
    if mean_brightness > max_brightness:
        warnings.append("Extreme brightness: image appears washed out or mostly blank background.")
    elif mean_brightness < min_brightness:
        warnings.append("Extreme darkness: image appears severely underexposed.")

    # 4. Blur check
    blur_score = calculate_blur_score(img)
    if blur_score < min_blur_score:
        warnings.append("High image blur: slide appears out of focus.")

    # 5. Tissue Content check
    tissue_percent = calculate_tissue_percentage(img)
    if tissue_percent < min_tissue_percent:
        warnings.append(f"Insufficient tissue content ({tissue_percent:.1f}% tissue area is below {min_tissue_percent}% threshold).")

    is_acceptable = len(warnings) == 0
    quality_status = "acceptable" if is_acceptable else "poor"

    return {
        "quality": quality_status,
        "is_acceptable": is_acceptable,
        "warnings": warnings,
        "tissue_percentage": round(tissue_percent, 2),
        "blur_score": round(blur_score, 2),
        "mean_brightness": round(mean_brightness, 2),
        "dimensions": {"width": w, "height": h},
    }
