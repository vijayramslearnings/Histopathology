"""Small-image patch extraction helpers for MIL and future WSI adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image


def extract_patches(
    image: Image.Image | np.ndarray,
    patch_size: int = 224,
    stride: int = 224,
) -> Iterator[Image.Image]:
    """Yield RGB patches from an image; incomplete edge patches are skipped."""
    if isinstance(image, Image.Image):
        array = np.asarray(image.convert("RGB"))
    else:
        array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] not in (3, 4):
        raise ValueError("Expected an RGB or RGBA image array.")
    height, width = array.shape[:2]
    for top in range(0, height - patch_size + 1, stride):
        for left in range(0, width - patch_size + 1, stride):
            patch = array[top : top + patch_size, left : left + patch_size, :3]
            yield Image.fromarray(patch)


def save_patches(
    image_path: str | Path,
    output_dir: str | Path,
    patch_size: int = 224,
    stride: int = 224,
) -> int:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    with Image.open(image_path) as image:
        for count, patch in enumerate(
            extract_patches(image, patch_size=patch_size, stride=stride), start=1
        ):
            patch.save(output_dir / f"patch_{count:05d}.png")
    return count


def extract_tissue_patches(
    image_path: str | Path,
    patch_size: int = 224,
    stride: int = 224,
    tissue_threshold: float = 0.15,
) -> list[dict[str, Any]]:
    """Extract tissue patch tiles with spatial coordinates and tissue area percentages."""
    results = []
    with Image.open(image_path) as image:
        array = np.asarray(image.convert("RGB"))
        height, width = array.shape[:2]
        idx = 0
        for top in range(0, height - patch_size + 1, stride):
            for left in range(0, width - patch_size + 1, stride):
                patch = array[top : top + patch_size, left : left + patch_size, :3]
                # Tissue fraction heuristic (hsv / luminance thresholding)
                mean_lum = np.mean(patch)
                tissue_pct = float(np.mean(mean_lum < 220))
                if tissue_pct >= tissue_threshold:
                    idx += 1
                    results.append({
                        "patch_id": f"p_{idx:03d}",
                        "x": left,
                        "y": top,
                        "width": patch_size,
                        "height": patch_size,
                        "tissue_percentage": round(tissue_pct * 100.0, 2),
                    })
    return results