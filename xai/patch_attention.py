"""Patch-level attention extraction, ranking, and visualization generator."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Tuple, Dict

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def extract_patch_attentions(
    patch_weights: list[float] | np.ndarray,
    scale_names: tuple[str, ...] = ("5x", "10x", "20x"),
    grid_size: int = 4,
    image_size: tuple[int, int] = (224, 224),
) -> List[Dict[str, Any]]:
    """Extract coordinate-mapped and normalized patch attention scores."""
    weights = np.asarray(patch_weights, dtype=np.float32)
    num_scales = len(scale_names)
    expected_patches = num_scales * grid_size * grid_size
    
    if weights.size != expected_patches:
        # Fallback if patch count differs from grid size
        grid_size = int(np.sqrt(weights.size // num_scales))
        expected_patches = num_scales * grid_size * grid_size

    # Min-max normalize scores to [0, 1]
    min_val, max_val = weights.min(), weights.max()
    if max_val - min_val > 1e-8:
        norm_weights = (weights - min_val) / (max_val - min_val)
    else:
        norm_weights = np.zeros_like(weights)

    img_w, img_h = image_size
    patch_w = img_w / grid_size
    patch_h = img_h / grid_size

    patches: List[Dict[str, Any]] = []
    idx = 0
    for scale_idx, scale_name in enumerate(scale_names):
        for row in range(grid_size):
            for col in range(grid_size):
                if idx < len(norm_weights):
                    score = float(norm_weights[idx])
                    raw_score = float(weights[idx])
                    x_start = int(col * patch_w)
                    y_start = int(row * patch_h)
                    patches.append({
                        "patch_id": idx,
                        "scale": scale_name,
                        "grid_row": row,
                        "grid_col": col,
                        "x": x_start,
                        "y": y_start,
                        "width": int(patch_w),
                        "height": int(patch_h),
                        "attention_score": round(score, 4),
                        "raw_score": round(raw_score, 4),
                    })
                    idx += 1
    return patches


def rank_patch_importance(
    patches: List[Dict[str, Any]]
) -> Dict[str, List[Dict[str, Any]]]:
    """Sort patches by attention score into top-5, top-10, and least-5 categories."""
    sorted_patches = sorted(patches, key=lambda p: p["attention_score"], reverse=True)
    return {
        "top_5": sorted_patches[:5],
        "top_10": sorted_patches[:10],
        "least_5": sorted_patches[-5:] if len(sorted_patches) >= 5 else sorted_patches,
        "all_sorted": sorted_patches,
    }


def create_patch_visualization(
    image_path: str | Path,
    patches: List[Dict[str, Any]],
    output_path: str | Path,
    top_k: int = 5,
    grid_size: int = 4,
    show_labels: bool = True,
) -> Path:
    """Create patch attention grid visualization with boundaries and top-k highlights."""
    image_path = Path(image_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    base_image = Image.open(image_path).convert("RGB")
    w, h = base_image.size
    
    # Overlay heatmap layer
    overlay = base_image.copy()
    draw_overlay = ImageDraw.Draw(overlay, "RGBA")
    
    # Compute average spatial patch attention across scales for grid visualization
    spatial_scores = np.zeros((grid_size, grid_size), dtype=np.float32)
    spatial_counts = np.zeros((grid_size, grid_size), dtype=np.int32)
    
    for p in patches:
        r, c = p.get("grid_row", 0), p.get("grid_col", 0)
        spatial_scores[r, c] += p["attention_score"]
        spatial_counts[r, c] += 1
        
    spatial_counts[spatial_counts == 0] = 1
    avg_spatial = spatial_scores / spatial_counts
    max_avg = avg_spatial.max()
    if max_avg > 0:
        avg_spatial /= max_avg

    patch_w = w / grid_size
    patch_h = h / grid_size

    # Draw semi-transparent intensity rects
    for r in range(grid_size):
        for c in range(grid_size):
            score = avg_spatial[r, c]
            x1, y1 = c * patch_w, r * patch_h
            x2, y2 = x1 + patch_w, y1 + patch_h
            alpha_val = int(score * 120)  # max alpha 120
            # Red/amber highlight
            draw_overlay.rectangle([x1, y1, x2, y2], fill=(255, int(150 * (1 - score)), 0, alpha_val))

    # Blend overlay with original
    blended = Image.alpha_composite(base_image.convert("RGBA"), overlay.convert("RGBA"))
    draw_final = ImageDraw.Draw(blended)

    # Draw patch grid lines
    for r in range(1, grid_size):
        draw_final.line([(0, r * patch_h), (w, r * patch_h)], fill=(255, 255, 255, 180), width=1)
    for c in range(1, grid_size):
        draw_final.line([(c * patch_w, 0), (c * patch_w, h)], fill=(255, 255, 255, 180), width=1)

    # Highlight top-k patches
    top_patches = sorted(patches, key=lambda p: p["attention_score"], reverse=True)[:top_k]
    for rank, p in enumerate(top_patches, start=1):
        x1, y1 = p["x"], p["y"]
        x2, y2 = x1 + p["width"], y1 + p["height"]
        draw_final.rectangle([x1, y1, x2, y2], outline=(0, 255, 255, 255), width=3)
        if show_labels:
            label_text = f"#{rank} ({p['attention_score']:.2f})"
            draw_final.rectangle([x1, y1, x1 + 70, y1 + 18], fill=(0, 0, 0, 200))
            draw_final.text((x1 + 3, y1 + 2), label_text, fill=(0, 255, 255, 255))

    blended.convert("RGB").save(output_path)
    return output_path
