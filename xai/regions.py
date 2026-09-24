"""Top important region bounding box extractor and crop image generator."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image


def extract_top_regions(
    image_path: str | Path,
    patches: List[Dict[str, Any]],
    output_dir: str | Path = "outputs/regions",
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Extract top-k important region bounding boxes and save cropped image assets."""
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(image_path) as img:
        img_rgb = img.convert("RGB")
        w, h = img_rgb.size

    # Sort patches by attention score descending
    sorted_patches = sorted(patches, key=lambda p: p["attention_score"], reverse=True)
    top_patches = sorted_patches[:top_k]

    region_records: List[Dict[str, Any]] = []
    for rank, patch in enumerate(top_patches, start=1):
        x = patch.get("x", 0)
        y = patch.get("y", 0)
        pw = patch.get("width", 56)
        ph = patch.get("height", 56)
        
        # Clamp crop coordinates within image bounds
        x1 = max(0, min(x, w - 1))
        y1 = max(0, min(y, h - 1))
        x2 = max(x1 + 1, min(x1 + pw, w))
        y2 = max(y1 + 1, min(y1 + ph, h))

        crop_img = img_rgb.crop((x1, y1, x2, y2))
        crop_filename = f"region_{rank}_{patch.get('scale', 'scale')}_score_{int(patch['attention_score']*100)}.png"
        crop_path = output_dir / crop_filename
        crop_img.save(crop_path)

        region_records.append({
            "region_id": rank,
            "scale": patch.get("scale", "high-scale"),
            "attention_score": patch.get("attention_score", 0.0),
            "coordinates": {
                "x": x1,
                "y": y1,
                "width": x2 - x1,
                "height": y2 - y1,
            },
            "crop_path": str(crop_path),
            "crop_url": f"/outputs/regions/{crop_filename}",
            "description": f"Region #{rank} with attention score {patch.get('attention_score', 0.0):.2f} at {patch.get('scale', 'high-scale')}.",
        })

    return region_records
