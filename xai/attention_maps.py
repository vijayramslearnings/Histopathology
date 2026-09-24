"""Utilities for saving learned scale, patch, and MIL attention maps."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def attention_heatmap(
    patch_weights: list[float] | np.ndarray,
    scales: int = 3,
    grid_size: int = 4,
) -> np.ndarray:
    values = np.asarray(patch_weights, dtype=np.float32)
    expected = scales * grid_size * grid_size
    if values.size != expected:
        raise ValueError(f"Expected {expected} patch attention values, received {values.size}.")
    values = values.reshape(scales, grid_size, grid_size).mean(axis=0)
    values -= values.min()
    maximum = values.max()
    return values / maximum if maximum > 0 else values


def save_attention_heatmap(
    heatmap: np.ndarray,
    output_path: str | Path,
    size: tuple[int, int] = (224, 224),
) -> None:
    image = Image.fromarray(np.uint8(np.clip(heatmap, 0, 1) * 255))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.resize(size, Image.Resampling.BICUBIC).save(output_path)


def save_attention_metadata(
    metadata: dict[str, Any],
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")