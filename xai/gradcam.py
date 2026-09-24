"""Grad-CAM visual explanation generator for ResNet50 backbone and multi-scale models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn


def find_target_layer(model: nn.Module) -> nn.Module:
    """Automatically identify target convolutional layer for Grad-CAM."""
    if hasattr(model, "target_layer") and callable(getattr(model, "target_layer")):
        return model.target_layer()
    
    # Traverse modules in reverse order to find the last Conv2d layer
    target_layer: Optional[nn.Module] = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            target_layer = module
            
    if target_layer is None:
        raise RuntimeError("Failed to find any nn.Conv2d target layer in the model.")
    return target_layer


class GradCAM:
    """Hook-based Grad-CAM generator capturing activations and gradients."""

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None) -> None:
        self.model = model
        self.target_layer = target_layer or find_target_layer(model)
        self.activations: Optional[Tensor] = None
        self.gradients: Optional[Tensor] = None
        self._forward_handle = self.target_layer.register_forward_hook(self._save_activation)
        self._backward_handle = self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module: nn.Module, _inputs: tuple[Tensor, ...], output: Tensor) -> None:
        self.activations = output.detach()

    def _save_gradient(
        self, _module: nn.Module, _inputs: tuple[Tensor, ...], outputs: tuple[Tensor, ...]
    ) -> None:
        self.gradients = outputs[0].detach()

    def close(self) -> None:
        """Safely remove forward and backward hooks."""
        if hasattr(self, "_forward_handle") and self._forward_handle:
            self._forward_handle.remove()
        if hasattr(self, "_backward_handle") and self._backward_handle:
            self._backward_handle.remove()

    def __enter__(self) -> GradCAM:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def __call__(
        self,
        model_input: Tensor | dict[str, Tensor],
        class_index: Optional[int] = None,
    ) -> np.ndarray:
        """Generate normalized 2D Grad-CAM heatmap array [H, W] in range [0, 1]."""
        self.model.zero_grad(set_to_none=True)
        output = self.model(model_input)
        logits = output["cancer_logits"] if isinstance(output, dict) else output
        
        if class_index is None:
            class_index = int(logits.argmax(dim=1)[0])
            
        score = logits[:, class_index].sum()
        score.backward(retain_graph=True)
        
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks failed to capture activations and gradients.")
            
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        
        # Determine target spatial resolution from input tensor
        if isinstance(model_input, dict):
            first_tensor = next(iter(model_input.values()))
            target_size = first_tensor.shape[-2:]
        elif model_input.ndim == 5:
            target_size = model_input.shape[-2:]
        else:
            target_size = model_input.shape[-2:]
            
        cam = torch.nn.functional.interpolate(
            cam, size=target_size, mode="bilinear", align_corners=False
        )
        cam = cam[0, 0].cpu().numpy()
        
        # Normalize heatmap to [0, 1]
        cam_min = cam.min()
        cam_max = cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
            
        return cam


def save_gradcam_outputs(
    image_path: str | Path,
    heatmap: np.ndarray,
    output_heatmap_path: str | Path,
    output_overlay_path: str | Path,
    alpha: float = 0.45,
) -> Tuple[Path, Path]:
    """Save raw heatmap image and blended overlay image to disk."""
    image_path = Path(image_path)
    output_heatmap_path = Path(output_heatmap_path)
    output_overlay_path = Path(output_overlay_path)

    image = Image.open(image_path).convert("RGB")
    h_w_size = image.size # (width, height)

    # Convert heatmap to uint8 array and resize to match original image
    heatmap_norm = np.clip(heatmap, 0, 1)
    raw_heatmap_pil = Image.fromarray(np.uint8(heatmap_norm * 255.0))
    raw_heatmap_resized = raw_heatmap_pil.resize(h_w_size, Image.Resampling.BILINEAR)

    # Save raw grayscale heatmap
    output_heatmap_path.parent.mkdir(parents=True, exist_ok=True)
    raw_heatmap_resized.save(output_heatmap_path)

    # Create RGB Colormap Overlay (Jet-like or Red gradient)
    norm_array = np.asarray(raw_heatmap_resized, dtype=np.float32) / 255.0
    
    # Custom high-contrast red/yellow colormap overlay for pathology heatmaps
    color_map = np.zeros((*norm_array.shape, 3), dtype=np.uint8)
    color_map[..., 0] = np.uint8(255 * norm_array)  # Red channel
    color_map[..., 1] = np.uint8(255 * (norm_array ** 2)) # Yellow highlight in peak zones
    
    overlay = Image.blend(image, Image.fromarray(color_map), alpha=alpha)
    output_overlay_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(output_overlay_path)

    return output_heatmap_path, output_overlay_path


def save_gradcam_overlay(
    image_path: str | Path,
    heatmap: np.ndarray,
    output_path: str | Path,
    alpha: float = 0.45,
) -> None:
    """Backward compatibility helper for saving overlay image."""
    save_gradcam_outputs(
        image_path=image_path,
        heatmap=heatmap,
        output_heatmap_path=Path(output_path).parent / "raw_heatmap.png",
        output_overlay_path=output_path,
        alpha=alpha,
    )