"""
heatmaps.py
Explainability module:
- Maps task-specific attention weights to spatial 2D patch coordinates.
- Applies percentile normalization and renders 2D spatial attention heatmaps.
- Extracts top-K attended patches for patch gallery inspection.
"""

import os
import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.models.multi_subtype_mil import MultiSubtypeMIL, TASKS

HEATMAPS_DIR = Path("reports/heatmaps")

def generate_slide_heatmap(
    model,
    slide_data,
    task="cancer_type",
    save_path=None,
    percentile=98.0,
):
    HEATMAPS_DIR.mkdir(parents=True, exist_ok=True)
    model.eval()

    f20 = slide_data["f20"]
    f5 = slide_data["f5"]
    parent = slide_data["parent"]
    coords20 = slide_data["coords20"]

    with torch.no_grad():
        logits, attn_dict = model(f20, f5, parent)

    if task not in attn_dict:
        raise ValueError(f"Task {task} not found in model attention outputs.")

    attn = attn_dict[task].cpu().numpy()  # [N]
    coords = coords20.cpu().numpy()       # [N, 2]

    # Percentile normalization
    p_high = np.percentile(attn, percentile)
    norm_attn = np.clip(attn / (p_high + 1e-8), 0.0, 1.0)

    # Plot spatial heatmap
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=norm_attn,
        cmap="inferno",
        s=40,
        edgecolors="none",
        alpha=0.85,
    )
    ax.invert_yaxis()  # Match pathology image coordinate conventions (top-left is (0,0))
    ax.set_title(f"Task Attention Heatmap: {task.upper()} ({slide_data.get('patient_id', 'Slide')})")
    ax.set_xlabel("X (microns/pixels)")
    ax.set_ylabel("Y (microns/pixels)")
    plt.colorbar(sc, ax=ax, label="Normalized Attention Weight")
    plt.tight_layout()

    if save_path is None:
        pid = slide_data.get("patient_id", "sample")
        save_path = HEATMAPS_DIR / f"{pid}_{task}_heatmap.png"

    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved heatmap to: {save_path}")

    # Top-K attended patches
    top_indices = np.argsort(attn)[::-1][:16]
    top_patches = coords[top_indices]

    return {
        "save_path": str(save_path),
        "top_indices": top_indices,
        "top_coords": top_patches,
        "norm_attn": norm_attn,
    }

if __name__ == "__main__":
    # Test on one pilot slide
    checkpoint_path = Path("checkpoints/best_multi_subtype_mil_fold0.pt")
    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, weights_only=False)
        model = MultiSubtypeMIL(in_dim=1024)
        model.load_state_dict(ckpt["model_state_dict"])
        
        # Load sample slide
        feat_files = list(Path("data/features/resnet50").glob("*.pt"))
        if feat_files:
            slide_data = torch.load(feat_files[0], weights_only=False)
            res = generate_slide_heatmap(model, slide_data, task="cancer_type")
            print("Heatmap generation test complete.")
