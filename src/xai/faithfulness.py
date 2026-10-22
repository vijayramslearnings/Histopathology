"""
faithfulness.py
Faithfulness validation of attention explanations:
- Deletion curve: progressively removes top-attended patches and measures drop in predicted class probability.
- Compared against random-order patch deletion.
- Produces faithfulness curve plot in reports/faithfulness_deletion_curve.png.
"""

import os
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.models.multi_subtype_mil import MultiSubtypeMIL, TASKS

REPORTS_DIR = Path("reports")

def run_faithfulness_test(model, slide_data, task="cancer_type", n_steps=10):
    model.eval()
    f20 = slide_data["f20"]
    f5 = slide_data["f5"]
    parent = slide_data["parent"]
    n_patches = f20.shape[0]

    with torch.no_grad():
        logits, attn_dict = model(f20, f5, parent)
        orig_probs = F.softmax(logits[task], dim=-1)
        target_class = torch.argmax(orig_probs).item()
        orig_conf = orig_probs[target_class].item()

    attn = attn_dict[task].cpu().numpy()
    sorted_indices_attn = np.argsort(attn)[::-1].copy()  # Most important first
    random_indices = np.random.permutation(n_patches).copy()

    drop_fractions = np.linspace(0.0, 0.8, n_steps)
    attn_probs = [orig_conf]
    rand_probs = [orig_conf]

    with torch.no_grad():
        for frac in drop_fractions[1:]:
            k = int(n_patches * frac)
            keep_attn_idx = sorted_indices_attn[k:]
            keep_rand_idx = random_indices[k:]

            # 1. Attended deletion
            f20_attn = f20[keep_attn_idx]
            parent_attn = parent[keep_attn_idx]
            l_attn, _ = model(f20_attn, f5, parent_attn)
            p_attn = F.softmax(l_attn[task], dim=-1)[target_class].item()
            attn_probs.append(p_attn)

            # 2. Random deletion
            f20_rand = f20[keep_rand_idx]
            parent_rand = parent[keep_rand_idx]
            l_rand, _ = model(f20_rand, f5, parent_rand)
            p_rand = F.softmax(l_rand[task], dim=-1)[target_class].item()
            rand_probs.append(p_rand)

    # Plot deletion curves
    fig, ax = plt.subplots(figsize=(7, 5))
    pct_dropped = drop_fractions * 100
    ax.plot(pct_dropped, attn_probs, "r-o", label="Attention-Guided Deletion (Top Patches)", linewidth=2)
    ax.plot(pct_dropped, rand_probs, "b--s", label="Random Deletion (Baseline)", linewidth=2)
    ax.set_xlabel("% Patches Removed")
    task_dict = getattr(model, "tasks", TASKS)
    class_name = task_dict[task][target_class] if (task in task_dict and target_class < len(task_dict[task])) else str(target_class)
    ax.set_ylabel(f"Predicted Probability ({class_name})")
    ax.set_title(f"Attention Faithfulness Test (Deletion Curve) - {task}")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    plt.tight_layout()

    out_plot = REPORTS_DIR / "faithfulness_deletion_curve.png"
    plt.savefig(out_plot, dpi=200)
    plt.close()
    print(f"Faithfulness deletion curve plot saved to: {out_plot}")

    return {
        "drop_fractions": drop_fractions.tolist(),
        "attn_probs": attn_probs,
        "rand_probs": rand_probs,
        "plot_path": str(out_plot),
    }

if __name__ == "__main__":
    checkpoint_path = Path("checkpoints/best_multi_subtype_mil_fold0.pt")
    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, weights_only=False)
        model = MultiSubtypeMIL(in_dim=1024)
        model.load_state_dict(ckpt["model_state_dict"])
        
        feat_files = list(Path("data/features/resnet50").glob("*.pt"))
        if feat_files:
            slide_data = torch.load(feat_files[0], weights_only=False)
            res = run_faithfulness_test(model, slide_data, task="cancer_type")
            print("Faithfulness evaluation test complete.")
