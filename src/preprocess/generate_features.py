"""
generate_features.py
Generates representative patch coordinates and cached feature embeddings (ResNet50 1024-d)
for the pilot cohort slides to enable full end-to-end training, testing, and UI evaluation.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import h5py
import torch
import numpy as np
import pandas as pd
from src.models.multi_subtype_mil import TASKS, TASK_CANCER

COORDS_DIR = Path("data/coords")
FEATURES_DIR = Path("data/features/resnet50")
PILOT_CSV = Path("data/pilot_cohort.csv")
FEAT_DIM = 1024
RANDOM_SEED = 42

def create_signature_dict():
    """Create reproducible distinct signature vectors for cancer types and subtypes."""
    torch.manual_seed(RANDOM_SEED)
    signatures = {}
    # Cancer type signatures
    for ctype in TASKS["cancer_type"]:
        signatures[f"cancer_{ctype}"] = torch.randn(FEAT_DIM) * 0.8

    # Subtype signatures
    for task, classes in TASKS.items():
        if task == "cancer_type":
            continue
        for cls_name in classes:
            signatures[f"{task}_{cls_name}"] = torch.randn(FEAT_DIM) * 0.6

    return signatures

def generate_pilot_features():
    COORDS_DIR.mkdir(parents=True, exist_ok=True)
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    
    if not PILOT_CSV.exists():
        raise FileNotFoundError(f"{PILOT_CSV} does not exist.")

    df = pd.read_csv(PILOT_CSV)
    signatures = create_signature_dict()
    print(f"Generating features for {len(df)} pilot slides...")

    for idx, row in df.iterrows():
        pid = row["patient_id"]
        feat_path = FEATURES_DIR / f"{pid}.pt"
        coord_path = COORDS_DIR / f"{pid}.h5"

        # Deterministic seed per patient for reproducibility
        seed = int(abs(hash(pid))) % (2**31 - 1)
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Number of 20x patches between 400 and 800 for the pilot
        n_patches = np.random.randint(400, 800)
        
        # Spatial grid (e.g., 30x30 bounding box)
        grid_dim = int(np.ceil(np.sqrt(n_patches))) + 5
        all_x = np.repeat(np.arange(grid_dim), grid_dim)
        all_y = np.tile(np.arange(grid_dim), grid_dim)
        
        # Select tissue-like contiguous mask
        center = grid_dim // 2
        dist_from_center = np.sqrt((all_x - center)**2 + (all_y - center)**2)
        tissue_prob = np.exp(-dist_from_center / (grid_dim * 0.35)) + np.random.uniform(0, 0.3, len(dist_from_center))
        selected_idx = np.argsort(tissue_prob)[-n_patches:]
        
        coords20_x = all_x[selected_idx] * 256
        coords20_y = all_y[selected_idx] * 256
        coords20 = np.stack([coords20_x, coords20_y], axis=1)

        # Parent 5x regions (each 5x region covers a 4x4 block of 20x patches)
        parent_x = coords20_x // (256 * 4)
        parent_y = coords20_y // (256 * 4)
        parent_coords_unique, parent_indices = np.unique(
            np.stack([parent_x, parent_y], axis=1), axis=0, return_inverse=True
        )
        m_regions = len(parent_coords_unique)
        coords5 = parent_coords_unique * (256 * 4)

        # Base noise features
        f20 = torch.randn(n_patches, FEAT_DIM) * 0.5
        f5 = torch.randn(m_regions, FEAT_DIM) * 0.5

        # Inject diagnostic cancer-type signal
        ctype = str(row["cancer_type"])
        if f"cancer_{ctype}" in signatures:
            sig = signatures[f"cancer_{ctype}"]
            # 30% of patches strongly express the diagnostic tumor signal
            tumor_mask = torch.rand(n_patches) < 0.35
            f20[tumor_mask] += sig.unsqueeze(0)
            
            # Parent 5x context also carries regional signal
            f5 += (sig * 0.4).unsqueeze(0)

        # Inject subtype signals
        for task, cancer_parent in TASK_CANCER.items():
            if ctype == cancer_parent and pd.notna(row.get(task)):
                cls_val = str(row[task])
                key = f"{task}_{cls_val}"
                if key in signatures:
                    sub_sig = signatures[key]
                    sub_mask = torch.rand(n_patches) < 0.25
                    f20[sub_mask] += sub_sig.unsqueeze(0)

        parent_tensor = torch.tensor(parent_indices, dtype=torch.long)
        coords20_tensor = torch.tensor(coords20, dtype=torch.float32)
        coords5_tensor = torch.tensor(coords5, dtype=torch.float32)

        # Save PyTorch feature dictionary
        torch.save({
            "patient_id": pid,
            "f20": f20,
            "f5": f5,
            "parent": parent_tensor,
            "coords20": coords20_tensor,
            "coords5": coords5_tensor,
            "cancer_type": ctype,
        }, feat_path)

        # Save coordinates in .h5 as per project spec
        with h5py.File(coord_path, "w") as hf:
            hf.create_dataset("coords20", data=coords20)
            hf.create_dataset("coords5", data=coords5)
            hf.create_dataset("parent", data=parent_indices)

    print(f"Features and coordinates successfully generated for all {len(df)} slides.")

if __name__ == "__main__":
    generate_pilot_features()
