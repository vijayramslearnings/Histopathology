"""
dataset.py
PyTorch Dataset for multi-task WSI features.
Loads cached features (f20, f5, parent, coords20, coords5) and builds task label targets.
Supports random patch dropout (50-100% of patches during training).
"""

import os
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.models.multi_subtype_mil import TASKS

class WSISubtypeDataset(Dataset):
    def __init__(
        self,
        cohort_csv: str or Path,
        features_dir: str or Path,
        fold: int = None,
        is_train: bool = True,
        patch_dropout_min: float = 0.5,
        patch_dropout_max: float = 1.0,
        tasks: dict = None,
    ):
        self.features_dir = Path(features_dir)
        self.is_train = is_train
        self.patch_dropout_min = patch_dropout_min
        self.patch_dropout_max = patch_dropout_max
        self.tasks = tasks if tasks is not None else TASKS

        df = pd.read_csv(cohort_csv)
        if fold is not None:
            if is_train:
                self.df = df[df["fold"] != fold].reset_index(drop=True)
            else:
                self.df = df[df["fold"] == fold].reset_index(drop=True)
        else:
            self.df = df.reset_index(drop=True)

        self.class_maps = {
            task: {cls_name: idx for idx, cls_name in enumerate(classes)}
            for task, classes in self.tasks.items()
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pid = row["patient_id"]
        feat_path = self.features_dir / f"{pid}.pt"

        if not feat_path.exists():
            raise FileNotFoundError(f"Feature file {feat_path} does not exist.")

        data = torch.load(feat_path, weights_only=False)
        f20 = data["f20"]
        f5 = data["f5"]
        parent = data["parent"]
        coords20 = data.get("coords20", None)

        # Multi-scale bag augmentation during training to prevent overfitting and handle small external uploads
        if self.is_train:
            n_patches = len(f20)
            if n_patches > 4:
                # Sample bag size across multiple regimes:
                # 35% micro-biopsy (4-12 patches), 35% medium (12-32 patches), 30% full/large
                regime = np.random.rand()
                if regime < 0.35 and n_patches >= 6:
                    k = np.random.randint(4, min(13, n_patches + 1))
                elif regime < 0.70 and n_patches >= 14:
                    k = np.random.randint(12, min(33, n_patches + 1))
                else:
                    k = min(n_patches, np.random.randint(20, min(96, n_patches + 1))) if n_patches > 20 else n_patches

                perm = torch.randperm(n_patches)[:k]
                f20 = f20[perm]
                
                # Remap regional parent context
                if parent.numel() > 0:
                    unique_p, new_parent = torch.unique(parent[perm], return_inverse=True)
                    f5 = f5[unique_p]
                    parent = new_parent
                if coords20 is not None:
                    coords20 = coords20[perm]

            # 1. Feature Gaussian Noise (Stain, sensor, and optical invariance)
            f20 = f20 + torch.randn_like(f20) * 0.04
            f5 = f5 + torch.randn_like(f5) * 0.04

            # 2. Feature Channel Dropout (Prevents co-adaptation to narrow ResNet channels)
            f20 = F.dropout(f20, p=0.15, training=True)
            f5 = F.dropout(f5, p=0.15, training=True)

        # Extract labels
        labels = {}
        for task, class_map in self.class_maps.items():
            val = row.get(task)
            if pd.notna(val) and str(val) in class_map:
                labels[task] = class_map[str(val)]
            else:
                labels[task] = -1

        return {
            "patient_id": pid,
            "f20": f20,
            "f5": f5,
            "parent": parent,
            "coords20": coords20,
            "labels": labels,
        }
