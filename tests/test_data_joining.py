"""
Unit tests for data joining and split leakage.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path


def test_master_labels_structure():
    master_path = Path("data/labels/master_labels.csv")
    assert master_path.exists(), "master_labels.csv must exist."
    df = pd.read_csv(master_path)
    
    # Assert one slide per patient (DX1)
    assert len(df) == df["patient_id"].nunique(), "Each row must be a unique patient."
    
    # Assert 12-character barcode format TCGA-XX-XXXX
    for pid in df["patient_id"].sample(min(100, len(df)), random_state=42):
        assert len(pid) == 12, f"Patient ID {pid} length is not 12."
        assert pid.startswith("TCGA-"), f"Patient ID {pid} does not start with TCGA-."
        
    # Check TSS length
    assert (df["tss"].str.len() == 2).all(), "All TSS codes must be 2 characters."


def test_split_leakage_check():
    """
    Ensure no patient overlaps between train and validation folds.
    """
    splits_dir = Path("data/splits")
    if not splits_dir.exists():
        return
        
    split_files = list(splits_dir.glob("*.csv"))
    for sf in split_files:
        df = pd.read_csv(sf)
        if "patient_id" in df.columns and "fold" in df.columns:
            for fold in df["fold"].unique():
                val_pids = set(df[df["fold"] == fold]["patient_id"])
                train_pids = set(df[df["fold"] != fold]["patient_id"])
                overlap = val_pids.intersection(train_pids)
                assert len(overlap) == 0, f"Patient leakage found in fold {fold}: {overlap}"


def test_multi_subtype_mil_forward():
    import torch
    from mainmodel import MultiSubtypeMIL, masked_multitask_loss

    model = MultiSubtypeMIL(in_dim=1024, dim=128)
    f20 = torch.randn(100, 1024)
    f5 = torch.randn(20, 1024)
    parent = torch.randint(0, 20, (100,))
    
    logits, attn = model(f20, f5, parent)
    assert "cancer_type" in logits
    assert logits["cancer_type"].shape[-1] in [5, 10]
    assert attn["cancer_type"].shape == (100,)
    
    # Test loss
    labels = {"cancer_type": 0, "brca_histology": 1}
    loss, parts = masked_multitask_loss(logits, labels)
    assert loss > 0
    assert "cancer_type" in parts
    assert "brca_histology" in parts
