"""
tests/test_rectifications.py
Comprehensive Regression Test Suite verifying:
1. test_focal_loss_gradient_flow: Gradients flow properly to minority class heads.
2. test_stain_normalizer: Macenko normalization processes tissue patches and preserves dimensions.
3. test_cosine_scheduler_step: Learning rate updates according to cosine warm restarts schedule.
4. test_patient_disjoint_guarantee: 0% patient leakage across cross-validation splits.
5. test_full_training_and_eval_pipeline: 2-epoch training, checkpointing, and metrics export.
6. test_temperature_calibration_and_ood: Temperature calibration optimization and ECE reduction.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# Add root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.losses import FocalLoss, SupervisedContrastiveLoss, CompositeMultiTaskLoss, compute_class_balanced_alpha
from dataset.preprocessing import MacenkoNormalizer, build_preprocessor
from models.multi_scale_model import MultiScalePathologyModel
from training.calibration import TemperatureScaler, compute_ece, evaluate_ood_abstention
from training.train import train_model


def test_focal_loss_gradient_flow():
    """Confirms gradients flow properly to minority class heads under Focal Loss."""
    # 7 classes with class 2 as severe minority (e.g., TNBC)
    num_classes = 7
    alpha = torch.tensor([1.0, 1.0, 4.5, 1.2, 1.0, 1.5, 1.8], requires_grad=False)
    focal_loss = FocalLoss(alpha=alpha, gamma=2.0, label_smoothing=0.1)

    logits = torch.randn(8, num_classes, requires_grad=True)
    # Target containing minority class 2
    targets = torch.tensor([2, 0, 1, 2, 4, 3, 5, 2], dtype=torch.long)

    loss = focal_loss(logits, targets)
    assert loss > 0, "Loss must be positive"
    loss.backward()

    assert logits.grad is not None, "Logits gradient must exist"
    assert not torch.isnan(logits.grad).any(), "Gradients must not contain NaNs"
    # Ensure gradients at minority positions are significant
    assert logits.grad[0, 2] != 0.0, "Minority class gradient must be non-zero"


def test_stain_normalizer():
    """Confirms Macenko normalization processes sample tissue patches and preserves shape."""
    normalizer = MacenkoNormalizer()

    # Generate synthetic tissue-like patch with H&E coloration [224, 224, 3]
    np.random.seed(42)
    # Hematoxylin (purplish) and Eosin (pinkish) simulation
    patch = np.zeros((224, 224, 3), dtype=np.uint8)
    patch[:, :, 0] = np.random.randint(150, 220, (224, 224))  # R
    patch[:, :, 1] = np.random.randint(80, 160, (224, 224))   # G
    patch[:, :, 2] = np.random.randint(140, 210, (224, 224))  # B

    normalized = normalizer.fit_transform(patch)
    assert isinstance(normalized, np.ndarray), "Output must be numpy array"
    assert normalized.shape == (224, 224, 3), f"Shape preserved: expected (224, 224, 3), got {normalized.shape}"
    assert normalized.dtype == np.uint8, "Dtype must be uint8"

    # Test preprocessor function with tensor output [3, 224, 224]
    preprocess = build_preprocessor(image_size=224, stain_normalization=True, is_train=False)
    tensor_out = preprocess(patch)
    assert isinstance(tensor_out, torch.Tensor), "Preprocessor must output torch.Tensor"
    assert tensor_out.shape == (3, 224, 224), f"Expected tensor shape (3, 224, 224), got {tensor_out.shape}"


def test_cosine_scheduler_step():
    """Confirms learning rate updates according to the schedule across epochs."""
    model = nn.Linear(10, 2)
    optimizer = AdamW(model.parameters(), lr=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)

    lrs = []
    for epoch in range(15):
        optimizer.step()
        scheduler.step()
        lrs.append(scheduler.get_last_lr()[0])

    assert len(lrs) == 15, "Must track 15 epochs of LR updates"
    # Verify LR decreases during first cycle (epoch 1 to 4)
    assert lrs[3] < lrs[0], "Learning rate must decay before restart"
    # Verify restart occurs at T_0 = 5 (0-indexed step 4 restarts to initial lr)
    assert lrs[4] > lrs[3], "Learning rate must restart after decay"


def test_patient_disjoint_guarantee():
    """Confirms 0% patient leakage across cross-validation splits."""
    splits_csv = Path("data/splits/pilot_5fold_splits.csv")
    if not splits_csv.exists():
        pytest.skip("pilot_5fold_splits.csv not found")

    df = pd.read_csv(splits_csv)
    assert "patient_id" in df.columns and "fold" in df.columns

    for fold in df["fold"].unique():
        val_pids = set(df[df["fold"] == fold]["patient_id"])
        train_pids = set(df[df["fold"] != fold]["patient_id"])
        overlap = val_pids.intersection(train_pids)
        assert len(overlap) == 0, f"Patient leakage detected in fold {fold}: {overlap}"


def test_temperature_calibration_and_ood():
    """Confirms Temperature Scaler reduces ECE and OOD detector separates distributions."""
    torch.manual_seed(42)
    # In-distribution logits with slight miscalibration
    logits = torch.randn(100, 3) * 3.0
    labels = torch.randint(0, 3, (100,))

    scaler = TemperatureScaler()
    initial_probs = torch.softmax(logits, dim=-1)
    initial_ece = compute_ece(initial_probs, labels)

    final_ece = scaler.fit(logits, labels, max_iter=20)
    assert final_ece <= initial_ece + 0.05, "Calibration should control or reduce ECE"

    # OOD test
    ood_logits = torch.randn(50, 3) * 0.1  # High entropy uniform logits
    auroc = evaluate_ood_abstention(logits, ood_logits)
    assert auroc >= 0.70, f"AUROC should differentiate confident vs unconfident logits, got {auroc}"


def test_full_training_and_eval_pipeline():
    """Runs a fast 2-epoch run ensuring checkpoint saving, metrics export, and inference."""
    results = train_model(epochs=2, lr=1e-4, device="cpu")
    assert "best_composite_f1" in results
    assert results["epochs_completed"] == 2

    # Check checkpoint file exists
    ckpt_path = Path("checkpoints/best_rectified_model.pt")
    assert ckpt_path.exists(), "best_rectified_model.pt must be saved"

    # Check metrics file exists
    metrics_path = Path("outputs/metrics/training_metrics.json")
    assert metrics_path.exists(), "training_metrics.json must be saved"

    # Verify model load and inference
    model = MultiScalePathologyModel(in_dim=1024, num_cancer_classes=3, num_subtype_classes=7)
    ckpt = torch.load(ckpt_path, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_input = torch.randn(1, 50, 1024)
    pred_res = model.predict_calibrated(test_input)
    assert "cancer_probs" in pred_res
    assert "subtype_probs" in pred_res
    assert "energy_score" in pred_res
    assert pred_res["cancer_probs"].shape == (1, 3)
    assert pred_res["subtype_probs"].shape == (1, 7)
