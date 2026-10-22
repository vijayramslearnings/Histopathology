"""
train_mil.py
Training engine for Multi-Task Cross-Scale MIL and baseline models.
Features:
- AdamW optimizer with cosine learning rate schedule
- Gradient accumulation
- Masked multi-task loss with training-fold class weights
- Patient-level validation and early stopping
- Saves best checkpoint and comprehensive evaluation metrics
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.multi_subtype_mil import MultiSubtypeMIL, masked_multitask_loss, class_weights_from_counts, TASKS
from src.models.baselines import MeanPoolingMIL, MaxPoolingMIL, ABMIL, TransMIL
from src.train.dataset import WSISubtypeDataset
from src.train.eval import evaluate_predictions

CHECKPOINTS_DIR = Path("checkpoints")
REPORTS_DIR = Path("reports")

def build_model(model_type="multi_subtype_mil", in_dim=1024, dim=256):
    if model_type == "multi_subtype_mil":
        return MultiSubtypeMIL(in_dim=in_dim, dim=dim)
    elif model_type == "mean_pooling":
        return MeanPoolingMIL(in_dim=in_dim, dim=dim)
    elif model_type == "max_pooling":
        return MaxPoolingMIL(in_dim=in_dim, dim=dim)
    elif model_type == "abmil":
        return ABMIL(in_dim=in_dim, dim=dim)
    elif model_type == "transmil":
        return TransMIL(in_dim=in_dim, dim=dim)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

def compute_fold_class_weights(dataset):
    counts = {t: [0] * len(classes) for t, classes in TASKS.items()}
    for item in dataset:
        for t, y in item["labels"].items():
            if y >= 0:
                counts[t][y] += 1
    return class_weights_from_counts(counts)

def train_one_fold(
    fold=0,
    model_type="multi_subtype_mil",
    cohort_csv="data/pilot_cohort.csv",
    features_dir="data/features/resnet50",
    epochs=15,
    lr=2e-4,
    grad_accum_steps=8,
    patience=5,
    seed=42,
    device="cpu",
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n==========================================")
    print(f"Training {model_type.upper()} on Fold {fold} (Device: {device})")
    print(f"==========================================")

    train_ds = WSISubtypeDataset(cohort_csv, features_dir, fold=fold, is_train=True)
    val_ds = WSISubtypeDataset(cohort_csv, features_dir, fold=fold, is_train=False)

    print(f"Train slides: {len(train_ds)}, Val slides: {len(val_ds)}")

    class_weights = compute_fold_class_weights(train_ds)
    # Move class weights to device
    class_weights = {t: w.to(device) for t, w in class_weights.items()}

    model = build_model(model_type).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_loss = float("inf")
    patience_counter = 0
    best_checkpoint_path = CHECKPOINTS_DIR / f"best_{model_type}_fold{fold}.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        total_train_loss = 0.0

        perm = np.random.permutation(len(train_ds))
        for step_idx, sample_i in enumerate(perm):
            sample = train_ds[sample_i]
            f20 = sample["f20"].to(device)
            f5 = sample["f5"].to(device)
            parent = sample["parent"].to(device)
            labels = sample["labels"]

            logits, _ = model(f20, f5, parent)
            loss, _ = masked_multitask_loss(logits, labels, class_weights=class_weights)
            
            scaled_loss = loss / grad_accum_steps
            scaled_loss.backward()
            total_train_loss += loss.item()

            if (step_idx + 1) % grad_accum_steps == 0 or (step_idx + 1) == len(train_ds):
                optimizer.step()
                optimizer.zero_grad()

        scheduler.step()
        avg_train_loss = total_train_loss / len(train_ds)

        # Validation
        model.eval()
        total_val_loss = 0.0
        val_preds = {t: [] for t in TASKS}
        val_targets = {t: [] for t in TASKS}

        with torch.no_grad():
            for sample in val_ds:
                f20 = sample["f20"].to(device)
                f5 = sample["f5"].to(device)
                parent = sample["parent"].to(device)
                labels = sample["labels"]

                logits, _ = model(f20, f5, parent)
                loss, _ = masked_multitask_loss(logits, labels, class_weights=class_weights)
                total_val_loss += loss.item()

                for t, l in logits.items():
                    probs = F.softmax(l, dim=-1).cpu().numpy()
                    val_preds[t].append(probs)
                    val_targets[t].append(labels[t])

        avg_val_loss = total_val_loss / len(val_ds)
        val_metrics = evaluate_predictions(val_preds, val_targets)
        cancer_auc = val_metrics.get("cancer_type", {}).get("roc_auc", 0.0)

        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Cancer AUC: {cancer_auc:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "val_metrics": val_metrics,
                "model_type": model_type,
                "fold": fold,
            }, best_checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

    # Load best model for final evaluation report
    best_data = torch.load(best_checkpoint_path, weights_only=False)
    metrics_file = REPORTS_DIR / f"metrics_{model_type}_fold{fold}.json"
    with open(metrics_file, "w") as f:
        json.dump(best_data["val_metrics"], f, indent=2)

    print(f"\nBest Model Saved to: {best_checkpoint_path}")
    print(f"Metrics saved to: {metrics_file}")
    return best_data["val_metrics"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--model", type=str, default="multi_subtype_mil")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    train_one_fold(fold=args.fold, model_type=args.model, epochs=args.epochs, lr=args.lr)
