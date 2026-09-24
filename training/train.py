"""CLI for baseline, proposed-model, and ablation training."""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

try:
    from dataset.dataset import (
        HistopathologyDataset,
        build_label_maps,
        load_metadata,
    )
    from dataset.preprocessing import BaselinePreprocessor, build_preprocessor
    from dataset.splitting import make_patient_splits
    from models.backbone import ResNetBaseline
    from models.multi_scale_model import MultiScaleAttentionNetwork
    from training.evaluator import evaluate_model, save_metrics
    from training.losses import MultiTaskLoss
except ImportError:
    from backend.dataset.dataset import (
        HistopathologyDataset,
        build_label_maps,
        load_metadata,
    )
    from backend.dataset.preprocessing import BaselinePreprocessor, build_preprocessor
    from backend.dataset.splitting import make_patient_splits
    from backend.models.backbone import ResNetBaseline
    from backend.models.multi_scale_model import MultiScaleAttentionNetwork
    from backend.training.evaluator import evaluate_model, save_metrics
    from backend.training.losses import MultiTaskLoss


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(config: dict[str, Any], labels: Any, model_name: str) -> nn.Module:
    model_config = config["model"]
    subtype_specs = {
        cname: len(smap)
        for cname, smap in (labels.subtype_specs or {}).items()
    }
    common = {
        "cancer_classes": len(labels.cancer_to_index),
        "subtype_classes": subtype_specs if subtype_specs else len(labels.subtype_to_index),
        "pretrained": bool(model_config.get("pretrained", True)),
        "hidden_dim": int(model_config.get("hidden_dim", 512)),
        "dropout": float(model_config.get("dropout", 0.3)),
    }
    if model_name == "baseline":
        return ResNetBaseline(**common)
    return MultiScaleAttentionNetwork(
        **common,
        fusion_dim=int(model_config.get("fusion_dim", 512)),
        use_scale_attention=bool(model_config.get("use_scale_attention", True)),
        use_patch_attention=bool(model_config.get("use_patch_attention", True)),
        use_mil=bool(model_config.get("use_mil", True)),
        shared_backbone=bool(model_config.get("shared_backbone", False)),
        patch_grid=int(model_config.get("patch_grid", 4)),
    )


def _make_loaders(
    splits: dict[str, Any],
    labels: Any,
    config: dict[str, Any],
    model_name: str,
) -> dict[str, DataLoader]:
    data_config = config["data"]
    image_size = int(data_config.get("image_size", 224))
    fractions = tuple(data_config.get("scale_fractions", [1.0, 0.75, 0.55]))
    datasets = {}
    for split_name, metadata in splits.items():
        if split_name == "ood_test":
            continue
        train = split_name == "train"
        processor = (
            BaselinePreprocessor(
                build_preprocessor(image_size, train, fractions).config
            )
            if model_name == "baseline"
            else build_preprocessor(image_size, train, fractions)
        )
        datasets[split_name] = HistopathologyDataset(
            metadata, labels, processor, model_name=model_name
        )
    loader_config = config["training"]
    return {
        name: DataLoader(
            dataset,
            batch_size=int(loader_config.get("batch_size", 8)),
            shuffle=name == "train",
            num_workers=int(loader_config.get("num_workers", 0)),
            pin_memory=torch.cuda.is_available(),
        )
        for name, dataset in datasets.items()
    }


def plot_learning_curves(
    history: list[dict[str, float]],
    model_name: str,
    ablation_name: str,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = [h["epoch"] for h in history]
    train_loss = [h["total"] for h in history]
    val_acc = [h["val_cancer_accuracy"] for h in history]

    # Plot loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label="Train Total Loss", color="#0284c7", linewidth=2)
    plt.title(f"Training Loss Curve ({model_name}_{ablation_name})", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    loss_path = output_dir / f"loss_curve_{model_name}_{ablation_name}.png"
    plt.savefig(loss_path, dpi=300)
    plt.close()

    # Plot accuracy curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, val_acc, label="Val Cancer Accuracy", color="#16a34a", linewidth=2)
    plt.title(f"Validation Accuracy Curve ({model_name}_{ablation_name})", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    acc_path = output_dir / f"accuracy_curve_{model_name}_{ablation_name}.png"
    plt.savefig(acc_path, dpi=300)
    plt.close()


def _run_one(
    config: dict[str, Any],
    model_name: str,
    ablation_name: str,
    output_dir: Path,
    device: torch.device,
) -> None:
    data_config = config["data"]
    metadata_path = data_config.get("metadata_path", "data/metadata.csv")
    if not Path(metadata_path).exists() and Path("data/metadata.csv").exists():
        metadata_path = "data/metadata.csv"

    metadata = load_metadata(metadata_path)
    splits = make_patient_splits(
        metadata,
        train_ratio=float(data_config.get("train_ratio", 0.7)),
        val_ratio=float(data_config.get("val_ratio", 0.15)),
        test_ratio=float(data_config.get("test_ratio", 0.15)),
        seed=int(config.get("seed", 42)),
    )
    labels = build_label_maps(metadata, data_config.get("cancer_mapping"))
    loaders = _make_loaders(splits, labels, config, model_name)
    model = build_model(config, labels, model_name).to(device)
    loss_fn = MultiTaskLoss(subtype_weight=float(config["training"].get("subtype_loss_weight", 1.0)))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"].get("learning_rate", 1e-4)),
        weight_decay=float(config["training"].get("weight_decay", 1e-4)),
    )
    epochs = int(config["training"].get("epochs", 10))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        running = {"total": 0.0, "cancer": 0.0, "subtype": 0.0}
        batches = 0
        for batch in loaders["train"]:
            inputs = batch["image"].to(device)
            outputs = model(inputs)

            # Move subtype targets dict to device if available
            sub_target_dict = None
            if "subtype_targets" in batch and isinstance(batch["subtype_targets"], dict):
                sub_target_dict = {
                    k: v.to(device) for k, v in batch["subtype_targets"].items()
                }

            losses = loss_fn(
                cancer_logits=outputs["cancer_logits"],
                cancer_targets=batch["cancer_label"].to(device),
                subtype_logits=outputs.get("subtype_logits"),
                subtype_targets=batch["subtype_label"].to(device),
                subtype_dict=outputs.get("subtype_dict"),
                cancer_names=labels.cancer_classes,
                subtype_target_dict=sub_target_dict,
            )

            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            optimizer.step()

            for key in running:
                running[key] += float(losses[key].detach().cpu())
            batches += 1

        scheduler.step()
        epoch_losses = {key: value / max(1, batches) for key, value in running.items()}
        val_metrics = evaluate_model(
            model, loaders["val"], device, labels.cancer_classes, labels.subtype_classes
        )
        val_score = 1.0 - val_metrics["cancer"]["macro_f1"]

        history.append(
            {
                "epoch": float(epoch),
                **epoch_losses,
                "val_cancer_accuracy": val_metrics["cancer"]["accuracy"],
                "val_cancer_macro_f1": val_metrics["cancer"]["macro_f1"],
            }
        )
        print(json.dumps({"epoch": epoch, "train": epoch_losses, "val": val_metrics["cancer"]}))

        if val_score < best_val:
            best_val = val_score
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            checkpoint = {
                "model_state": best_state,
                "model_config": {
                    "name": model_name,
                    **config["model"],
                    **(
                        {
                            "use_scale_attention": not ablation_name.endswith("no_scale_attention"),
                            "use_patch_attention": not ablation_name.endswith("no_patch_attention"),
                            "use_mil": not ablation_name.endswith("no_mil"),
                        }
                        if model_name == "multiscale"
                        else {}
                    ),
                },
                "cancer_classes": labels.cancer_classes,
                "subtype_classes": labels.subtype_classes,
                "subtype_specs": labels.subtype_specs,
                "image_size": int(data_config.get("image_size", 224)),
                "scale_fractions": list(data_config.get("scale_fractions", [1.0, 0.75, 0.55])),
                "history": history,
            }
            checkpoint_dir = Path("checkpoints")
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_name = (
                "resnet50_baseline.pt"
                if model_name == "baseline" and ablation_name == "full"
                else f"{model_name}_{ablation_name}.pt"
            )
            torch.save(checkpoint, checkpoint_dir / checkpoint_name)
            # Also save canonical best_model.pt
            if model_name == "multiscale" and ablation_name == "full":
                torch.save(checkpoint, checkpoint_dir / "best_model.pt")

    # Save learning curves
    plot_learning_curves(history, model_name, ablation_name, output_dir)

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate_model(
        model, loaders["test"], device, labels.cancer_classes, labels.subtype_classes
    )
    test_metrics["training"] = {
        "model": model_name,
        "ablation": ablation_name,
        "patient_counts": {name: int(frame["patient_id"].nunique()) for name, frame in splits.items() if name != "ood_test"},
    }
    metrics_dir = output_dir / "outputs" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    save_metrics(test_metrics, metrics_dir / f"{model_name}_{ablation_name}.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model", choices=("baseline", "multiscale"), default="multiscale")
    parser.add_argument("--ablation", default="full", help="full or comma-separated ablations.")
    parser.add_argument("--device", default=None, help="cpu, cuda, or auto.")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists() and Path("config.yaml").exists():
        config_path = Path("config.yaml")

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seed_everything(int(config.get("seed", 42)))
    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)

    ablations = [item.strip() for item in args.ablation.split(",") if item.strip()]
    for ablation in ablations:
        run_config = copy.deepcopy(config)
        model_name = args.model
        if model_name == "multiscale":
            if ablation == "no_scale_attention":
                run_config["model"]["use_scale_attention"] = False
            elif ablation == "no_patch_attention":
                run_config["model"]["use_patch_attention"] = False
            elif ablation == "no_mil":
                run_config["model"]["use_mil"] = False
            elif ablation != "full":
                raise ValueError(
                    "Ablations must be full, no_scale_attention, no_patch_attention, or no_mil."
                )
        elif ablation != "full":
            raise ValueError("Ablations are currently supported for the multiscale model.")
        
        output_root = Path(config.get("outputs", {}).get("root", "."))
        _run_one(
            run_config,
            model_name,
            ablation,
            output_root,
            device,
        )


if __name__ == "__main__":
    main()