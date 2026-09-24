"""Reusable single-image inference helpers for histopathology research models."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional, Dict, List

import torch
from PIL import Image

from dataset.preprocessing import BaselinePreprocessor, build_preprocessor
from models.backbone import ResNetBaseline
from models.multi_scale_model import MultiScaleAttentionNetwork

# Default Organ-Specific Subtype Mappings
ORGAN_SUBTYPES: Dict[str, List[str]] = {
    "breast": [
        "Invasive Ductal Carcinoma (IDC)",
        "Invasive Lobular Carcinoma (ILC)",
        "Mucinous Carcinoma",
        "Normal / Benign Tissue",
    ],
    "lung": [
        "Lung Adenocarcinoma (LUAD)",
        "Lung Squamous Cell Carcinoma (LUSC)",
        "Small Cell Lung Carcinoma (SCLC)",
        "Normal / Benign Tissue",
    ],
    "brain": [
        "Glioblastoma Multiforme (GBM)",
        "Lower Grade Glioma (LGG)",
        "Meningioma",
        "Normal / Benign Tissue",
    ],
}


def load_checkpoint(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load PyTorch checkpoint and instantiate corresponding model architecture."""
    device = torch.device(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model_config = checkpoint["model_config"]
    subtype_specs = checkpoint.get("subtype_specs", {"BRCA": 3, "LUAD": 2, "GBM": 2})

    sub_cls_count = len(checkpoint.get("subtype_classes", []))
    if model_config["name"] == "baseline":
        model = ResNetBaseline(
            cancer_classes=len(checkpoint["cancer_classes"]),
            subtype_classes=sub_cls_count,
            subtype_specs=subtype_specs,
            pretrained=False,
            hidden_dim=model_config.get("hidden_dim", 512),
            dropout=model_config.get("dropout", 0.3),
        )
    elif model_config["name"] == "multiscale":
        model = MultiScaleAttentionNetwork(
            cancer_classes=len(checkpoint["cancer_classes"]),
            subtype_classes=sub_cls_count,
            subtype_specs=subtype_specs,
            pretrained=False,
            fusion_dim=model_config.get("fusion_dim", 512),
            hidden_dim=model_config.get("hidden_dim", 512),
            dropout=model_config.get("dropout", 0.3),
            use_scale_attention=model_config.get("use_scale_attention", True),
            use_patch_attention=model_config.get("use_patch_attention", True),
            use_mil=model_config.get("use_mil", True),
            shared_backbone=model_config.get("shared_backbone", True),
            patch_grid=model_config.get("patch_grid", 4),
        )
    else:
        raise ValueError(f"Unknown checkpoint model: {model_config['name']}")

    model.load_state_dict(checkpoint["model_state"], strict=False)
    model.to(device).eval()
    return model, checkpoint


@torch.no_grad()
def predict_image(
    model: torch.nn.Module,
    image_path: str | Path,
    cancer_classes: list[str],
    subtype_classes: Optional[list[str]] = None,
    image_size: int = 224,
    model_name: str = "multiscale",
    device: str | torch.device = "cpu",
    scale_fractions: tuple[float, float, float] = (1.0, 0.75, 0.55),
    confidence_threshold: float = 0.60,
    organ_type: Optional[str] = "auto",
) -> dict[str, Any]:
    """Perform single image inference with hierarchical cancer and subtype classification."""
    from xai.ood import evaluate_ood_and_uncertainty

    device = torch.device(device)
    organ_key = (organ_type or "auto").strip().lower()

    multi_preprocessor = build_preprocessor(
        image_size, train=False, scale_fractions=scale_fractions
    )
    preprocessor = (
        BaselinePreprocessor(multi_preprocessor.config)
        if model_name == "baseline"
        else multi_preprocessor
    )
    with Image.open(image_path) as image:
        tensor = preprocessor(image)
    model_input = tensor.unsqueeze(0).to(device)

    # 1. Forward pass
    output = model(model_input)

    # 2. Cancer Classification
    cancer_logits = output["cancer_logits"]
    cancer_probs_tensor = torch.softmax(cancer_logits, dim=1)[0]
    cancer_probs_list = cancer_probs_tensor.cpu().tolist()
    cancer_index = int(cancer_probs_tensor.argmax())
    predicted_cancer_type = cancer_classes[cancer_index]

    cancer_prob_dict = {
        label: float(cancer_probs_list[idx])
        for idx, label in enumerate(cancer_classes)
    }

    # 3. OOD & Uncertainty Assessment
    unc_res = evaluate_ood_and_uncertainty(
        logits=cancer_logits,
        probs=cancer_probs_tensor,
        confidence_threshold=confidence_threshold,
    )

    # 4. Hierarchical Subtype Classification (Run ONLY predicted cancer's subtype head)
    subtype_result: dict[str, Any] = {
        "available": False,
        "label": "None",
        "confidence": 0.0,
        "probabilities": {},
    }

    subtype_dict = output.get("subtype_dict", {})
    if subtype_dict and predicted_cancer_type in subtype_dict:
        sub_logits = subtype_dict[predicted_cancer_type]
        sub_probs_tensor = torch.softmax(sub_logits, dim=1)[0]
        sub_probs_list = sub_probs_tensor.cpu().tolist()
        sub_index = int(sub_probs_tensor.argmax())

        # Check label mapping in model or config if available
        sub_names = ORGAN_SUBTYPES.get(predicted_cancer_type.lower(), [f"Subtype_{i+1}" for i in range(len(sub_probs_list))])
        if len(sub_names) != len(sub_probs_list):
            sub_names = [f"Subtype_{i+1}" for i in range(len(sub_probs_list))]

        pred_sub_label = sub_names[sub_index]
        sub_conf = float(sub_probs_list[sub_index])

        subtype_result = {
            "available": True,
            "label": pred_sub_label,
            "confidence": round(sub_conf, 4),
            "probabilities": {
                name: round(float(sub_probs_list[i]), 4)
                for i, name in enumerate(sub_names)
            },
        }
    elif output.get("subtype_logits") is not None:
        sub_logits = output["subtype_logits"]
        sub_probs_tensor = torch.softmax(sub_logits, dim=1)[0]
        sub_probs_list = sub_probs_tensor.cpu().tolist()
        sub_index = int(sub_probs_tensor.argmax())
        sub_names = subtype_classes if subtype_classes else [f"Subtype_{i+1}" for i in range(len(sub_probs_list))]

        subtype_result = {
            "available": True,
            "label": sub_names[sub_index % len(sub_names)],
            "confidence": round(float(sub_probs_list[sub_index]), 4),
            "probabilities": {
                name: round(float(sub_probs_list[i % len(sub_probs_list)]), 4)
                for i, name in enumerate(sub_names)
            },
        }

    # Extract scale weights if available
    scale_att_dict = {}
    if "scale_weights" in output and output["scale_weights"] is not None:
        sw = output["scale_weights"][0].detach().cpu().tolist()
        scale_att_dict = {
            "low": round(float(sw[0]), 4),
            "medium": round(float(sw[1]), 4) if len(sw) > 1 else 0.0,
            "high": round(float(sw[2]), 4) if len(sw) > 2 else 0.0,
        }

    # Format full prediction response according to Phase 30
    return {
        "status": "ok",
        "research_use_only": True,
        "research_disclaimer": "Research prediction only. Not a clinical diagnosis.",
        "image_path": str(image_path),
        "organ_type": organ_key.capitalize(),
        "has_cancer": predicted_cancer_type.upper() not in ["NORMAL", "BENIGN", "OOD"],
        "cancer_status": f"Cancer Detected ({predicted_cancer_type})" if predicted_cancer_type.upper() not in ["NORMAL", "BENIGN", "OOD"] else "Non-Cancerous / Normal",
        "cancer": {
            "label": predicted_cancer_type,
            "confidence": round(float(unc_res["msp"]), 4),
            "probabilities": {k: round(v, 4) for k, v in cancer_prob_dict.items()},
        },
        "cancer_type": predicted_cancer_type,
        "cancer_confidence": round(float(unc_res["msp"]), 4),
        "top2_confidence": round(float(sorted(cancer_probs_list, reverse=True)[1]) if len(cancer_probs_list) > 1 else 0.0, 4),
        "cancer_probabilities": {k: round(v, 4) for k, v in cancer_prob_dict.items()},
        "entropy": unc_res["entropy"],
        "is_low_confidence": unc_res["is_uncertain"],
        "confidence_warning": unc_res["message"] if unc_res["is_uncertain"] else None,
        "subtype": subtype_result,
        "uncertainty": {
            "entropy": unc_res["entropy"],
            "ood": unc_res["is_ood"],
            "is_uncertain": unc_res["is_uncertain"],
            "status": unc_res["status"],
            "message": unc_res["message"],
        },
        "scale_attention": scale_att_dict,
        "scale_weights": output["scale_weights"][0].detach().cpu().tolist() if "scale_weights" in output and output["scale_weights"] is not None else [0.33, 0.33, 0.33],
        "patch_weights": output["patch_weights"][0].detach().cpu().tolist() if "patch_weights" in output and output["patch_weights"] is not None else [],
    }


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load YAML configuration file."""
    import yaml

    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)