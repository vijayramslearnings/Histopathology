"""Multi-task losses with masked subtype supervision."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn


class MultiTaskLoss(nn.Module):
    """Cancer cross-entropy plus hierarchical masked subtype cross-entropy."""

    def __init__(
        self,
        subtype_weight: float = 1.0,
        cancer_weight: Optional[Tensor] = None,
        subtype_weight_vector: Optional[Tensor] = None,
    ) -> None:
        super().__init__()
        self.subtype_weight = float(subtype_weight)
        self.register_buffer("cancer_weight_vector", cancer_weight)
        self.register_buffer("subtype_weight_vector", subtype_weight_vector)

    def forward(
        self,
        cancer_logits: Tensor,
        cancer_targets: Tensor,
        subtype_logits: Optional[Tensor] = None,
        subtype_targets: Optional[Tensor] = None,
        subtype_dict: Optional[dict[str, Tensor]] = None,
        cancer_names: Optional[list[str]] = None,
        subtype_target_dict: Optional[dict[str, Tensor]] = None,
    ) -> dict[str, Tensor]:
        cancer_loss = nn.functional.cross_entropy(
            cancer_logits, cancer_targets, weight=self.cancer_weight_vector
        )

        subtype_loss = cancer_loss.new_zeros(())
        if subtype_dict and cancer_names and subtype_target_dict:
            # Calculate hierarchical loss per cancer type
            total_sub_samples = 0
            for idx, cname in enumerate(cancer_names):
                if cname in subtype_dict and cname in subtype_target_dict:
                    mask = (cancer_targets == idx)
                    if torch.any(mask):
                        sub_logits = subtype_dict[cname][mask]
                        sub_targets = subtype_target_dict[cname][mask]
                        valid_mask = sub_targets >= 0
                        if torch.any(valid_mask):
                            sub_loss = nn.functional.cross_entropy(
                                sub_logits[valid_mask], sub_targets[valid_mask]
                            )
                            subtype_loss = subtype_loss + sub_loss
                            total_sub_samples += 1
            if total_sub_samples > 0:
                subtype_loss = subtype_loss / total_sub_samples
        elif subtype_logits is not None and subtype_targets is not None and torch.any(subtype_targets >= 0):
            subtype_loss = nn.functional.cross_entropy(
                subtype_logits,
                subtype_targets,
                weight=self.subtype_weight_vector,
                ignore_index=-1,
            )

        total = cancer_loss + self.subtype_weight * subtype_loss
        return {"total": total, "cancer": cancer_loss, "subtype": subtype_loss}