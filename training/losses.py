"""
training/losses.py
Robust Multi-Task Loss Hierarchy for Computational Pathology:
1. Focal Loss with Class Balancing and Label Smoothing (epsilon = 0.1, gamma = 2.0)
2. Supervised Contrastive Loss (SupCon) for latent feature separation
3. Composite Multi-Task Loss combining Cancer CE, Subtype Focal+Smooth, and SupCon
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Union


class FocalLoss(nn.Module):
    """
    Focal Loss with class-balancing alpha_t and optional label smoothing.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        label_smoothing: float = 0.1,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha  # Per-class weight tensor [C]
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: [B, C] unnormalized logits
        targets: [B] ground truth class indices
        """
        num_classes = logits.size(-1)
        
        # Apply label smoothing to target distribution
        if self.label_smoothing > 0.0:
            smooth_targets = torch.full_like(logits, self.label_smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        else:
            smooth_targets = F.one_hot(targets, num_classes=num_classes).float()

        log_p = F.log_softmax(logits, dim=-1)
        p = torch.exp(log_p)

        # Focal weighting: (1 - p)^gamma
        focal_weight = torch.pow(1.0 - p, self.gamma)

        # Class weights alpha_t
        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            if alpha.ndim == 1 and alpha.size(0) == num_classes:
                # Shape [1, C]
                alpha_weight = alpha.unsqueeze(0)
                focal_weight = focal_weight * alpha_weight

        # Weighted cross entropy
        loss = -torch.sum(smooth_targets * focal_weight * log_p, dim=-1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Learning (SupCon, Khosla et al.)
    L_SupCon = sum_{i} (-1 / |P(i)|) * sum_{p in P(i)} log( exp(z_i . z_p / tau) / sum_{a} exp(z_i . z_a / tau) )
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        features: [B, D] L2-normalized projection vectors
        labels: [B] integer class labels
        """
        device = features.device
        batch_size = features.shape[0]

        if batch_size <= 1:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Ensure features are L2 normalized
        features = F.normalize(features, p=2, dim=-1)

        # Similarity matrix [B, B]
        similarity = torch.matmul(features, features.T) / self.temperature

        # Create mask for same class (positive pairs)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # Exclude self-contrast (diagonal)
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
        mask = mask * logits_mask

        # For numerical stability: subtract max per row
        logits_max, _ = torch.max(similarity * logits_mask, dim=1, keepdim=True)
        logits = similarity - logits_max.detach()

        # Denominator: sum over all distinct elements
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        # Mean of log-likelihood over positives
        mask_sum = mask.sum(1)
        # Avoid division by zero if an instance has no positive pair in the batch
        has_positives = mask_sum > 0

        if not has_positives.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        mean_log_prob_pos = (mask * log_prob).sum(1)[has_positives] / mask_sum[has_positives]
        loss = -mean_log_prob_pos.mean()
        return loss


class CompositeMultiTaskLoss(nn.Module):
    """
    L_Total = L_Cancer_CE + lambda_subtype * L_Subtype_Focal + lambda_supcon * L_SupCon
    """
    def __init__(
        self,
        subtype_alpha: Optional[torch.Tensor] = None,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.1,
        supcon_weight: float = 0.1,
        subtype_weight: float = 1.0,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.cancer_loss_fn = nn.CrossEntropyLoss()
        self.subtype_loss_fn = FocalLoss(
            alpha=subtype_alpha,
            gamma=focal_gamma,
            label_smoothing=label_smoothing,
        )
        self.supcon_loss_fn = SupervisedContrastiveLoss(temperature=temperature)
        self.supcon_weight = supcon_weight
        self.subtype_weight = subtype_weight

    def forward(
        self,
        cancer_logits: torch.Tensor,
        cancer_targets: torch.Tensor,
        subtype_logits: Optional[torch.Tensor] = None,
        subtype_targets: Optional[torch.Tensor] = None,
        projected_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        
        # 1. Primary Cancer Cross-Entropy
        l_cancer = self.cancer_loss_fn(cancer_logits, cancer_targets)
        total_loss = l_cancer

        parts = {"cancer_ce": l_cancer}

        # 2. Subtype Focal + Label Smoothing
        if subtype_logits is not None and subtype_targets is not None:
            # Filter out missing/unlabeled targets (-1)
            valid_mask = subtype_targets >= 0
            if valid_mask.any():
                l_subtype = self.subtype_loss_fn(
                    subtype_logits[valid_mask],
                    subtype_targets[valid_mask],
                )
                total_loss = total_loss + self.subtype_weight * l_subtype
                parts["subtype_focal"] = l_subtype
            else:
                parts["subtype_focal"] = torch.tensor(0.0, device=cancer_logits.device)
        else:
            parts["subtype_focal"] = torch.tensor(0.0, device=cancer_logits.device)

        # 3. Supervised Contrastive Loss
        if (
            projected_features is not None
            and subtype_targets is not None
            and self.supcon_weight > 0.0
        ):
            valid_mask = subtype_targets >= 0
            if valid_mask.sum() > 1:
                l_supcon = self.supcon_loss_fn(
                    projected_features[valid_mask],
                    subtype_targets[valid_mask],
                )
                total_loss = total_loss + self.supcon_weight * l_supcon
                parts["supcon"] = l_supcon
            else:
                parts["supcon"] = torch.tensor(0.0, device=cancer_logits.device)
        else:
            parts["supcon"] = torch.tensor(0.0, device=cancer_logits.device)

        parts["total"] = total_loss
        return parts


def compute_class_balanced_alpha(counts: Dict[int, int] or list, beta: float = 0.9999) -> torch.Tensor:
    """
    Computes class-balanced alpha weights (Cui et al., CVPR 2019):
    E_n = (1 - beta^n) / (1 - beta)
    alpha_i = (1 / E_n_i) / sum_j (1 / E_n_j) * C
    """
    if isinstance(counts, dict):
        max_cls = max(counts.keys())
        counts_list = [counts.get(i, 0) for i in range(max_cls + 1)]
    else:
        counts_list = list(counts)

    c = len(counts_list)
    effective_num = [1.0 - (beta ** n) if n > 0 else 1.0 for n in counts_list]
    weights = [1.0 / max(e, 1e-6) for e in effective_num]
    total_weight = sum(weights)
    normalized = [(w / total_weight) * c for w in weights]
    return torch.tensor(normalized, dtype=torch.float32)
