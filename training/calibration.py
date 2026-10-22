"""
training/calibration.py
Temperature Scaling and Out-of-Distribution (OOD) Abstention:
- Temperature scaling parameter optimization to minimize ECE (target: ECE <= 0.03)
- Expected Calibration Error (ECE) metric calculation
- OOD Abstention detector based on energy scores / maximum softmax probability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score
from typing import Tuple, Dict


def compute_ece(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    """
    Computes Expected Calibration Error (ECE).
    probs: [N, C] predicted class probabilities
    labels: [N] true integer labels
    """
    confidences, predictions = torch.max(probs, dim=1)
    accuracies = predictions.eq(labels)

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = confidences.gt(bin_lower) * confidences.le(bin_upper)
        prop_in_bin = in_bin.float().mean().item()

        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean().item()
            avg_confidence_in_bin = confidences[in_bin].mean().item()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)


class TemperatureScaler(nn.Module):
    """
    Post-hoc Temperature Scaling (Guo et al., ICML 2017).
    Calibrates logits: p_i = softmax(z_i / T)
    """
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        t = torch.clamp(self.temperature, min=0.1, max=10.0)
        return logits / t

    def fit(self, val_logits: torch.Tensor, val_labels: torch.Tensor, max_iter: int = 50) -> float:
        """Optimizes temperature T on validation set using NLLLoss."""
        nll_criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=max_iter)

        def eval_step():
            optimizer.zero_grad()
            scaled_logits = self.forward(val_logits)
            loss = nll_criterion(scaled_logits, val_labels)
            loss.backward()
            return loss

        optimizer.step(eval_step)
        with torch.no_grad():
            cal_probs = F.softmax(self.forward(val_logits), dim=-1)
            ece = compute_ece(cal_probs, val_labels)
        return ece


def evaluate_ood_abstention(
    in_dist_logits: torch.Tensor,
    ood_logits: torch.Tensor,
) -> float:
    """
    Evaluates OOD detection performance using maximum softmax probability.
    Returns AUROC.
    """
    in_scores = F.softmax(in_dist_logits, dim=-1).max(dim=-1)[0].detach().cpu().numpy()
    ood_scores = F.softmax(ood_logits, dim=-1).max(dim=-1)[0].detach().cpu().numpy()

    y_true = np.concatenate([np.ones_like(in_scores), np.zeros_like(ood_scores)])
    scores = np.concatenate([in_scores, ood_scores])

    auroc = roc_auc_score(y_true, scores)
    return float(auroc)
