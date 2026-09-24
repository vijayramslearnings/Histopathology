"""Out-Of-Distribution (OOD) detection, temperature calibration, and uncertainty scoring."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.optimize import minimize_scalar


class TemperatureScaler:
    """Optimize temperature parameter T on validation set to minimize Expected Calibration Error (ECE)."""

    def __init__(self, temperature: float = 1.0) -> None:
        self.temperature = max(0.1, float(temperature))

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> float:
        """Find optimal temperature T using negative log likelihood on validation set."""
        logits_tensor = torch.tensor(logits, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        def nll_func(temp: float) -> float:
            if temp <= 0.05:
                return 1e9
            scaled_logits = logits_tensor / temp
            loss = torch.nn.functional.cross_entropy(scaled_logits, labels_tensor)
            return float(loss.item())

        res = minimize_scalar(nll_func, bounds=(0.1, 5.0), method="bounded")
        self.temperature = float(res.x)
        return self.temperature

    def scale_logits(self, logits: torch.Tensor | np.ndarray) -> np.ndarray:
        """Apply learned temperature scaling to raw logits."""
        if isinstance(logits, torch.Tensor):
            logits_np = logits.detach().cpu().numpy()
        else:
            logits_np = np.asarray(logits, dtype=np.float32)
        return logits_np / self.temperature


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE)."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == labels

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin

    return float(ece)


def compute_brier_score(probs: np.ndarray, labels: np.ndarray, num_classes: int = 3) -> float:
    """Compute mean Brier Score for multi-class predictions."""
    one_hot = np.eye(num_classes)[labels]
    brier = np.mean(np.sum((probs - one_hot) ** 2, axis=1))
    return float(brier)


def calculate_energy_score(logits: torch.Tensor | np.ndarray, temperature: float = 1.0) -> float:
    """Calculate Energy Score: E(x) = -T * log(sum(exp(z_i / T)))."""
    if isinstance(logits, torch.Tensor):
        logits_np = logits.detach().cpu().numpy()
    else:
        logits_np = np.asarray(logits, dtype=np.float32)
        
    scaled = logits_np / max(0.1, temperature)
    # Log-sum-exp trick for numerical stability
    max_val = np.max(scaled)
    lse = max_val + np.log(np.sum(np.exp(scaled - max_val)))
    energy = -temperature * lse
    return float(energy)


def evaluate_ood_and_uncertainty(
    logits: torch.Tensor | np.ndarray,
    probs: torch.Tensor | np.ndarray,
    confidence_threshold: float = 0.65,
    entropy_threshold: float = 1.10,
    energy_threshold: float = -2.0,
    temperature: float = 1.0,
) -> Dict[str, Any]:
    """Evaluate Maximum Softmax Probability (MSP), Entropy, Energy Score, and OOD flags."""
    if isinstance(probs, torch.Tensor):
        probs_np = probs.detach().cpu().numpy().flatten()
    else:
        probs_np = np.asarray(probs, dtype=np.float32).flatten()

    msp = float(np.max(probs_np))
    entropy = float(-np.sum(probs_np * np.log2(probs_np + 1e-12)))
    energy = calculate_energy_score(logits, temperature=temperature)

    is_low_confidence = msp < confidence_threshold
    is_high_entropy = entropy > entropy_threshold
    is_energy_ood = energy > energy_threshold

    is_uncertain = is_low_confidence or is_high_entropy
    is_ood = is_energy_ood or (msp < 0.45)

    if is_ood:
        status = "ood"
        message = "Image may be outside the model's training distribution. Unable to provide a reliable class prediction."
    elif is_uncertain:
        status = "uncertain"
        message = "Uncertain Prediction — The model does not show strong separation between the trained classes."
    else:
        status = "valid"
        message = "High-confidence in-distribution prediction."

    return {
        "status": status,
        "is_ood": is_ood,
        "is_uncertain": is_uncertain,
        "msp": round(msp, 4),
        "entropy": round(entropy, 4),
        "energy_score": round(energy, 4),
        "temperature": temperature,
        "message": message,
    }
