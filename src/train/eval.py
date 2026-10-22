"""
eval.py
Evaluation metrics calculation for multi-task MIL:
- Macro ROC-AUC (one-vs-rest)
- Macro F1 score
- Balanced Accuracy
- Confusion Matrix
- Error cascade comparison
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score, confusion_matrix
import torch
import torch.nn.functional as F
from src.models.multi_subtype_mil import TASKS, TASK_CANCER

def evaluate_predictions(preds_dict, targets_dict, tasks=None):
    """
    preds_dict: dict task -> list of predicted probability vectors [N_slides, num_classes]
    targets_dict: dict task -> list of ground truth integer labels [N_slides] (-1 for missing)
    """
    if tasks is None:
        tasks = TASKS
    metrics = {}
    for task, class_names in tasks.items():
        if task not in preds_dict or len(preds_dict[task]) == 0:
            continue

        probs = np.array(preds_dict[task])      # [N, C]
        targets = np.array(targets_dict[task])  # [N]

        # Filter out slides without labels for this task
        mask = targets >= 0
        if np.sum(mask) == 0:
            continue

        t_sub = targets[mask]
        p_sub = probs[mask]
        preds_cls = np.argmax(p_sub, axis=1)

        task_metrics = {}
        # Balanced accuracy & Macro F1
        task_metrics["balanced_acc"] = float(balanced_accuracy_score(t_sub, preds_cls))
        task_metrics["macro_f1"] = float(f1_score(t_sub, preds_cls, average="macro", zero_division=0))

        # ROC AUC
        n_classes = len(class_names)
        present_classes = np.unique(t_sub)
        if len(present_classes) > 1:
            try:
                if n_classes == 2:
                    task_metrics["roc_auc"] = float(roc_auc_score(t_sub, p_sub[:, 1]))
                else:
                    task_metrics["roc_auc"] = float(roc_auc_score(t_sub, p_sub, multi_class="ovr", average="macro"))
            except Exception:
                task_metrics["roc_auc"] = np.nan
        else:
            task_metrics["roc_auc"] = np.nan

        # Confusion Matrix
        cm = confusion_matrix(t_sub, preds_cls, labels=list(range(n_classes)))
        task_metrics["confusion_matrix"] = cm.tolist()
        task_metrics["n_samples"] = int(np.sum(mask))

        metrics[task] = task_metrics

    return metrics
