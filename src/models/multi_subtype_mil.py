"""
src/models/multi_subtype_mil.py
Alias pointing to mainmodel.py (primary deep learning prediction model).
"""

from mainmodel import (
    TASKS,
    TASK_CANCER,
    PAN_CANCER_TASKS,
    PAN_CANCER_TASK_CANCER,
    GatedAttention,
    CrossScaleEncoder,
    MultiSubtypeMIL,
    CrossValidationEnsembleMIL,
    load_pan_cancer_model,
    masked_multitask_loss,
    class_weights_from_counts,
)
