"""
src/models/mainmodel.py
Deep Learning Model Architecture and Factory for Whole-Slide Computational Pathology.
Imported directly from root mainmodel.
"""

import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

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
