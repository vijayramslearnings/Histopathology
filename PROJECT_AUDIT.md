# PROJECT_AUDIT.md: Comprehensive Initial Architecture & Codebase Audit

> **Systematic Audit Report for Computational Pathology Framework**
> Date: 2026-09-19
> Scope: Deep Learning Architecture, Dataset Pipeline, Label Taxonomy, Training & Loss Functions, Inference, and XAI Explanations.

---

## 1. Current Architecture
The system is built as a multi-tier computational pathology workspace:
- **Backend Core**: Python 3.11+ / PyTorch framework with FastAPI (`app/main.py`) serving prediction, XAI explanation, error analysis, and chatbot APIs.
- **Model Layers**: Dual architecture featuring a standard ResNet-50 baseline (`models/backbone.py`) and a 3-branch multi-scale attention network with patch attention and MIL pooling (`models/multi_scale_model.py`).
- **Frontend App**: Next.js (React 19, TailwindCSS, Framer Motion) client app (`frontend/`) serving interactive UI visualizations.

---

## 2. Dataset Source
- Currently configured to load metadata from `backend/data/metadata.csv` (or `data/metadata.csv`).
- Images represent histopathology slide/patch samples corresponding to TCGA-like cancer cohorts.

---

## 3. Number of Classes
- **Cancer Types**: 3 primary organ cancer classes (`BRCA`, `LUAD`, `GBM`).
- **Subtypes**: Variable per dataset config. Currently parsed dynamically from metadata string values (e.g., `Subtype_A`, `Subtype_B` or TCGA histopathological subtypes).

---

## 4. Current Cancer Labels
- `BRCA`: Breast Invasive Carcinoma
- `LUAD`: Lung Adenocarcinoma
- `GBM`: Glioblastoma Multiforme

---

## 5. Current Subtype Labels
- Currently stored as strings in metadata (`subtype` column).
- Previously handled as a single global flat subtype dictionary rather than cancer-specific subtype heads, creating potential label domain cross-contamination across organ sites.

---

## 6. Label Encoding
- Cancer classes mapped via contiguity check (`cancer_mapping` in `config.yaml`): `BRCA: 0`, `LUAD: 1`, `GBM: 2`.
- Subtype classes dynamically indexed across all unique subtype string entries in metadata, assigning `-1` to missing/unlabeled subtype samples.

---

## 7. Train / Validation / Test Split
- Configured default ratios: `70% train`, `15% validation`, `15% test` (`config.yaml`).

---

## 8. Patient-Level Splitting Verification
- **Patient Disjoint Guarantee**: Splitting in `dataset/splitting.py` uses `GroupKFold` / group partitioning on `patient_id`.
- Verified strict requirement: All slides and patches from any given `patient_id` reside exclusively in one split (Train, Val, or Test).

---

## 9. Image Preprocessing
- Resize to target input resolution (default `224x224`).
- Standard ImageNet color normalization (Mean: `[0.485, 0.456, 0.406]`, Std: `[0.229, 0.224, 0.225]`).
- Multi-scale branch context downsampling (scale fractions `1.0`, `0.75`, `0.55`).

---

## 10. Data Augmentation
- Baseline training supports Random Horizontal Flip, Random Vertical Flip, Random Affine/Rotation, and Color Jitter during training phase. Validation/Test use deterministic center cropping/scaling.

---

## 11. Model Architecture
- **Backbone**: Pretrained ImageNet ResNet-50 feature extractor up to `layer4`.
- **Feature Fusion**: `CrossScaleFusion` reducing 2048-dim feature maps to `512` dimensions.
- **Attention Modules**:
  - `ScaleAttention`: Learnable softmax weighting across the 3 spatial scale branches.
  - `PatchAttention`: Spatial patch relevance weighting.
  - `MILPooling`: Gated/Attention-based Multiple Instance Learning token aggregation.
- **Classification Heads**: Linear projection heads for primary cancer classification and subtype classification.

---

## 12. Loss Function
- Multi-Task Loss (`training/losses.py`):
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{cancer}} + \lambda_{\text{subtype}} \cdot \mathcal{L}_{\text{subtype}}$$
  where $\mathcal{L}_{\text{cancer}}$ is Cross-Entropy loss and $\mathcal{L}_{\text{subtype}}$ is masked Cross-Entropy ignoring `-1` unannotated subtype samples.

---

## 13. Optimizer
- `AdamW` optimizer with weight decay (`1e-4`) and learning rate (`1e-4`).

---

## 14. Learning-Rate Scheduler
- Currently uses constant LR or manual epoch decay in standard training runs; needs formal cosine annealing / step LR scheduler integration.

---

## 15. Class Balancing
- Supports class-weighted `CrossEntropyLoss` via `cancer_weight_vector` and `subtype_weight_vector`.

---

## 16. Current Metrics
- Evaluates Accuracy, Balanced Accuracy, Macro-F1, and Confusion Matrices for cancer type and subtype predictions.

---

## 17. Current Checkpoint System
- Checkpoints (`checkpoints/multiscale_full.pt`, `resnet50_baseline.pt`) save `model_state`, `model_config`, `cancer_classes`, `subtype_classes`, `image_size`, `scale_fractions`, and training history.

---

## 18. Current Inference Process
- Executed via `prediction/inference.py`:
  Loads image $\to$ applies preprocessing $\to$ runs forward pass $\to$ computes Softmax probabilities $\to$ computes Prediction Entropy $\to$ flags low confidence if confidence $< 0.50$ or entropy is elevated.

---

## 19. Current XAI Implementation
- **Grad-CAM**: Computes target layer gradients on `backbone.model[-1][-1].conv3`.
- **Attention Maps**: Extracts scale attention weights and spatial patch attention weights.
- **Quantitative Metrics**: Faithfulness (Deletion/Insertion AUC) and Transformation Stability (SSIM & correlation under rotation/contrast shifts).

---

## 20. Identified Bugs & Architecture Flaws

1. **Global Subtype Head vs. Cancer-Specific Subtype Heads**:
   - *Bug*: The original model used a single global subtype head spanning all subtype labels across all cancer types.
   - *Consequence*: Subtype classifiers for Breast (BRCA) were evaluated against subtype classes from Lung (LUAD) or Brain (GBM), leading to missing or confused subtype predictions.

2. **Absence of Hierarchical Subtype Gating**:
   - *Bug*: Subtype predictions were made independently of cancer type logits without hierarchical gating.
   - *Consequence*: Model could predict `LUAD` cancer while executing a `BRCA` subtype head, causing subtype mismatches.

3. **Inference Preprocessing Discrepancies**:
   - *Bug*: Independent inference pipelines in API routes previously constructed custom transform pipelines instead of reusing the single canonical preprocessor.
   - *Consequence*: Potential color mode / normalization mismatch between training and inference.

4. **Missing OOD & Temperature Calibration Thresholding**:
   - *Bug*: Raw softmax probabilities were not calibrated using temperature scaling, leading to overconfident incorrect predictions.

5. **Lack of Dedicated Misclassification Archiving**:
   - *Bug*: Misclassifications during evaluation were not automatically logging full diagnostic XAI visual bundles (`gradcam.png`, `top_patches`, `prediction.json`).

---

## 21. Recommended Fixes

1. **Implement Hierarchical Cancer-Specific Subtype Heads**:
   - Create separate classification heads for BRCA, LUAD, and GBM subtypes.
   - During training, route loss computation exclusively through the ground-truth cancer type's subtype head.
   - During inference, evaluate only the subtype head corresponding to the top-predicted cancer class.

2. **Enforce Canonical Preprocessing Module**:
   - Use `build_preprocessor()` across training, validation, testing, inference, and API upload routes.

3. **Add Temperature Calibration & OOD Detection**:
   - Implement temperature scaling and Maximum Softmax Probability / Energy-based OOD detection to return `UNCERTAIN` or `OUT_OF_DISTRIBUTION` for invalid/ambiguous inputs.

4. **Automate Dataset Inspection & Verification**:
   - Generate `dataset_report.csv`, `dataset_report.json`, and `label_distribution.png` to audit every single sample prior to training.

5. **Build Baseline First & Execute Full Ablation Matrix**:
   - Validate simple ResNet-50 baseline first before training the multi-scale MIL model.
   - Save error analysis artifacts for every misclassification in `outputs/errors/`.
