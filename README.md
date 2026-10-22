# Multi-Cancer Type and Subtype Prediction from Whole-Slide Images (Cross-Scale Attention MIL)

This repository implements a deep learning computational pathology system for predicting primary cancer type (5 sites) and task-specific molecular and histological subtypes from Whole-Slide Images (WSI), using a **cross-scale attention multi-task MIL** architecture.

## Overview of Tasks
- **Cancer Type (5 classes)**: BRCA, LUNG, KIDNEY, COLON, GLIOMA
- **Subtype Tasks**:
  - `brca_histology`: IDC vs ILC
  - `brca_pam50`: LumA, LumB, HER2, Basal
  - `lung_histology`: LUAD vs LUSC
  - `kidney_histology`: KIRC vs KIRP vs KICH
  - `colon_msi`: MSS vs MSI-H
  - `glioma_grade`: LGG vs GBM
  - `glioma_idh`: IDH-wt vs IDH-mut

## Repository Structure
```
wsi-subtypes/
  configs/            # YAML: data.yaml, extract.yaml, train.yaml
  data/
    manifests/        # GDC manifests per project (.tsv) + all_slides_metadata.csv
    labels/           # raw label files + master_labels.csv
    slides/           # temporary .svs (deleted after extraction)
    coords/           # patch coordinates (.h5) per slide
    features/         # cached features (.pt/.h5) per slide
    splits/           # 5-fold cross-validation patient splits
  src/
    download/         # build_manifests.py, fetch_labels.py
    preprocess/       # tissue_mask.py, make_coords.py, extract_features.py
    models/           # multi_subtype_mil.py, baselines.py, fusion.py
    train/            # train_mil.py, eval.py
    xai/              # heatmaps.py, faithfulness.py
    app/              # streamlit_app.py
  reports/            # class_counts.md, label_sources.md
  tests/              # pytest unit tests
  requirements.txt
  README.md
```

## Quick Start

### 1. Build Manifests & Harmonize Labels (Phase 1)
```bash
python src/download/build_manifests.py
python src/download/fetch_labels.py
```

### 2. Run Tests
```bash
pytest tests/
```
