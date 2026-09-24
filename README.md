# Explainable Multi-Scale Attention-Based Deep Learning Framework for Multi-Cancer and Subtype Classification Using Histopathology Images

> **RESEARCH-USE ONLY DISCLAIMER**:
> This software is a research prototype for computational pathology.
> It is **NOT** a clinical diagnostic system and **MUST NEVER** be used for clinical decision-making or patient diagnosis.
> Model prediction outputs, confidence levels, Grad-CAM, and attention heatmaps are experimental model inspection aids only.

---

## 1. Overview & Research Objective

This repository implements a modular computational pathology backend designed for explainable multi-cancer and subtype classification using Whole-Slide Images (WSI) and histopathology image patches.

Key Architectural Principles:
- **Metadata-Driven Label System**: Flexible cancer type and subtype metadata mapping without hardcoded biological assumptions.
- **Patient-Level Data Splitting & Leakage Detection**: Strict guarantees that images/patches from the same patient or slide never cross train/val/test splits.
- **Multi-Scale Feature Processing & Attention**: Hierarchical fusion and learnable scale and patch attention mechanisms (5×, 10×, 20× magnification approximations).
- **Multiple Instance Learning (MIL)**: Aggregation of variable numbers of patch representations into slide-level classification.
- **Uncertainty & Abstention**: Confidence scoring, entropy estimation, temperature calibration, and configurable criteria for low-confidence abstention.
- **Faithful & Stable Explainability (XAI)**: Grad-CAM, spatial patch attention heatmaps, occlusion mapping, deletion/insertion faithfulness tests, and transformation stability tests.
- **Automated Research PDF Reporting**: Synthesis of predictions, uncertainty, XAI visualizations, and research disclaimers into PDF reports.

---

## 2. Phase 1 Skeleton Architecture

```text
backend/
├── app/
│   ├── main.py                # FastAPI application entry point with CORS, logging & exception handling
│   ├── api/
│   │   ├── routes/
│   │   │   └── health.py      # /health and /health/model endpoints
│   │   └── dependencies.py
│   ├── core/
│   │   ├── config.py          # Pydantic Settings & compute device detection (CUDA/MPS/CPU)
│   │   ├── logging.py         # Structured logging setup (privacy-enforced)
│   │   └── exceptions.py      # Custom PathologyException hierarchy & JSON error handlers
│   └── schemas/
│       └── health.py          # Health response Pydantic models
├── configs/
│   └── config.yaml            # Project configuration
├── tests/
│   └── test_health.py         # Pytest suite for Phase 1 health endpoints
├── requirements.txt           # Python dependencies
├── .env.example               # Environment configuration template
└── README.md
```

---

## 3. Installation & Quickstart

### Prerequisites
- Python 3.11+
- Virtual environment (recommended)

### Environment Setup

```bash
# Navigate to project root
cd /path/to/backend

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Configuration

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

---

## 4. Running the FastAPI Server

Launch the server with Uvicorn:

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Expected startup log output:
```text
[INFO] Starting Explainable Multi-Scale Attention Histopathology AI v1.0.0
[INFO] Environment: development | Active Device: mps (or cuda / cpu)
[INFO] RESEARCH NOTICE: System initialized in Experimental Research Mode.
INFO:  Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

---

## 5. Testing the Interactive API Documentation (Swagger / ReDoc)

Once Uvicorn is running, open your browser and navigate to:

- **Interactive OpenAPI (Swagger UI)**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc Documentation**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
- **OpenAPI JSON Schema**: [http://127.0.0.1:8000/openapi.json](http://127.0.0.1:8000/openapi.json)

---

## 6. Endpoints Available in Phase 1

### 1. `GET /health` (or `GET /api/v1/health`)
Returns overall service status, version, and research disclaimer.

**Response**:
```json
{
  "status": "ok",
  "service": "histopathology-ai-backend",
  "version": "1.0.0",
  "disclaimer": "Experimental research prediction only. Not a clinical diagnostic system. Not for clinical use."
}
```

### 2. `GET /health/model` (or `GET /api/v1/health/model`)
Returns active compute hardware (CUDA/MPS/CPU), model initialization state, and checkpoint status.

**Response**:
```json
{
  "status": "ok",
  "model_loaded": false,
  "device": "mps",
  "model_version": "1.0.0",
  "checkpoint_status": "not_loaded",
  "disclaimer": "Experimental research prediction only. Not a clinical diagnostic system. Not for clinical use."
}
```

---

## 7. Running Automated Tests

Run the automated pytest test suite:

```bash
PYTHONPATH=. pytest tests/test_health.py -v
```

Expected output:
```text
tests/test_health.py::test_health_endpoint PASSED [ 33%]
tests/test_health.py::test_model_health_endpoint PASSED [ 66%]
tests/test_health.py::test_api_v1_health_routes PASSED [100%]
3 passed in 0.92s
```

---

## 8. Development Roadmap

- [x] **Phase 1**: Backend skeleton, core configuration, logging, exceptions, health API, pytest suite
- [ ] **Phase 2**: Database layer (SQLite / SQLAlchemy / Alembic migration-ready)
- [ ] **Phase 3**: Image validation service (`POST /api/v1/analysis/validate-image`)
- [ ] **Phase 4**: WSI tiled reading & patch extraction (`POST /api/v1/preprocessing/...`)
- [ ] **Phase 5**: Dataset metadata validation, patient-level split & leakage detection
- [ ] **Phase 6**: Deep learning model components (ResNet-50 backbone, Cross-Scale Fusion, Scale Attention, Patch Attention, MIL, Cancer & Subtype Classifiers)
- [ ] **Phase 7**: Inference service, uncertainty estimation & abstention logic
- [ ] **Phase 8**: Explainability module (Grad-CAM, Patch Attention Heatmaps, Occlusion, Faithfulness & Stability evaluation)
- [ ] **Phase 9**: Evaluation service, Ablation framework & Experiment tracking
- [ ] **Phase 10**: Report generation (ReportLab PDF) & Docker containerization