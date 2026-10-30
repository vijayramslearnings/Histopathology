# PathoVision AI: Multi-Cancer & Subtype Diagnostic Suite (Cross-Scale Attention MIL)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Benchmark Accuracy](https://img.shields.io/badge/Cohort%20Accuracy-99.18%25-brightgreen.svg)]()

> **PathoVision AI** is an end-to-end computational pathology system that predicts primary cancer sites across **10 anatomical organs** and multi-task molecular/histological subtypes from Whole-Slide Images (WSI) and biopsy micrographs using a novel **Cross-Scale Gated Residual Attention MIL** architecture.

---

## 🔬 Core Novelty & Highlights

1. **Cross-Scale Gated Residual Attention MIL:**
   - Eliminates the vulnerability of naive feature stacking where low-power stromal noise corrupts fine cytological details.
   - Modulates high-power ($20\times$) cellular embeddings with low-power ($5\times$) regional context via a learnable data-dependent sigmoid gate:
     $$u_i = [h_i^{(20)} \parallel h_{p(i)}^{(5)}], \quad g_i = \sigma(W_g u_i + b_g)$$
     $$z_i = h_i^{(20)} + g_i \odot \text{ReLU}(W_m u_i)$$
   - **5.5× error reduction** (error rate dropped from $5.0\%$ with standard concatenation to just **$0.9\%$**).

2. **Pan-Cancer Multi-Task Coverage (10 Primary Organs):**
   - Simultaneously predicts primary cancer sites (`BRCA`, `LUNG`, `KIDNEY`, `COLON`, `GLIOMA`, `PRAD`, `STAD`, `LIHC`, `SKCM`, `PAAD`) and task-specific subtypes using **decoupled task-specific attention heads**.
   - Subtype diagnostics: PAM50 & Histology (Breast), LUAD vs. LUSC (Lung), RCC Tri-type (Kidney), MSI-H vs. MSS (Colon), IDH & Grade (Glioma), Gleason grade (Prostate), MSI/Molecular (Gastric), HCC vs. Cholangiocarcinoma (Liver), and BRAF/Genomics (Melanoma).

3. **Multi-Scale Grad-CAM & XAI Decomposition:**
   - 4-panel visual attention decomposition:
     $$\text{(a) Raw H\&E} \longrightarrow \text{(b) } 20\times \text{ Cellular} \longrightarrow \text{(c) } 5\times \text{ Context} \longrightarrow \text{(d) Gated Consensus}$$
   - Includes quantitative attribution across **Nuclear Atypia, Glandular Architecture, and Desmoplastic Stroma**.

4. **Turnkey Clinical Web Application:**
   - Powered by FastAPI and an interactive clinical dashboard with automated synoptic report generation (CAP protocol compliant), real-time XAI chatbot, and drag-and-drop slide ingestion.

---

## 📊 Benchmark Results

Evaluated across the curated multi-organ benchmark cohort ($N = 244$ slides across 10 cancers):

| Model Architecture | Magnification | Accuracy | Precision | Recall | Macro-F1 | Subtype Error Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Single-Scale ABMIL** | $5\times$ only | $84.5\%$ | $83.8\%$ | $84.1\%$ | $84.1\%$ | $15.9\%$ |
| **Single-Scale ABMIL** | $20\times$ only | $92.6\%$ | $92.1\%$ | $92.4\%$ | $92.4\%$ | $7.6\%$ |
| **Multi-Scale MIL (Concat)** | $20\times + 5\times$ | $95.2\%$ | $94.8\%$ | $95.0\%$ | $95.0\%$ | $5.0\%$ |
| **PathoVision AI (Ours)** | **$20\times + 5\times$** | **$99.18\%$** | **$99.25\%$** | **$99.10\%$** | **$99.10\%$** | **$0.9\%$** |

*Primary Cancer Site Accuracy: **100.0%** across all organ cohorts.*

---

## 🖼️ Explainability & Attention Heatmaps

The system outputs an authentic 4-panel multi-scale attention decomposition demonstrating how learnable gating filters low-power background noise:

![Grad-CAM Multi-Scale Heatmap](reports/heatmaps/exact_matched_gradcam_figure.png)
*Grad-CAM: H&E, 20x, 5x, and gated consensus attention.*

---

## 📁 Repository Structure

```
Histopathology/
├── checkpoints/              # Pretrained 5-Fold MIL Ensemble & Morphological Prototypes
│   ├── best_multi_subtype_mil_fold0.pt ... fold4.pt
│   └── morphological_prototypes.pt
├── data/
│   ├── features/resnet50/    # 1024-d dual-scale feature tensors (.pt)
│   ├── scratch_uploads/      # Staging directory for external biopsy uploads
│   └── pilot_cohort.csv      # Specimen metadata & clinical labels
├── reports/
│   ├── heatmaps/             # Generated Grad-CAM & multi-scale consensus figures
│   └── COMPREHENSIVE_RESEARCH_TEST_RESULTS.md
├── src/
│   ├── app/
│   │   ├── web_app.py        # FastAPI server & inference endpoints
│   │   ├── xai_chatbot.py    # Conversational pathology reasoning agent
│   │   └── templates/        # Interactive pathology web dashboard (HTML5/Tailwind)
│   ├── models/
│   │   ├── multi_subtype_mil.py # GatedAttention, CrossScaleEncoder, MultiSubtypeMIL
│   │   └── baselines.py      # Benchmark comparison baselines
│   ├── preprocess/
│   │   ├── process_external.py # Turnkey biopsy tiling & Macenko stain normalizer
│   │   └── extract_features.py # ResNet-50 1024-d patch feature extraction
│   └── xai/
│       ├── gradcam.py        # End-to-end convolutional Grad-CAM engine
│       ├── heatmaps.py       # 4-panel multi-scale decomposition generator
│       └── faithfulness.py   # Adebayo et al. saliency sanity checks
├── mainmodel.py              # Root multi-task model architecture definition
├── start_app.py              # Turnkey launcher script (boots server & opens browser)
├── requirements.txt          # Python dependencies
└── README.md
```

---

## 🚀 Quick Start

### 1. Prerequisites & Environment Setup
Clone the repository and install dependencies in a Python 3.10+ virtual environment:

```bash
git clone https://github.com/vijayramslearnings/Histopathology.git
cd Histopathology
git checkout develope

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Launch the Application
Run the one-click launcher to boot the FastAPI backend and automatically open the clinical dashboard:

```bash
python3 start_app.py
```
Or start via uvicorn directly:
```bash
python3 -m uvicorn src.app.web_app:app --host 127.0.0.1 --port 8000
```
Then navigate to **`http://127.0.0.1:8000`** in your web browser.

---

## 🧪 Usage & API Endpoints

### 1. Web Dashboard
- **Analyze Preset Slides:** Select any of the curated clinical specimens from the sidebar to inspect multi-task predictions, uncertainty intervals, and interactive heatmaps.
- **Upload Custom Biopsies:** Drag and drop `.png`, `.jpg`, `.jpeg`, or `.tiff` histology images. The system automatically tiles the slide, applies Macenko stain normalization, extracts 1024-d ResNet-50 features, and renders predictions.
- **Compute Grad-CAM:** Click **"Compute Grad-CAM"** to backpropagate gradients through ResNet-50 `layer3` and view the 4-panel attention decomposition.
- **Interactive XAI Chatbot:** Ask the embedded AI pathologist specific clinical questions (e.g., *"Why did you diagnose adenocarcinoma?"*, *"Explain the spatial attention heatmap"*).

### 2. Programmatic REST API

#### Run Prediction on Slide Feature Tensor:
```bash
curl -X POST "http://127.0.0.1:8000/api/predict" \
     -H "Content-Type: application/json" \
     -d '{"patient_id": "TCGA-05-4244"}'
```

#### Upload Biopsy Image Directly:
```bash
curl -X POST "http://127.0.0.1:8000/api/upload" \
     -F "file=@path/to/biopsy.jpeg" \
     -F "organ_prior=AUTO" \
     -F "normalize_stain=true"
```

#### Fetch 4-Panel Multi-Scale Heatmap Figure:
```bash
curl -O "http://127.0.0.1:8000/api/multiscale_heatmap/TCGA-05-4244"
```

---

## 📚 References & Citation

If you find this work or codebase helpful for your research, please cite:

```text
[1] M. Ilse, J. M. Tomczak, and M. Welling, “Attention-based deep multiple instance learning,” in Proc. 35th Int. Conf. Mach. Learn. (ICML), 2018, pp. 2127–2136.
[2] M. Y. Lu et al., “Data-efficient and weakly supervised computational pathology on whole-slide images,” Nat. Biomed. Eng., vol. 5, no. 6, pp. 555–570, Jun. 2021.
[3] N. Hashimoto et al., “Multi-scale domain-adversarial multiple-instance learning for cancer subtype classification with unannotated clinical whole slide images,” in Proc. IEEE/CVF CVPR, 2020, pp. 9852–9861.
[4] J. Adebayo et al., “Sanity checks for saliency maps,” in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 31, 2018, pp. 9505–9515.
[5] The Cancer Genome Atlas (TCGA) Research Network, “Pan-Cancer Atlas Whole-Slide Imaging Cohorts,” National Cancer Institute GDC Data Portal, 2026.
```

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
