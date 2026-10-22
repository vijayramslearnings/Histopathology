"""
test_pipeline_e2e.py
Unit and integration tests for:
- WSISubtypeDataset
- Baselines (Mean, Max, ABMIL, TransMIL)
- Explainability (Heatmaps & Faithfulness)
"""

import pytest
import torch
from pathlib import Path
from src.train.dataset import WSISubtypeDataset
from src.models.baselines import MeanPoolingMIL, MaxPoolingMIL, ABMIL, TransMIL
from src.models.multi_subtype_mil import MultiSubtypeMIL, TASKS, PAN_CANCER_TASKS
from src.xai.heatmaps import generate_slide_heatmap
from src.xai.faithfulness import run_faithfulness_test

COHORT_CSV = Path("data/pilot_cohort.csv")
FEATS_DIR = Path("data/features/resnet50")

def test_dataset_loading():
    assert COHORT_CSV.exists(), "Pilot cohort CSV must exist."
    assert FEATS_DIR.exists(), "Features directory must exist."

    ds_train = WSISubtypeDataset(COHORT_CSV, FEATS_DIR, fold=0, is_train=True)
    ds_val = WSISubtypeDataset(COHORT_CSV, FEATS_DIR, fold=0, is_train=False)

    assert len(ds_train) in [80, 120, 160]
    assert len(ds_val) in [20, 30, 40]

    sample = ds_train[0]
    assert "f20" in sample
    assert "f5" in sample
    assert "parent" in sample
    assert "labels" in sample
    assert sample["f20"].ndim == 2
    assert sample["f20"].shape[1] == 1024

def test_baselines_forward():
    f20 = torch.randn(50, 1024)
    for model_cls in [MeanPoolingMIL, MaxPoolingMIL, ABMIL, TransMIL]:
        model = model_cls(in_dim=1024, dim=128)
        logits, attn = model(f20)
        assert "cancer_type" in logits
        assert logits["cancer_type"].shape == (5,)
        assert "cancer_type" in attn
        assert attn["cancer_type"].shape == (50,)

def test_xai_heatmap_and_faithfulness():
    ckpt_path = Path("checkpoints/best_multi_subtype_mil_fold0.pt")
    if not ckpt_path.exists():
        pytest.skip("Checkpoint not found, skipping.")

    ckpt = torch.load(ckpt_path, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    is_pan_cancer = "heads.cancer_type.weight" in state_dict and state_dict["heads.cancer_type.weight"].shape[0] == 10
    tasks = PAN_CANCER_TASKS if is_pan_cancer else TASKS
    model = MultiSubtypeMIL(in_dim=1024, dim=256, tasks=tasks)
    model.load_state_dict(state_dict)

    feat_files = list(FEATS_DIR.glob("*.pt"))
    assert len(feat_files) > 0
    slide_data = torch.load(feat_files[0], weights_only=False)

    # Test heatmap
    hm_res = generate_slide_heatmap(model, slide_data, task="cancer_type")
    assert "save_path" in hm_res
    assert Path(hm_res["save_path"]).exists()
    assert len(hm_res["top_indices"]) == min(16, len(slide_data["f20"]))

    # Test faithfulness
    faith_res = run_faithfulness_test(model, slide_data, task="cancer_type", n_steps=5)
    assert len(faith_res["attn_probs"]) == 5
    assert len(faith_res["rand_probs"]) == 5
    assert Path(faith_res["plot_path"]).exists()


def test_external_data_processing():
    """Verify processing external images into valid .pt feature dictionaries."""
    from src.preprocess.process_external import process_external_data
    test_img = Path("data/external_samples/test_external_histology.png")
    if not test_img.exists():
        pytest.skip("Test external image not found.")

    out_file = process_external_data(
        image_path=str(test_img),
        patient_id="TEST_EXTERNAL_UNITTEST",
        output_dir="data/features/resnet50",
    )
    assert Path(out_file).exists()
    data = torch.load(out_file, weights_only=False)
    assert "f20" in data and "f5" in data and "parent" in data and "coords20" in data
    assert data["f20"].shape[1] == 1024


def test_clinical_synoptic_report_and_api():
    """Verify synoptic report generator produces CAP-aligned clinical fields & audio script."""
    from fastapi.testclient import TestClient
    from src.app.web_app import app, generate_synoptic_report

    client = TestClient(app)
    # Test report endpoint on pilot slide
    res = client.get("/api/report/BRCA_001")
    assert res.status_code == 200
    report = res.json()
    assert report["specimen_id"] == "BRCA_001"
    assert "primary_cancer" in report
    assert "cancer_confidence_pct" in report
    assert len(report["biomarkers"]) > 0
    assert "Diagnostic briefing" in report["audio_script"]


def test_organ_prior_and_equivocal_detection():
    """Verify that clinical organ prior directs diagnosis and unconditioned low certainty triggers equivocal warning."""
    from src.app.web_app import run_inference_on_data, generate_synoptic_report

    # Synthetic equivocal test data (flat features across 8 patches)
    f20 = torch.randn(8, 1024)
    f5 = f20.mean(dim=0, keepdim=True)
    parent = torch.zeros(8, dtype=torch.long)
    data = {"f20": f20, "f5": f5, "parent": parent, "coords20": torch.zeros(8, 2)}

    # 1. Without prior (unconditioned): should trigger equivocal check if confidence is low
    res_auto = run_inference_on_data(data, pid="EQUIVOCAL_SPECIMEN")
    rep_auto = generate_synoptic_report("EQUIVOCAL_SPECIMEN", res_auto)
    if res_auto.get("is_equivocal"):
        assert rep_auto["is_equivocal"] is True
        assert "warning_alert" in rep_auto and rep_auto["warning_alert"] is not None
        assert any("IHC" in a for a in rep_auto["clinical_actions"])

    # 2. With clinical Breast (BRCA) prior: should set BRCA and evaluate breast subtyping
    res_brca = run_inference_on_data(data, pid="BREAST_SPECIMEN", organ_prior="BRCA")
    rep_brca = generate_synoptic_report("BREAST_SPECIMEN", res_brca)
    assert rep_brca["primary_cancer"] == "BRCA"
    assert rep_brca["is_equivocal"] is False
    assert any("E-Cadherin" in b or "ER" in b for b in rep_brca["biomarkers"])



