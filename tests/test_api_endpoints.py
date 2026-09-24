"""Integration tests for FastAPI REST API endpoints."""

from __future__ import annotations

import tempfile
from pathlib import Path
from PIL import Image
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "disclaimer" in data


def test_predict_and_explain_and_chat_endpoints():
    with tempfile.TemporaryDirectory() as tmp_dir:
        img_path = Path(tmp_dir) / "sample_slide.png"
        Image.new("RGB", (224, 224), color="green").save(img_path)

        # 1. POST /predict
        pred_res = client.post("/predict", json={"image_path": str(img_path)})
        assert pred_res.status_code == 200
        pred_data = pred_res.json()
        assert "cancer_type" in pred_data
        assert "cancer_confidence" in pred_data
        assert "cancer_probabilities" in pred_data

        # 2. POST /explain
        explain_res = client.post(
            "/explain",
            json={
                "image_path": str(img_path),
                "evaluate_eval_metrics": False,
            },
        )
        assert explain_res.status_code == 200
        explain_data = explain_res.json()
        assert "structured_findings" in explain_data
        assert "findings_summary" in explain_data

        # 3. POST /chat
        chat_res = client.post(
            "/chat",
            json={
                "message": "What did the model predict?",
                "analysis_result": explain_data["structured_findings"],
                "session_id": "test_api_session",
            },
        )
        assert chat_res.status_code == 200
        chat_data = chat_res.json()
        assert "answer" in chat_data
        assert "research_disclaimer" in chat_data

        # 4. POST /validate-image
        val_res = client.post("/validate-image", json={"image_path": str(img_path)})
        assert val_res.status_code == 200
        val_data = val_res.json()
        assert "quality" in val_data
        assert "is_acceptable" in val_data

        # 5. POST /analyze-error
        err_res = client.post(
            "/analyze-error",
            json={
                "image_path": str(img_path),
                "actual_label": "LUAD",
                "predicted_label": "BRCA",
                "confidence": 0.86,
                "class_probabilities": {"BRCA": 0.86, "LUAD": 0.11, "GBM": 0.03},
            },
        )
        assert err_res.status_code == 200
        err_data = err_res.json()
        assert err_data["error_type"] == "LUAD_to_BRCA"
        assert err_data["is_correct"] is False
