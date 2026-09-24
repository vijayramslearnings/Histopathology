"""API Route for misclassification error analysis and archiving."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException, status

from xai.error_analysis import analyze_prediction_error

router = APIRouter(tags=["Error Analysis"])


class AnalyzeErrorRequest(BaseModel):
    image_path: str = Field(..., description="Path to the sample image.")
    actual_label: str = Field(..., description="Ground truth cancer label (e.g. LUAD, BRCA, GBM).")
    predicted_label: str = Field(..., description="Predicted cancer label (e.g. BRCA).")
    confidence: float = Field(..., description="Model confidence score.")
    class_probabilities: Dict[str, float] = Field(..., description="Full class probability map.")
    scale_attention: Optional[Dict[str, float]] = Field(default=None, description="Optional scale attention weights.")


class AnalyzeErrorResponse(BaseModel):
    image_path: str
    actual_label: str
    predicted_label: str
    is_correct: bool
    error_type: str
    confidence: float
    class_probabilities: Dict[str, float]
    scale_attention: Dict[str, float]
    error_archive_folder: Optional[str] = None
    research_disclaimer: str = "Experimental Research Prototype - Not For Clinical Diagnostic Use"


@router.post("/analyze-error", response_model=AnalyzeErrorResponse, summary="Archive and Log Misclassified Prediction")
async def analyze_error_endpoint(payload: AnalyzeErrorRequest) -> AnalyzeErrorResponse:
    """Analyze and log misclassified images to outputs/errors/{actual}_to_{predicted}/ for error research."""
    path = Path(payload.image_path)
    if not path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image file not found at path: {payload.image_path}",
        )
    try:
        record = analyze_prediction_error(
            image_path=payload.image_path,
            actual_label=payload.actual_label,
            predicted_label=payload.predicted_label,
            confidence=payload.confidence,
            class_probabilities=payload.class_probabilities,
            scale_attention=payload.scale_attention,
        )
        return AnalyzeErrorResponse(**record)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error analysis logging failed: {str(e)}",
        )
