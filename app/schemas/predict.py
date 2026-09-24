"""Pydantic schemas for prediction endpoint."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    image_path: str = Field(..., description="Path to the histopathology image file.")
    organ_type: Optional[str] = Field("auto", description="Organ/tissue origin: 'breast', 'lung', 'brain', or 'auto'.")
    checkpoint_path: Optional[str] = Field(None, description="Path to model .pt checkpoint.")


class SubtypePrediction(BaseModel):
    label: Optional[str] = None
    confidence: Optional[float] = None
    probabilities: Dict[str, float] = Field(default_factory=dict)
    message: Optional[str] = None


class PredictResponse(BaseModel):
    research_use_only: bool = True
    research_disclaimer: str = Field(...)
    image_path: str = Field(...)
    organ_type: str = Field("Auto", description="Specified or resolved organ tissue origin.")
    has_cancer: bool = Field(True, description="Whether cancer is predicted present.")
    cancer_status: str = Field(..., description="Summary status (e.g. 'Cancer Detected (BRCA)').")
    cancer_type: str = Field(...)
    cancer_confidence: float = Field(...)
    top2_confidence: float = Field(0.0)
    cancer_probabilities: Dict[str, float] = Field(...)
    entropy: float = Field(0.0)
    is_low_confidence: bool = Field(False)
    confidence_warning: Optional[str] = None
    subtype: SubtypePrediction = Field(...)
    scale_weights: Optional[List[float]] = None
    patch_weights: Optional[List[float]] = None
