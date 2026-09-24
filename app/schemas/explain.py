"""Pydantic schemas for explain endpoint."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ExplainRequest(BaseModel):
    image_path: str = Field(..., description="Path to the histopathology image file.")
    organ_type: Optional[str] = Field("auto", description="Organ/tissue origin: 'breast', 'lung', 'brain', or 'auto'.")
    checkpoint_path: Optional[str] = Field(None, description="Path to model .pt checkpoint.")
    target_class_index: Optional[int] = Field(None, description="Optional target cancer class index for Grad-CAM.")
    evaluate_eval_metrics: bool = Field(True, description="Whether to compute faithfulness and stability metrics.")


class ExplainResponse(BaseModel):
    research_disclaimer: str = Field(...)
    structured_findings: Dict[str, Any] = Field(...)
    findings_summary: str = Field(...)
    xai_combination: Dict[str, Any] = Field(...)
