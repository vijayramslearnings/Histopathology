"""API Route for image quality validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException, status

from xai.quality import validate_image_quality

router = APIRouter(tags=["Image Validation"])


class ValidateImageRequest(BaseModel):
    image_path: str = Field(..., description="Absolute or relative path to the image file to inspect.")


class ValidateImageResponse(BaseModel):
    quality: str
    is_acceptable: bool
    warnings: List[str]
    tissue_percentage: float
    blur_score: float
    mean_brightness: float
    dimensions: Dict[str, int]
    research_disclaimer: str = "Experimental Research Prototype - Not For Clinical Diagnostic Use"


@router.post("/validate-image", response_model=ValidateImageResponse, summary="Inspect Input Image Quality & Tissue Content")
async def validate_image_endpoint(payload: ValidateImageRequest) -> ValidateImageResponse:
    """Validate image resolution, tissue content percentage, brightness, and focus blur prior to inference."""
    path = Path(payload.image_path)
    if not path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image file not found at path: {payload.image_path}",
        )
    try:
        report = validate_image_quality(payload.image_path)
        return ValidateImageResponse(**report)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Image validation failed: {str(e)}",
        )
