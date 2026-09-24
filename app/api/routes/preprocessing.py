"""API Route for WSI tiling and patch extraction inspection."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException, status

from dataset.patch_extraction import extract_tissue_patches

router = APIRouter(tags=["Preprocessing"])


class ExtractPatchesRequest(BaseModel):
    image_path: str = Field(..., description="Path to input slide/patch image.")
    patch_size: int = Field(224, description="Patch size dimensions in pixels.")
    stride: int = Field(224, description="Tile stride step size in pixels.")
    tissue_threshold: float = Field(0.15, description="Minimum tissue fraction threshold.")


class ExtractPatchesResponse(BaseModel):
    image_path: str
    total_patches: int
    patch_size: int
    tissue_patches: List[Dict[str, Any]]
    research_disclaimer: str = "Experimental Research Prototype - Not For Clinical Diagnostic Use"


@router.post("/preprocessing/extract-patches", response_model=ExtractPatchesResponse, summary="Tile Slide & Extract Tissue Patches")
async def extract_patches_endpoint(payload: ExtractPatchesRequest) -> ExtractPatchesResponse:
    """Extract tissue patch tiles from whole-slide image or large microscopy image."""
    path = Path(payload.image_path)
    if not path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image file not found at path: {payload.image_path}",
        )
    try:
        patches = extract_tissue_patches(
            image_path=payload.image_path,
            patch_size=payload.patch_size,
            stride=payload.stride,
            tissue_threshold=payload.tissue_threshold,
        )
        return ExtractPatchesResponse(
            image_path=payload.image_path,
            total_patches=len(patches),
            patch_size=payload.patch_size,
            tissue_patches=patches,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Patch extraction failed: {str(e)}",
        )
