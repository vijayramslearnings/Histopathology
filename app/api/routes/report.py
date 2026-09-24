"""API Route for generating automated PDF research reports."""

from __future__ import annotations

from typing import Any, Dict
from fastapi import APIRouter, HTTPException, Response, status
from pydantic import BaseModel, Field

from app.services.report_generator import generate_pdf_report

router = APIRouter(tags=["Report Generation"])


class GenerateReportRequest(BaseModel):
    analysis_result: Dict[str, Any] = Field(..., description="Full prediction and XAI findings JSON object.")


@router.post("/generate-report", summary="Generate Downloadable Research PDF Report")
async def generate_report_endpoint(payload: GenerateReportRequest) -> Response:
    """Synthesize predictions, multi-scale attention, uncertainty, and XAI findings into a research PDF report."""
    try:
        pdf_bytes = generate_pdf_report(payload.analysis_result)
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": "attachment; filename=histopathology_ai_report.pdf",
                "X-Research-Disclaimer": "Experimental Research Prototype - Not For Clinical Diagnostic Use",
            },
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"PDF report generation failed: {str(e)}",
        )
