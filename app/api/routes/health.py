from fastapi import APIRouter
from app.core.config import settings
from app.schemas.health import HealthResponse, ModelHealthResponse

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Get Service Health Status",
    description="Returns the health status of the histopathology research AI backend service."
)
async def get_health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        service=settings.SERVICE_NAME,
        version=settings.VERSION,
        disclaimer=settings.RESEARCH_DISCLAIMER
    )


@router.get(
    "/health/model",
    response_model=ModelHealthResponse,
    summary="Get Deep Learning Model Diagnostics",
    description="Returns hardware compute target, model initialization state, and checkpoint loading status."
)
async def get_model_health() -> ModelHealthResponse:
    device = settings.get_device()
    return ModelHealthResponse(
        status="ok",
        model_loaded=False,
        device=device,
        model_version=settings.MODEL_VERSION,
        checkpoint_status="not_loaded",
        disclaimer=settings.RESEARCH_DISCLAIMER
    )
