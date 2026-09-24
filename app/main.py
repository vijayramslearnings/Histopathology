import os
import sys

# Ensure backend root directory and parent directory are in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
PARENT_DIR = os.path.dirname(BASE_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

import time
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.core.logging import logger
from app.core.exceptions import (
    PathologyException,
    pathology_exception_handler,
    generic_exception_handler,
)
from app.api.routes import health, predict, explain, chat, validate_image, analyze_error, report, preprocessing


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {settings.PROJECT_NAME} v{settings.VERSION}")
    logger.info(f"Environment: {settings.APP_ENV} | Active Device: {settings.get_device()}")
    logger.info("RESEARCH NOTICE: System initialized in Experimental Research Mode.")
    
    # Ensure outputs directory exists for serving static media
    outputs_path = Path("outputs")
    outputs_path.mkdir(parents=True, exist_ok=True)
    yield
    logger.info(f"Shutting down {settings.PROJECT_NAME}")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=(
        "Research prototype backend for computational pathology: "
        "Explainable Multi-Scale Attention-Based Deep Learning Framework for Multi-Cancer and Subtype Classification.\n\n"
        "**RESEARCH-USE DISCLAIMER**: This system is strictly an experimental research prototype. "
        "It is NOT a medical device, nor a clinical diagnostic system. "
        "Outputs must never be used to make clinical decisions or formulate diagnostic conclusions."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception Handlers
app.add_exception_handler(PathologyException, pathology_exception_handler)  # type: ignore
app.add_exception_handler(Exception, generic_exception_handler)


@app.middleware("http")
async def add_research_disclaimer_header_and_logging(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000

    # Attach mandatory research disclaimer header
    response.headers["X-Research-Disclaimer"] = (
        "Experimental Research Prototype - Not For Clinical Diagnostic Use"
    )
    response.headers["X-Process-Time-Ms"] = f"{process_time:.2f}"

    logger.info(
        f"{request.method} {request.url.path} - Status: {response.status_code} - Time: {process_time:.2f}ms"
    )
    return response


# Mount static directory for generated heatmaps, overlays, and region crop assets
outputs_dir = Path("outputs")
outputs_dir.mkdir(parents=True, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=str(outputs_dir)), name="outputs")

@app.get("/", summary="Root Welcome Endpoint")
@app.get(settings.API_V1_STR, summary="API v1 Welcome Endpoint")
async def root():
    return {
        "status": "ok",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "docs_url": "/docs",
        "health_url": "/health",
        "endpoints": {
            "predict": "/api/v1/predict",
            "predict_upload": "/api/v1/predict/upload",
            "explain": "/api/v1/explain",
            "chat": "/api/v1/chat",
            "health": "/api/v1/health",
            "validate_image": "/api/v1/validate-image",
            "analyze_error": "/api/v1/analyze-error",
            "generate_report": "/api/v1/generate-report",
            "extract_patches": "/api/v1/preprocessing/extract-patches",
        },
        "disclaimer": settings.RESEARCH_DISCLAIMER,
    }


# Include Routers at root level
app.include_router(health.router)
app.include_router(predict.router)
app.include_router(explain.router)
app.include_router(chat.router)
app.include_router(validate_image.router)
app.include_router(analyze_error.router)
app.include_router(report.router)
app.include_router(preprocessing.router)

# Also mount under API v1 prefix
app.include_router(health.router, prefix=settings.API_V1_STR)
app.include_router(predict.router, prefix=settings.API_V1_STR)
app.include_router(explain.router, prefix=settings.API_V1_STR)
app.include_router(chat.router, prefix=settings.API_V1_STR)
app.include_router(validate_image.router, prefix=settings.API_V1_STR)
app.include_router(analyze_error.router, prefix=settings.API_V1_STR)
app.include_router(report.router, prefix=settings.API_V1_STR)
app.include_router(preprocessing.router, prefix=settings.API_V1_STR)

