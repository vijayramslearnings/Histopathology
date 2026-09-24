from typing import Any, Optional, Dict
from fastapi import Request, status
from fastapi.responses import JSONResponse
from app.core.logging import logger


class PathologyException(Exception):
    """Base exception class for pathology backend errors."""
    code: str = "INTERNAL_SERVER_ERROR"
    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class InvalidImageError(PathologyException):
    code = "INVALID_IMAGE"
    status_code = status.HTTP_400_BAD_REQUEST


class UnsupportedFormatError(PathologyException):
    code = "UNSUPPORTED_FORMAT"
    status_code = status.HTTP_415_UNSUPPORTED_MEDIA_TYPE


class ModelLoadError(PathologyException):
    code = "MODEL_LOAD_ERROR"
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR


class PredictionError(PathologyException):
    code = "PREDICTION_ERROR"
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR


class DatasetError(PathologyException):
    code = "DATASET_ERROR"
    status_code = status.HTTP_400_BAD_REQUEST


class DataLeakageError(PathologyException):
    code = "DATA_LEAKAGE_DETECTED"
    status_code = status.HTTP_422_UNPROCESSABLE_CONTENT


class XAIError(PathologyException):
    code = "XAI_ERROR"
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR


class ReportGenerationError(PathologyException):
    code = "REPORT_GENERATION_ERROR"
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR


async def pathology_exception_handler(request: Request, exc: PathologyException) -> JSONResponse:
    """FastAPI Exception Handler for custom PathologyException hierarchy."""
    logger.error(f"PathologyException [{exc.code}]: {exc.message} - Path: {request.url.path}")
    content = {
        "error": {
            "code": exc.code,
            "message": exc.message,
        }
    }
    if exc.details:
        content["error"]["details"] = exc.details
    return JSONResponse(status_code=exc.status_code, content=content)


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all exception handler preventing stack traces from being exposed."""
    logger.error(f"Unhandled Exception: {str(exc)} - Path: {request.url.path}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred during processing."
            }
        }
    )
