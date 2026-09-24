from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Schema for basic system health status."""
    status: str = Field("ok", description="Overall service status")
    service: str = Field("histopathology-ai-backend", description="Service identifier")
    version: str = Field("1.0.0", description="Backend application version")
    disclaimer: str = Field(
        ...,
        description="Research-use disclaimer declaring that outputs are non-clinical experimental predictions."
    )


class ModelHealthResponse(BaseModel):
    """Schema for AI model state and hardware diagnostic health."""
    status: str = Field("ok", description="Model subsystem status")
    model_loaded: bool = Field(False, description="Indicates if deep learning model weights are loaded")
    device: str = Field(..., description="Active compute device (cuda, mps, cpu)")
    model_version: str = Field(..., description="Target model framework version")
    checkpoint_status: str = Field("not_loaded", description="Checkpoint loading status")
    disclaimer: str = Field(
        ...,
        description="Research-use disclaimer declaring that outputs are non-clinical experimental predictions."
    )
