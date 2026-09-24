import os
from pathlib import Path
from typing import List, Optional
import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application Settings for the Histopathology AI Research Backend.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Project Info
    PROJECT_NAME: str = "Explainable Multi-Scale Attention Histopathology AI"
    SERVICE_NAME: str = "histopathology-ai-backend"
    VERSION: str = "1.0.0"
    APP_ENV: str = "development"
    DEBUG: bool = True
    API_V1_STR: str = "/api/v1"
    HOST: str = "127.0.0.1"
    PORT: int = 8000

    # Research Disclaimer
    RESEARCH_DISCLAIMER: str = (
        "Experimental research prediction only. "
        "Not a clinical diagnostic system. Not for clinical use."
    )

    # Target Cancer Types
    CANCER_TYPES: List[str] = ["BRCA", "LUAD", "GBM"]
    
    # Hardware & Model Configuration
    DEFAULT_DEVICE: str = "auto"
    MODEL_VERSION: str = "1.0.0"
    CHECKPOINT_DIR: Path = Path("checkpoints")
    OUTPUT_DIR: Path = Path("outputs")
    CONFIG_PATH: Path = Path("configs/config.yaml")

    # Database
    DATABASE_URL: str = "sqlite:///./pathology.db"

    def get_device(self) -> str:
        """
        Detect and return available compute hardware (CUDA > MPS > CPU).
        """
        if self.DEFAULT_DEVICE.lower() != "auto":
            return self.DEFAULT_DEVICE.lower()
        
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"


settings = Settings()
