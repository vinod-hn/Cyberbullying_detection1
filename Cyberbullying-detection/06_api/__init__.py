"""
Cyberbullying Detection API Package.

Usage:
    uvicorn 06_api.main:app --reload
"""

from .main import app
from .app_config import settings, get_settings
from .models_loader import (
    CyberbullyingDetector,
    get_detector,
    list_available_models,
    get_best_model,
)

__version__ = "1.0.0"

__all__ = [
    "app",
    "settings",
    "get_settings",
    "CyberbullyingDetector",
    "get_detector",
    "list_available_models",
    "get_best_model",
]
