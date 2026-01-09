"""
Cyberbullying Detection API Package.

Usage:
    uvicorn 06_api.main:app --reload
"""

# Be defensive: allow imports both as a package and via direct module loading
try:
    from .main import app
    from .app_config import settings, get_settings
    from .models_loader import (
        CyberbullyingDetector,
        get_detector,
        list_available_models,
        get_best_model,
    )
except Exception:  # pragma: no cover - fallback for non-package import contexts
    import importlib
    _main = importlib.import_module("06_api.main")
    app = getattr(_main, "app")
    _config = importlib.import_module("06_api.app_config")
    settings = getattr(_config, "settings")
    get_settings = getattr(_config, "get_settings")
    _ml = importlib.import_module("06_api.models_loader")
    CyberbullyingDetector = getattr(_ml, "CyberbullyingDetector")
    get_detector = getattr(_ml, "get_detector")
    list_available_models = getattr(_ml, "list_available_models")
    get_best_model = getattr(_ml, "get_best_model")

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
