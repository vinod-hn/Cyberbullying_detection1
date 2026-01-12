"""
Health check endpoint.
GET /health
"""

from fastapi import APIRouter
from datetime import datetime

try:
    from ..schemas import HealthResponse, ModelInfo
    from ..models_loader import list_available_models
except ImportError:
    from schemas import HealthResponse, ModelInfo
    from models_loader import list_available_models

router = APIRouter()


def get_database_status() -> dict:
    """Get database status information."""
    try:
        from ..db_helper import get_db_info
        info = get_db_info()
        return {
            "connected": info.get("exists", False),
            "type": info.get("database_type", "unknown"),
            "path": info.get("database_path", "N/A"),
            "size_bytes": info.get("size_bytes", 0)
        }
    except Exception as e:
        return {
            "connected": False,
            "type": "unknown",
            "error": str(e)
        }


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Check API health status.
    
    Returns current status, available models, and API version.
    """
    available_models = list_available_models()
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        models_available=list(available_models.keys()),
        default_model="bert",
        version="1.0.0"
    )


@router.get("/health/detailed")
async def health_check_detailed():
    """
    Get detailed health status including database and model information.
    """
    available_models = list_available_models()
    db_status = get_database_status()
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "api_version": "1.0.0",
        "database": db_status,
        "models": {
            "available": list(available_models.keys()),
            "default": "bert",
            "count": len(available_models)
        },
        "features": {
            "single_prediction": True,
            "batch_prediction": True,
            "conversation_analysis": True,
            "feedback_collection": True,
            "statistics": True
        }
    }


@router.get("/models", response_model=dict)
async def get_models():
    """
    List all available models with performance metrics.
    
    Returns model information including accuracy, F1 score, and status.
    """
    models = list_available_models()
    
    return {
        "models": [
            ModelInfo(
                model_type=name,
                accuracy=perf["accuracy"],
                f1_score=perf["f1"],
                status="available",
                description=perf.get("description", "")
            )
            for name, perf in models.items()
        ],
        "recommended": "bert",
        "total": len(models)
    }


@router.get("/model/{model_type}/info")
async def get_model_info(model_type: str):
    """
    Get detailed information about a specific model.
    
    - **model_type**: The model to get info for (bert, mbert, indicbert, baseline)
    """
    models = list_available_models()
    
    if model_type not in models:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=404, 
            detail=f"Model '{model_type}' not found. Available: {list(models.keys())}"
        )
    
    perf = models[model_type]
    return {
        "model_type": model_type,
        "accuracy": perf["accuracy"],
        "f1_score": perf["f1"],
        "status": "available",
        "description": perf.get("description", f"{model_type.upper()} classifier"),
        "supported_languages": ["english", "kannada", "code-mixed"]
    }
