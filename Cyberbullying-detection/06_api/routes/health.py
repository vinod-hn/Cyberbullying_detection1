"""
Health check endpoint.
GET /health
"""

from fastapi import APIRouter
from datetime import datetime

from ..schemas import HealthResponse, ModelInfo
from ..models_loader import list_available_models

router = APIRouter()


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
