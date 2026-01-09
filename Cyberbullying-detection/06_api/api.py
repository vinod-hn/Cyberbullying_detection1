"""
Cyberbullying Detection API - FastAPI Routes

Provides REST API endpoints for cyberbullying detection using pre-trained models.

Endpoints:
    POST /predict - Single text prediction
    POST /predict/batch - Batch prediction
    GET /models - List available models
    GET /health - Health check

Usage:
    uvicorn api:app --host 0.0.0.0 --port 8000
"""

import os
import sys
from pathlib import Path
from typing import List, Optional
from datetime import datetime
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI imports
try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import Response, FileResponse
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logger.error("FastAPI not installed. Run: pip install fastapi uvicorn")

# Import detector
try:
    from model_loader import CyberbullyingDetector, list_available_models, get_best_model
except ImportError:
    # Try relative import
    sys.path.insert(0, str(PROJECT_ROOT / '03_models'))
    from model_loader import CyberbullyingDetector, list_available_models, get_best_model


# =============================================================================
# Pydantic Models
# =============================================================================

class PredictionRequest(BaseModel):
    """Request model for single prediction."""
    text: str = Field(..., description="Text to analyze for cyberbullying", min_length=1)
    model_type: Optional[str] = Field(
        'bert', 
        description="Model to use: bert, mbert, indicbert, baseline"
    )

class BatchPredictionRequest(BaseModel):
    """Request model for batch prediction."""
    texts: List[str] = Field(..., description="List of texts to analyze", min_length=1)
    model_type: Optional[str] = Field('bert', description="Model to use")
    batch_size: Optional[int] = Field(32, description="Processing batch size")

class PredictionResponse(BaseModel):
    """Response model for prediction."""
    text: str
    prediction: str
    confidence: float
    is_cyberbullying: bool
    probabilities: Optional[dict] = None

class BatchPredictionResponse(BaseModel):
    """Response model for batch prediction."""
    predictions: List[PredictionResponse]
    total: int
    processing_time_ms: float

class ModelInfo(BaseModel):
    """Model information response."""
    model_type: str
    accuracy: float
    f1_score: float
    status: str

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    models_available: List[str]
    default_model: str


# =============================================================================
# FastAPI App
# =============================================================================

if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="Cyberbullying Detection API",
        description="""
        API for detecting cyberbullying in text using pre-trained ML models.
        
        ## Features
        - Multiple model options (BERT, mBERT, IndicBERT, Baseline)
        - Single and batch prediction endpoints
        - Confidence scores and probabilities
        - Support for Kannada-English code-mixed text
        
        ## Models Performance
        - **BERT**: 99.88% F1 Score (Best)
        - **mBERT**: 99.57% F1 Score (Multilingual)
        - **IndicBERT**: 99.76% F1 Score (Indian Languages)
        - **Baseline**: ~95% F1 Score (Fastest)
        """,
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Model cache
    _model_cache = {}

    # Minimal transparent 1x1 PNG as favicon (base64)
    FAVICON_PNG_BASE64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
    )

    def get_detector(model_type: str = 'bert') -> CyberbullyingDetector:
        """Get or create a detector instance."""
        if model_type not in _model_cache:
            try:
                _model_cache[model_type] = CyberbullyingDetector(model_type=model_type)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
        return _model_cache[model_type]


    # =============================================================================
    # Endpoints
    # =============================================================================

    @app.get("/", tags=["Root"])
    async def root():
        """API root endpoint."""
        return {
            "message": "Cyberbullying Detection API",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/health"
        }

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon():
        """Serve a tiny built-in favicon to avoid 404s from browsers."""
        import base64
        return Response(base64.b64decode(FAVICON_PNG_BASE64), media_type="image/png")

    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    async def health_check():
        """Check API health status."""
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now().isoformat(),
            models_available=list(list_available_models().keys()),
            default_model="bert"
        )

    @app.get("/models", tags=["Models"])
    async def get_models():
        """List all available models with performance metrics."""
        models = list_available_models()
        return {
            "models": [
                ModelInfo(
                    model_type=name,
                    accuracy=perf['accuracy'],
                    f1_score=perf['f1'],
                    status="available"
                )
                for name, perf in models.items()
            ],
            "recommended": "bert",
            "total": len(models)
        }

    @app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
    async def predict(request: PredictionRequest):
        """
        Predict if a single text contains cyberbullying.
        
        - **text**: The text to analyze
        - **model_type**: Model to use (bert, mbert, indicbert, baseline)
        """
        try:
            detector = get_detector(request.model_type)
            result = detector.predict(request.text)
            
            return PredictionResponse(
                text=result['text'],
                prediction=result['prediction'],
                confidence=result['confidence'],
                is_cyberbullying=result['is_cyberbullying'],
                probabilities=result.get('probabilities')
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
    async def predict_batch(request: BatchPredictionRequest):
        """
        Predict cyberbullying for multiple texts.
        
        - **texts**: List of texts to analyze
        - **model_type**: Model to use
        - **batch_size**: Processing batch size
        """
        import time
        start_time = time.time()
        
        try:
            detector = get_detector(request.model_type)
            results = detector.predict_batch(request.texts, batch_size=request.batch_size)
            
            processing_time = (time.time() - start_time) * 1000
            
            return BatchPredictionResponse(
                predictions=[
                    PredictionResponse(
                        text=r['text'],
                        prediction=r['prediction'],
                        confidence=r['confidence'],
                        is_cyberbullying=r['is_cyberbullying'],
                        probabilities=r.get('probabilities')
                    )
                    for r in results
                ],
                total=len(results),
                processing_time_ms=processing_time
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/model/{model_type}/info", tags=["Models"])
    async def get_model_info(model_type: str):
        """Get detailed information about a specific model."""
        try:
            detector = get_detector(model_type)
            return detector.get_model_info()
        except Exception as e:
            raise HTTPException(status_code=404, detail=str(e))


else:
    app = None
    print("FastAPI not available. Install with: pip install fastapi uvicorn")


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    if FASTAPI_AVAILABLE:
        import uvicorn
        print("\n🚀 Starting Cyberbullying Detection API...")
        print("📖 API Docs: http://localhost:8000/docs")
        print("🔄 Health Check: http://localhost:8000/health\n")
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        print("❌ FastAPI not installed. Run: pip install fastapi uvicorn")
