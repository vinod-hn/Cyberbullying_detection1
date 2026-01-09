"""
Batch prediction endpoint.
POST /predict/batch
"""

from fastapi import APIRouter, HTTPException
import time
import uuid

from ..schemas import BatchPredictionRequest, BatchPredictionResponse, PredictionResponse
from ..models_loader import get_detector

router = APIRouter()


@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict cyberbullying for multiple texts.
    
    - **texts**: List of texts to analyze
    - **model_type**: Model to use
    - **batch_size**: Processing batch size (1-128)
    
    Returns predictions for all texts with total processing time.
    """
    start_time = time.time()
    
    try:
        detector = get_detector(request.model_type)
        results = detector.predict_batch(request.texts, batch_size=request.batch_size)
        
        processing_time = (time.time() - start_time) * 1000
        
        predictions = [
            PredictionResponse(
                text=r["text"],
                prediction=r["prediction"],
                confidence=r["confidence"],
                is_cyberbullying=r["is_cyberbullying"],
                probabilities=r.get("probabilities"),
                prediction_id=str(uuid.uuid4())
            )
            for r in results
        ]
        
        return BatchPredictionResponse(
            predictions=predictions,
            total=len(predictions),
            processing_time_ms=processing_time
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")
