"""
Single text prediction endpoint.
POST /predict
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Optional
import uuid

from ..schemas import PredictionRequest, PredictionResponse
from ..models_loader import get_detector

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict if a single text contains cyberbullying.
    
    - **text**: The text to analyze
    - **model_type**: Model to use (bert, mbert, indicbert, baseline)
    
    Returns prediction with confidence score and probabilities.
    """
    try:
        detector = get_detector(request.model_type)
        result = detector.predict(request.text)
        
        return PredictionResponse(
            text=result["text"],
            prediction=result["prediction"],
            confidence=result["confidence"],
            is_cyberbullying=result["is_cyberbullying"],
            probabilities=result.get("probabilities"),
            prediction_id=str(uuid.uuid4())
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
