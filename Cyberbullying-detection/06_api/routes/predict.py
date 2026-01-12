"""
Single text prediction endpoint.
POST /predict
"""

from fastapi import APIRouter, HTTPException, Request
import uuid
import time
import logging

try:
    from ..schemas import PredictionRequest, PredictionResponse
    from ..models_loader import get_detector
except ImportError:
    from schemas import PredictionRequest, PredictionResponse
    from models_loader import get_detector

router = APIRouter()
logger = logging.getLogger(__name__)


def save_prediction_to_db(
    text: str,
    model_type: str,
    prediction: str,
    confidence: float,
    is_cyberbullying: bool,
    probabilities: dict,
    prediction_id: str,
    inference_time_ms: float = None,
    ip_address: str = None
):
    """Save prediction to database if available."""
    try:
        from ..db_helper import get_db_context, get_prediction_repository, get_audit_log_repository
        
        ctx = get_db_context()
        if ctx is None:
            return False
            
        with ctx as db:
            pred_repo = get_prediction_repository(db)
            if pred_repo:
                message, pred = pred_repo.create_with_message(
                    text=text,
                    model_type=model_type,
                    predicted_label=prediction,
                    confidence=confidence,
                    is_cyberbullying=is_cyberbullying,
                    probabilities=probabilities,
                    inference_time_ms=inference_time_ms,
                    source="api"
                )
                
                # Log the prediction
                audit_repo = get_audit_log_repository(db)
                if audit_repo:
                    audit_repo.log_prediction(
                        prediction_id=prediction_id,
                        ip_address=ip_address,
                        model_type=model_type
                    )
                
                logger.debug(f"Prediction saved to database: {prediction_id}")
                return True
        return False
    except Exception as e:
        logger.warning(f"Failed to save prediction to database: {e}")
        return False


@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest, req: Request = None):
    """
    Predict if a single text contains cyberbullying.
    
    - **text**: The text to analyze
    - **model_type**: Model to use (bert, mbert, indicbert, baseline)
    
    Returns prediction with confidence score and probabilities.
    """
    try:
        start_time = time.time()
        
        detector = get_detector(request.model_type)
        result = detector.predict(request.text)
        
        inference_time_ms = (time.time() - start_time) * 1000
        prediction_id = str(uuid.uuid4())
        
        # Get client IP if available
        ip_address = None
        if req:
            ip_address = req.client.host if req.client else None
        
        # Save to database (async-friendly, non-blocking)
        save_prediction_to_db(
            text=result["text"],
            model_type=request.model_type,
            prediction=result["prediction"],
            confidence=result["confidence"],
            is_cyberbullying=result["is_cyberbullying"],
            probabilities=result.get("probabilities", {}),
            prediction_id=prediction_id,
            inference_time_ms=inference_time_ms,
            ip_address=ip_address
        )
        
        return PredictionResponse(
            text=result["text"],
            prediction=result["prediction"],
            confidence=result["confidence"],
            is_cyberbullying=result["is_cyberbullying"],
            probabilities=result.get("probabilities"),
            prediction_id=prediction_id
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
