"""
Conversation-level prediction endpoint.
POST /predict/conversation
"""

from fastapi import APIRouter, HTTPException
import time
from typing import List

from ..schemas import (
    ConversationPredictionRequest,
    ConversationPredictionResponse,
    ConversationMessageResult,
)
from ..models_loader import get_detector

router = APIRouter()


def detect_escalation(results: List[dict]) -> bool:
    """
    Detect if there's an escalation pattern in the conversation.
    Escalation = increasing severity or confidence over consecutive messages.
    """
    if len(results) < 3:
        return False
    
    # Check for pattern of increasing cyberbullying indicators
    bullying_streak = 0
    max_streak = 0
    
    for r in results:
        if r["is_cyberbullying"]:
            bullying_streak += 1
            max_streak = max(max_streak, bullying_streak)
        else:
            bullying_streak = 0
    
    # Escalation if 3+ consecutive bullying messages or increasing confidence
    return max_streak >= 3


def calculate_risk_score(results: List[dict]) -> float:
    """
    Calculate overall conversation risk score (0-1).
    """
    if not results:
        return 0.0
    
    bullying_count = sum(1 for r in results if r["is_cyberbullying"])
    avg_confidence = sum(r["confidence"] for r in results if r["is_cyberbullying"]) / max(bullying_count, 1)
    
    # Weighted score: ratio of bullying messages + average confidence
    ratio_score = bullying_count / len(results)
    risk_score = (ratio_score * 0.6) + (avg_confidence * 0.4 * ratio_score)
    
    return min(risk_score, 1.0)


@router.post("/predict/conversation", response_model=ConversationPredictionResponse)
async def predict_conversation(request: ConversationPredictionRequest):
    """
    Analyze an entire conversation for cyberbullying patterns.
    
    - **messages**: List of conversation messages in chronological order
    - **model_type**: Model to use
    - **include_context**: Whether to use conversation context for prediction
    
    Returns per-message predictions plus overall risk assessment.
    """
    start_time = time.time()
    
    try:
        detector = get_detector(request.model_type)
        
        # Extract texts for batch prediction
        texts = [msg.text for msg in request.messages]
        results = detector.predict_batch(texts, batch_size=32)
        
        # Build response with context scores
        message_results = []
        flagged_indices = []
        
        for i, r in enumerate(results):
            # Context score could factor in surrounding messages
            context_score = None
            if request.include_context and len(results) > 1:
                # Simple context: average of neighbors
                neighbors = []
                if i > 0:
                    neighbors.append(results[i-1]["confidence"] if results[i-1]["is_cyberbullying"] else 0)
                if i < len(results) - 1:
                    neighbors.append(results[i+1]["confidence"] if results[i+1]["is_cyberbullying"] else 0)
                context_score = sum(neighbors) / len(neighbors) if neighbors else 0.0
            
            message_results.append(ConversationMessageResult(
                text=r["text"],
                prediction=r["prediction"],
                confidence=r["confidence"],
                is_cyberbullying=r["is_cyberbullying"],
                context_score=context_score
            ))
            
            if r["is_cyberbullying"]:
                flagged_indices.append(i)
        
        processing_time = (time.time() - start_time) * 1000
        
        return ConversationPredictionResponse(
            messages=message_results,
            overall_risk_score=calculate_risk_score(results),
            escalation_detected=detect_escalation(results),
            flagged_messages=flagged_indices,
            processing_time_ms=processing_time
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Conversation prediction failed: {str(e)}")
