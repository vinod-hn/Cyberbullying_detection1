"""
Feedback endpoint for user corrections.
POST /feedback
"""

from fastapi import APIRouter, HTTPException
import uuid
from datetime import datetime
from typing import List
import logging

try:
    from ..schemas import FeedbackRequest, FeedbackResponse
except ImportError:
    from schemas import FeedbackRequest, FeedbackResponse

router = APIRouter()
logger = logging.getLogger(__name__)

# In-memory feedback store (fallback if database unavailable)
_feedback_store: List[dict] = []


def save_feedback_to_db(
    prediction_id: str,
    is_correct: bool,
    correct_label: str = None,
    comments: str = None
) -> str:
    """Save feedback to database. Returns feedback_id or None."""
    try:
        from ..db_helper import get_db_context, get_feedback_repository
        
        ctx = get_db_context()
        if ctx is None:
            return None
        
        with ctx as db:
            repo = get_feedback_repository(db)
            if repo:
                feedback = repo.create(
                    prediction_id=prediction_id,
                    is_correct=is_correct,
                    correct_label=correct_label,
                    comments=comments
                )
                logger.info(f"Feedback saved to database: {feedback.feedback_id}")
                return feedback.feedback_id
        return None
    except Exception as e:
        logger.warning(f"Failed to save feedback to database: {e}")
        return None


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest):
    """
    Submit feedback on a prediction for model improvement.
    
    - **prediction_id**: The ID of the prediction being corrected
    - **correct_label**: The correct classification (cyberbullying/not_cyberbullying)
    - **comments**: Optional additional context
    
    Feedback is stored for future model retraining and analysis.
    """
    try:
        # Validate correct_label
        valid_labels = ["cyberbullying", "not_cyberbullying"]
        if request.correct_label.lower() not in valid_labels:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid label. Must be one of: {valid_labels}"
            )
        
        # Determine if prediction was correct
        is_correct = request.correct_label.lower() == "not_cyberbullying"
        
        # Try to save to database first
        feedback_id = save_feedback_to_db(
            prediction_id=request.prediction_id,
            is_correct=is_correct,
            correct_label=request.correct_label.lower(),
            comments=request.comments
        )
        
        # Fallback to in-memory store if database fails
        if not feedback_id:
            feedback_id = str(uuid.uuid4())
            feedback_entry = {
                "feedback_id": feedback_id,
                "prediction_id": request.prediction_id,
                "correct_label": request.correct_label.lower(),
                "comments": request.comments,
                "timestamp": datetime.now().isoformat(),
            }
            _feedback_store.append(feedback_entry)
        
        return FeedbackResponse(
            success=True,
            message="Thank you for your feedback! It will help improve our models.",
            feedback_id=feedback_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to submit feedback: {str(e)}")


@router.get("/feedback/count", include_in_schema=False)
async def get_feedback_count():
    """Get total feedback count (admin endpoint)."""
    return {"total_feedback": len(_feedback_store)}
