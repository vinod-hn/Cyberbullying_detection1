"""
Feedback endpoint for user corrections.
POST /feedback
"""

from fastapi import APIRouter, HTTPException
import uuid
from datetime import datetime
from typing import List

from ..schemas import FeedbackRequest, FeedbackResponse

router = APIRouter()

# In-memory feedback store (replace with database in production)
_feedback_store: List[dict] = []


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
        
        feedback_id = str(uuid.uuid4())
        
        # Store feedback
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
