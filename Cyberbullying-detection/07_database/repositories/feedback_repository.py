"""
Feedback Repository - CRUD operations for prediction feedback.
"""

from typing import List, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import desc, func

# Handle both relative and absolute imports
try:
    from ..models import Feedback
except ImportError:
    from models import Feedback


class FeedbackRepository:
    """Repository for Feedback CRUD operations."""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create(
        self,
        prediction_id: str,
        is_correct: bool,
        correct_label: Optional[str] = None,
        comments: Optional[str] = None
    ) -> Feedback:
        """Create a new feedback entry."""
        feedback = Feedback(
            prediction_id=prediction_id,
            is_correct=is_correct,
            correct_label=correct_label,
            comments=comments
        )
        self.db.add(feedback)
        self.db.commit()
        self.db.refresh(feedback)
        return feedback
    
    def get_by_id(self, feedback_id: int) -> Optional[Feedback]:
        """Get feedback by ID."""
        return self.db.query(Feedback).filter(Feedback.id == feedback_id).first()
    
    def get_by_uuid(self, feedback_uuid: str) -> Optional[Feedback]:
        """Get feedback by UUID."""
        return self.db.query(Feedback).filter(Feedback.feedback_id == feedback_uuid).first()
    
    def get_by_prediction(self, prediction_id: str) -> List[Feedback]:
        """Get all feedback for a prediction."""
        return self.db.query(Feedback).filter(
            Feedback.prediction_id == prediction_id
        ).order_by(desc(Feedback.created_at)).all()
    
    def get_all(
        self,
        skip: int = 0,
        limit: int = 100,
        is_correct: Optional[bool] = None
    ) -> List[Feedback]:
        """Get all feedback with filters and pagination."""
        query = self.db.query(Feedback)
        
        if is_correct is not None:
            query = query.filter(Feedback.is_correct == is_correct)
        
        return query.order_by(desc(Feedback.created_at)).offset(skip).limit(limit).all()
    
    def get_incorrect_predictions(self, limit: int = 100) -> List[Feedback]:
        """Get feedback indicating incorrect predictions."""
        return self.db.query(Feedback).filter(
            Feedback.is_correct == False
        ).order_by(desc(Feedback.created_at)).limit(limit).all()
    
    def count(self, is_correct: Optional[bool] = None) -> int:
        """Count feedback with optional filter."""
        query = self.db.query(func.count(Feedback.id))
        
        if is_correct is not None:
            query = query.filter(Feedback.is_correct == is_correct)
        
        return query.scalar()
    
    def get_accuracy_stats(self) -> dict:
        """Get model accuracy statistics based on feedback."""
        total = self.count()
        correct = self.count(is_correct=True)
        incorrect = self.count(is_correct=False)
        
        # Get incorrect label distribution
        incorrect_labels = self.db.query(
            Feedback.correct_label,
            func.count(Feedback.id).label('count')
        ).filter(
            Feedback.is_correct == False,
            Feedback.correct_label.isnot(None)
        ).group_by(Feedback.correct_label).all()
        
        return {
            "total_feedback": total,
            "correct_predictions": correct,
            "incorrect_predictions": incorrect,
            "accuracy_percentage": (correct / total * 100) if total > 0 else None,
            "incorrect_label_distribution": {
                r.correct_label: r.count for r in incorrect_labels
            }
        }
    
    def get_recent(self, hours: int = 24) -> List[Feedback]:
        """Get feedback from the last N hours."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        return self.db.query(Feedback).filter(
            Feedback.created_at >= cutoff
        ).order_by(desc(Feedback.created_at)).all()
    
    def delete(self, feedback_id: int) -> bool:
        """Delete a feedback by ID."""
        feedback = self.get_by_id(feedback_id)
        if feedback:
            self.db.delete(feedback)
            self.db.commit()
            return True
        return False
    
    def has_feedback(self, prediction_id: str) -> bool:
        """Check if a prediction has any feedback."""
        return self.db.query(Feedback).filter(
            Feedback.prediction_id == prediction_id
        ).first() is not None
