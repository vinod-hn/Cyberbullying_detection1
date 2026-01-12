"""
Prediction Repository - CRUD operations for predictions.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import desc, func

# Handle both relative and absolute imports
try:
    from ..models import Prediction, Message
except ImportError:
    from models import Prediction, Message


class PredictionRepository:
    """Repository for Prediction CRUD operations."""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create(
        self,
        message_id: int,
        model_type: str,
        predicted_label: str,
        confidence: float,
        is_cyberbullying: bool,
        probabilities: dict = None,
        inference_time_ms: float = None
    ) -> Prediction:
        """Create a new prediction."""
        prediction = Prediction(
            message_id=message_id,
            model_type=model_type,
            predicted_label=predicted_label,
            confidence=confidence,
            is_cyberbullying=is_cyberbullying,
            probabilities=probabilities,
            inference_time_ms=inference_time_ms
        )
        self.db.add(prediction)
        self.db.commit()
        self.db.refresh(prediction)
        return prediction
    
    def create_with_message(
        self,
        text: str,
        model_type: str,
        predicted_label: str,
        confidence: float,
        is_cyberbullying: bool,
        probabilities: dict = None,
        inference_time_ms: float = None,
        source: str = "api"
    ) -> tuple:
        """Create a message and prediction together. Returns (message, prediction)."""
        # Create message first
        message = Message(text=text, source=source)
        self.db.add(message)
        self.db.flush()  # Get the ID without committing
        
        # Create prediction
        prediction = Prediction(
            message_id=message.id,
            model_type=model_type,
            predicted_label=predicted_label,
            confidence=confidence,
            is_cyberbullying=is_cyberbullying,
            probabilities=probabilities,
            inference_time_ms=inference_time_ms
        )
        self.db.add(prediction)
        self.db.commit()
        self.db.refresh(message)
        self.db.refresh(prediction)
        
        return message, prediction
    
    def get_by_id(self, prediction_id: int) -> Optional[Prediction]:
        """Get prediction by ID."""
        return self.db.query(Prediction).filter(Prediction.id == prediction_id).first()
    
    def get_by_uuid(self, prediction_uuid: str) -> Optional[Prediction]:
        """Get prediction by UUID."""
        return self.db.query(Prediction).filter(Prediction.prediction_id == prediction_uuid).first()
    
    def get_by_message_id(self, message_id: int) -> List[Prediction]:
        """Get all predictions for a message."""
        return self.db.query(Prediction).filter(
            Prediction.message_id == message_id
        ).order_by(desc(Prediction.created_at)).all()
    
    def get_all(
        self,
        skip: int = 0,
        limit: int = 100,
        model_type: Optional[str] = None,
        is_cyberbullying: Optional[bool] = None
    ) -> List[Prediction]:
        """Get all predictions with filters and pagination."""
        query = self.db.query(Prediction)
        
        if model_type:
            query = query.filter(Prediction.model_type == model_type)
        
        if is_cyberbullying is not None:
            query = query.filter(Prediction.is_cyberbullying == is_cyberbullying)
        
        return query.order_by(desc(Prediction.created_at)).offset(skip).limit(limit).all()
    
    def get_recent(self, hours: int = 24) -> List[Prediction]:
        """Get predictions from the last N hours."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        return self.db.query(Prediction).filter(
            Prediction.created_at >= cutoff
        ).order_by(desc(Prediction.created_at)).all()
    
    def get_cyberbullying_predictions(
        self,
        skip: int = 0,
        limit: int = 100,
        min_confidence: float = 0.0
    ) -> List[Prediction]:
        """Get only cyberbullying predictions."""
        return self.db.query(Prediction).filter(
            Prediction.is_cyberbullying == True,
            Prediction.confidence >= min_confidence
        ).order_by(desc(Prediction.created_at)).offset(skip).limit(limit).all()
    
    def count(
        self,
        model_type: Optional[str] = None,
        is_cyberbullying: Optional[bool] = None
    ) -> int:
        """Count predictions with optional filters."""
        query = self.db.query(func.count(Prediction.id))
        
        if model_type:
            query = query.filter(Prediction.model_type == model_type)
        
        if is_cyberbullying is not None:
            query = query.filter(Prediction.is_cyberbullying == is_cyberbullying)
        
        return query.scalar()
    
    def get_label_distribution(self) -> Dict[str, int]:
        """Get count of predictions per label."""
        results = self.db.query(
            Prediction.predicted_label,
            func.count(Prediction.id).label('count')
        ).group_by(Prediction.predicted_label).all()
        
        return {r.predicted_label: r.count for r in results}
    
    def get_model_distribution(self) -> Dict[str, int]:
        """Get count of predictions per model."""
        results = self.db.query(
            Prediction.model_type,
            func.count(Prediction.id).label('count')
        ).group_by(Prediction.model_type).all()
        
        return {r.model_type: r.count for r in results}
    
    def get_stats(self, days: int = 7) -> Dict[str, Any]:
        """Get comprehensive prediction statistics."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        # Basic counts
        total = self.count()
        recent = self.db.query(func.count(Prediction.id)).filter(
            Prediction.created_at >= cutoff
        ).scalar()
        
        bullying_count = self.count(is_cyberbullying=True)
        
        # Average confidence
        avg_confidence = self.db.query(func.avg(Prediction.confidence)).scalar() or 0.0
        
        # By day
        daily = self.db.query(
            func.date(Prediction.created_at).label('date'),
            func.count(Prediction.id).label('total'),
            func.sum(func.case(
                (Prediction.is_cyberbullying == True, 1),
                else_=0
            )).label('bullying')
        ).filter(
            Prediction.created_at >= cutoff
        ).group_by(
            func.date(Prediction.created_at)
        ).all()
        
        return {
            "total_predictions": total,
            "predictions_last_n_days": recent,
            "bullying_predictions": bullying_count,
            "neutral_predictions": total - bullying_count,
            "bullying_percentage": (bullying_count / total * 100) if total > 0 else 0,
            "average_confidence": round(avg_confidence, 4),
            "label_distribution": self.get_label_distribution(),
            "model_distribution": self.get_model_distribution(),
            "daily_stats": [
                {
                    "date": str(d.date),
                    "total": d.total,
                    "bullying": d.bullying or 0
                }
                for d in daily
            ]
        }
    
    def get_average_inference_time(self, model_type: Optional[str] = None) -> float:
        """Get average inference time in milliseconds."""
        query = self.db.query(func.avg(Prediction.inference_time_ms))
        
        if model_type:
            query = query.filter(Prediction.model_type == model_type)
        
        return query.scalar() or 0.0
    
    def delete(self, prediction_id: int) -> bool:
        """Delete a prediction by ID."""
        prediction = self.get_by_id(prediction_id)
        if prediction:
            self.db.delete(prediction)
            self.db.commit()
            return True
        return False
