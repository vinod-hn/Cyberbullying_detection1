"""
Predictions listing endpoint for dashboard.
GET /predictions
"""

from fastapi import APIRouter, Query
from typing import Optional, List
from datetime import datetime, timedelta
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/predictions")
async def get_predictions(
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(10, ge=1, le=100, description="Items per page"),
    severity: Optional[str] = Query(None, description="Filter by severity"),
    platform: Optional[str] = Query(None, description="Filter by platform"),
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)")
):
    """
    Get paginated list of predictions for dashboard table.
    
    Returns predictions with filtering and pagination support.
    """
    try:
        # Try to get from database
        from db_helper import get_db_context, get_prediction_repository
        
        ctx = get_db_context()
        if ctx:
            with ctx as db:
                pred_repo = get_prediction_repository(db)
                if pred_repo:
                    # Get predictions from database
                    predictions = pred_repo.get_recent(limit=per_page * page)
                    
                    if predictions:
                        # Format for dashboard
                        data = []
                        for p in predictions:
                            severity_level = "low"
                            if p.confidence >= 0.9 and p.is_cyberbullying:
                                severity_level = "critical"
                            elif p.confidence >= 0.75 and p.is_cyberbullying:
                                severity_level = "high"
                            elif p.confidence >= 0.5 and p.is_cyberbullying:
                                severity_level = "medium"
                            
                            data.append({
                                "id": p.id,
                                "date": p.created_at.strftime("%Y-%m-%d") if p.created_at else datetime.now().strftime("%Y-%m-%d"),
                                "student_id": f"User_{p.id}",
                                "message": p.message.text[:100] if p.message else "N/A",
                                "platform": "api",
                                "severity": severity_level,
                                "score": p.predicted_label,
                                "confidence": f"{p.confidence * 100:.0f}%",
                                "is_cyberbullying": p.is_cyberbullying,
                                "model": p.model_type
                            })
                        
                        # Apply pagination
                        start_idx = (page - 1) * per_page
                        end_idx = start_idx + per_page
                        paginated_data = data[start_idx:end_idx]
                        
                        return {
                            "data": paginated_data,
                            "total": len(data),
                            "page": page,
                            "per_page": per_page
                        }
        
        # Return empty if no database
        return {
            "data": [],
            "total": 0,
            "page": page,
            "per_page": per_page
        }
        
    except Exception as e:
        logger.warning(f"Failed to get predictions: {e}")
        return {
            "data": [],
            "total": 0,
            "page": page,
            "per_page": per_page
        }
