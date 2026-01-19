"""
Statistics endpoint for dashboard graphs.
GET /stats
"""

from fastapi import APIRouter, HTTPException, Query
from datetime import datetime, timedelta
from typing import Optional
import logging

try:
    from ..schemas import StatisticsResponse
except ImportError:
    from schemas import StatisticsResponse

router = APIRouter()
logger = logging.getLogger(__name__)

# In-memory stats store (fallback if database unavailable)
_stats_store = {
    "total_predictions": 0,
    "cyberbullying_count": 0,
    "not_cyberbullying_count": 0,
    "severity_distribution": {"low": 0, "medium": 0, "high": 0, "critical": 0},
    "language_distribution": {"english": 0, "kannada": 0, "code_mixed": 0},
    # Label distribution for donut chart when DB is not available
    "label_distribution": {},
    # Cached daily counts for fallback mode
    "daily_counts": [],
    # Optional cached monthly trend when DB is not available
    "monthly_trend": {},
    "model_usage": {"bert": 0, "mbert": 0, "indicbert": 0, "baseline": 0}
}


def get_stats_from_db(days: int = 7) -> Optional[dict]:
    """Get statistics from database."""
    try:
        from ..db_helper import get_db_context, get_prediction_repository, get_feedback_repository
        
        ctx = get_db_context()
        if ctx is None:
            return None
            
        with ctx as db:
            pred_repo = get_prediction_repository(db)
            if not pred_repo:
                return None
                
            stats = pred_repo.get_stats(days=days)
            
            # Get feedback stats
            feedback_repo = get_feedback_repository(db)
            accuracy = {}
            if feedback_repo:
                accuracy = feedback_repo.get_accuracy_stats()
            
            return {
                "total_predictions": stats["total_predictions"],
                "cyberbullying_count": stats["bullying_predictions"],
                "not_cyberbullying_count": stats["neutral_predictions"],
                "label_distribution": stats["label_distribution"],
                "model_usage": stats["model_distribution"],
                "daily_stats": stats["daily_stats"],
                "average_confidence": stats["average_confidence"],
                "feedback_accuracy": accuracy.get("accuracy_percentage")
            }
    except Exception as e:
        logger.warning(f"Failed to get stats from database: {e}")
        return None


def update_stats(prediction_result: dict, model_type: str = "bert"):
    """Update statistics after a prediction (call from predict routes)."""
    _stats_store["total_predictions"] += 1
    
    if prediction_result.get("is_cyberbullying"):
        _stats_store["cyberbullying_count"] += 1
    else:
        _stats_store["not_cyberbullying_count"] += 1

    # Update label distribution if prediction label is available
    label = prediction_result.get("prediction")
    if label:
        if "label_distribution" not in _stats_store or _stats_store["label_distribution"] is None:
            _stats_store["label_distribution"] = {}
        _stats_store["label_distribution"][label] = _stats_store["label_distribution"].get(label, 0) + 1
    
    # Update model usage
    if model_type in _stats_store["model_usage"]:
        _stats_store["model_usage"][model_type] += 1


@router.get("/stats", response_model=StatisticsResponse)
async def get_statistics(
    days: Optional[int] = Query(7, description="Number of days for daily counts", ge=1, le=90)
):
    """
    Get prediction statistics for dashboard visualization.
    
    - **days**: Number of days to include in daily counts (1-90)
    
    Returns aggregated statistics for graphs:
    - Bullying type distribution (pie chart)
    - Severity level counts (bar chart)
    - Daily prediction trends (line chart)
    - Language distribution (donut chart)
    - Model usage breakdown
    """
    try:
        # Try database first
        db_stats = get_stats_from_db(days=days)
        
        if db_stats:
            # Format daily counts for response
            daily_counts = [
                {
                    "date": d["date"],
                    "predictions": d["total"],
                    "cyberbullying": d["bullying"]
                }
                for d in db_stats.get("daily_stats", [])
            ]

            # If no daily data, generate empty structure
            if not daily_counts:
                today = datetime.now()
                daily_counts = [
                    {
                        "date": (today - timedelta(days=i)).strftime("%Y-%m-%d"),
                        "predictions": 0,
                        "cyberbullying": 0
                    }
                    for i in range(days - 1, -1, -1)
                ]

            # Build monthly trend from daily stats
            monthly_trend: dict[str, int] = {}
            for d in daily_counts:
                try:
                    dt = datetime.strptime(d["date"], "%Y-%m-%d")
                    month_key = dt.strftime("%b")  # e.g., Jan, Feb
                except Exception:
                    # If date format is unexpected, skip aggregation for safety
                    continue

                monthly_trend[month_key] = monthly_trend.get(month_key, 0) + int(d.get("predictions", 0))

            return StatisticsResponse(
                total_predictions=db_stats["total_predictions"],
                cyberbullying_count=db_stats["cyberbullying_count"],
                not_cyberbullying_count=db_stats["not_cyberbullying_count"],
                severity_distribution=_stats_store["severity_distribution"],  # Not tracked in DB yet
                language_distribution=_stats_store["language_distribution"],  # Not tracked in DB yet
                daily_counts=daily_counts,
                model_usage=db_stats.get("model_usage", _stats_store["model_usage"]),
                label_distribution=db_stats.get("label_distribution", {}),
                monthly_trend=monthly_trend,
            )
        
        # Fallback to in-memory stats
        if not _stats_store["daily_counts"]:
            today = datetime.now()
            _stats_store["daily_counts"] = [
                {
                    "date": (today - timedelta(days=i)).strftime("%Y-%m-%d"),
                    "predictions": 0,
                    "cyberbullying": 0
                }
                for i in range(days - 1, -1, -1)
            ]

        # Build monthly trend from fallback daily counts
        fallback_daily = _stats_store["daily_counts"][ -days: ]
        monthly_trend: dict[str, int] = {}
        for d in fallback_daily:
            try:
                dt = datetime.strptime(d["date"], "%Y-%m-%d")
                month_key = dt.strftime("%b")
            except Exception:
                continue

            monthly_trend[month_key] = monthly_trend.get(month_key, 0) + int(d.get("predictions", 0))

        return StatisticsResponse(
            total_predictions=_stats_store["total_predictions"],
            cyberbullying_count=_stats_store["cyberbullying_count"],
            not_cyberbullying_count=_stats_store["not_cyberbullying_count"],
            severity_distribution=_stats_store["severity_distribution"],
            language_distribution=_stats_store["language_distribution"],
            daily_counts=fallback_daily,
            model_usage=_stats_store["model_usage"],
            label_distribution=_stats_store.get("label_distribution", {}),
            monthly_trend=monthly_trend,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve statistics: {str(e)}")


@router.post("/stats/reset", include_in_schema=False)
async def reset_statistics():
    """Reset all statistics (admin only, hidden from docs)."""
    global _stats_store
    _stats_store = {
        "total_predictions": 0,
        "cyberbullying_count": 0,
        "not_cyberbullying_count": 0,
        "severity_distribution": {"low": 0, "medium": 0, "high": 0, "critical": 0},
        "language_distribution": {"english": 0, "kannada": 0, "code_mixed": 0},
        "label_distribution": {},
        "daily_counts": [],
        "monthly_trend": {},
        "model_usage": {"bert": 0, "mbert": 0, "indicbert": 0, "baseline": 0}
    }
    return {"message": "Statistics reset successfully"}
