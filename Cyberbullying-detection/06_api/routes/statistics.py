"""
Statistics endpoint for dashboard graphs.
GET /stats
"""

from fastapi import APIRouter, HTTPException, Query
from datetime import datetime, timedelta
from typing import Optional

from ..schemas import StatisticsResponse

router = APIRouter()

# In-memory stats store (replace with database in production)
_stats_store = {
    "total_predictions": 0,
    "cyberbullying_count": 0,
    "not_cyberbullying_count": 0,
    "severity_distribution": {"low": 0, "medium": 0, "high": 0, "critical": 0},
    "language_distribution": {"english": 0, "kannada": 0, "code_mixed": 0},
    "daily_counts": [],
    "model_usage": {"bert": 0, "mbert": 0, "indicbert": 0, "baseline": 0}
}


def update_stats(prediction_result: dict, model_type: str = "bert"):
    """Update statistics after a prediction (call from predict routes)."""
    _stats_store["total_predictions"] += 1
    
    if prediction_result.get("is_cyberbullying"):
        _stats_store["cyberbullying_count"] += 1
    else:
        _stats_store["not_cyberbullying_count"] += 1
    
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
        # Generate sample daily counts if empty
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
        
        return StatisticsResponse(
            total_predictions=_stats_store["total_predictions"],
            cyberbullying_count=_stats_store["cyberbullying_count"],
            not_cyberbullying_count=_stats_store["not_cyberbullying_count"],
            severity_distribution=_stats_store["severity_distribution"],
            language_distribution=_stats_store["language_distribution"],
            daily_counts=_stats_store["daily_counts"][-days:],
            model_usage=_stats_store["model_usage"]
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
        "daily_counts": [],
        "model_usage": {"bert": 0, "mbert": 0, "indicbert": 0, "baseline": 0}
    }
    return {"message": "Statistics reset successfully"}
