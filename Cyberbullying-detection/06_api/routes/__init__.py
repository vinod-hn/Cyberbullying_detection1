"""
API Routes package.
"""

from fastapi import APIRouter

try:
    from .predict import router as predict_router
    from .batch_predict import router as batch_predict_router
    from .conversation_predict import router as conversation_router
    from .statistics import router as statistics_router
    from .health import router as health_router
    from .feedback import router as feedback_router
    from .auth import router as auth_router
    from .csv_upload import router as csv_upload_router
    from .predictions import router as predictions_router
    from .export import router as export_router
    from .history import router as history_router
except ImportError:
    from predict import router as predict_router
    from batch_predict import router as batch_predict_router
    from conversation_predict import router as conversation_router
    from statistics import router as statistics_router
    from health import router as health_router
    from feedback import router as feedback_router
    from auth import router as auth_router
    from csv_upload import router as csv_upload_router
    from predictions import router as predictions_router
    from export import router as export_router
    from history import router as history_router

# Main API router that includes all sub-routers
api_router = APIRouter()

api_router.include_router(health_router, tags=["Health"])
api_router.include_router(auth_router, prefix="/auth", tags=["Authentication"])
api_router.include_router(predict_router, tags=["Prediction"])
api_router.include_router(batch_predict_router, tags=["Prediction"])
api_router.include_router(conversation_router, tags=["Prediction"])
api_router.include_router(predictions_router, tags=["Predictions"])
api_router.include_router(statistics_router, tags=["Statistics"])
api_router.include_router(feedback_router, tags=["Feedback"])
api_router.include_router(csv_upload_router, tags=["CSV Upload"])
api_router.include_router(export_router, tags=["Export"])
api_router.include_router(history_router, tags=["History"])

__all__ = ["api_router"]
