"""
Response schemas for Cyberbullying Detection API.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    """Response model for prediction."""
    text: str
    prediction: str
    confidence: float
    is_cyberbullying: bool
    probabilities: Optional[Dict[str, float]] = None
    prediction_id: Optional[str] = None


class BatchPredictionResponse(BaseModel):
    """Response model for batch prediction."""
    predictions: List[PredictionResponse]
    total: int
    processing_time_ms: float


class ConversationMessageResult(BaseModel):
    """Prediction result for a single message in conversation."""
    text: str
    prediction: str
    confidence: float
    is_cyberbullying: bool
    context_score: Optional[float] = None


class ConversationPredictionResponse(BaseModel):
    """Response model for conversation-level prediction."""
    messages: List[ConversationMessageResult]
    overall_risk_score: float = Field(..., description="Overall conversation risk score (0-1)")
    escalation_detected: bool = Field(..., description="Whether escalation pattern detected")
    flagged_messages: List[int] = Field(..., description="Indices of flagged messages")
    processing_time_ms: float


class ModelInfo(BaseModel):
    """Model information response."""
    model_type: str
    accuracy: float
    f1_score: float
    status: str
    description: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    models_available: List[str]
    default_model: str
    version: str = "1.0.0"


class StatisticsResponse(BaseModel):
    """Statistics response for dashboard graphs."""
    total_predictions: int
    cyberbullying_count: int
    not_cyberbullying_count: int
    severity_distribution: Dict[str, int]
    language_distribution: Dict[str, int]
    daily_counts: List[Dict[str, Any]]
    model_usage: Dict[str, int]


class FeedbackResponse(BaseModel):
    """Response model for feedback submission."""
    success: bool
    message: str
    feedback_id: Optional[str] = None


class TokenResponse(BaseModel):
    """Response model for JWT token."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int = Field(..., description="Token expiry time in seconds")
