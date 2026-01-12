"""
Pydantic schemas for API request/response validation.
"""

try:
    from .request_schemas import (
        PredictionRequest,
        BatchPredictionRequest,
        ConversationMessage,
        ConversationPredictionRequest,
        FeedbackRequest,
        LoginRequest,
    )

    from .response_schemas import (
        PredictionResponse,
        BatchPredictionResponse,
        ConversationMessageResult,
        ConversationPredictionResponse,
        HealthResponse,
        ModelInfo,
        StatisticsResponse,
        FeedbackResponse,
        TokenResponse,
    )
except ImportError:
    from request_schemas import (
        PredictionRequest,
        BatchPredictionRequest,
        ConversationMessage,
        ConversationPredictionRequest,
        FeedbackRequest,
        LoginRequest,
    )

    from response_schemas import (
        PredictionResponse,
        BatchPredictionResponse,
        ConversationMessageResult,
        ConversationPredictionResponse,
        HealthResponse,
        ModelInfo,
        StatisticsResponse,
        FeedbackResponse,
        TokenResponse,
    )

__all__ = [
    # Requests
    "PredictionRequest",
    "BatchPredictionRequest",
    "ConversationMessage",
    "ConversationPredictionRequest",
    "FeedbackRequest",
    "LoginRequest",
    # Responses
    "PredictionResponse",
    "BatchPredictionResponse",
    "ConversationMessageResult",
    "ConversationPredictionResponse",
    "HealthResponse",
    "ModelInfo",
    "StatisticsResponse",
    "FeedbackResponse",
    "TokenResponse",
]
