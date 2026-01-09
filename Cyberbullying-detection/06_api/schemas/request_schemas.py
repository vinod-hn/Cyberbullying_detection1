"""
Request schemas for Cyberbullying Detection API.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Request model for single prediction."""
    text: str = Field(..., description="Text to analyze for cyberbullying", min_length=1)
    model_type: Optional[str] = Field(
        "bert",
        description="Model to use: bert, mbert, indicbert, baseline"
    )


class BatchPredictionRequest(BaseModel):
    """Request model for batch prediction."""
    texts: List[str] = Field(..., description="List of texts to analyze", min_length=1)
    model_type: Optional[str] = Field("bert", description="Model to use")
    batch_size: Optional[int] = Field(32, description="Processing batch size", ge=1, le=128)


class ConversationMessage(BaseModel):
    """Single message in a conversation."""
    text: str = Field(..., description="Message text")
    user_id: Optional[str] = Field(None, description="User identifier")
    timestamp: Optional[str] = Field(None, description="Message timestamp (ISO format)")


class ConversationPredictionRequest(BaseModel):
    """Request model for conversation-level prediction."""
    messages: List[ConversationMessage] = Field(
        ..., 
        description="List of conversation messages in order",
        min_length=1
    )
    model_type: Optional[str] = Field("bert", description="Model to use")
    include_context: Optional[bool] = Field(
        True, 
        description="Whether to use conversation context for prediction"
    )


class FeedbackRequest(BaseModel):
    """Request model for user feedback on predictions."""
    prediction_id: str = Field(..., description="ID of the prediction to provide feedback for")
    correct_label: str = Field(..., description="The correct label (cyberbullying/not_cyberbullying)")
    comments: Optional[str] = Field(None, description="Additional comments from the user")


class LoginRequest(BaseModel):
    """Request model for local JWT authentication."""
    username: str = Field(..., description="Username", min_length=1)
    password: str = Field(..., description="Password", min_length=1)
