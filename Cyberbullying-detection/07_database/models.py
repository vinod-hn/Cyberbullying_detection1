"""
SQLAlchemy ORM models for Cyberbullying Detection System.

Models:
- Message: Original text messages analyzed
- Prediction: Model predictions with metadata
- User: User profiles for tracking
- Alert: High-severity alerts requiring attention
- AuditLog: System activity logging
"""

import uuid
from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, Text, DateTime,
    ForeignKey, JSON, Enum as SQLEnum, Index
)
from sqlalchemy.orm import relationship
from enum import Enum

# Handle both relative and absolute imports
try:
    from .db_config import Base
except ImportError:
    from db_config import Base


def generate_uuid():
    """Generate a UUID string."""
    return str(uuid.uuid4())


class SeverityLevel(str, Enum):
    """Severity levels for alerts and predictions."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertStatus(str, Enum):
    """Status of alerts."""
    PENDING = "pending"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    DISMISSED = "dismissed"


class Message(Base):
    """
    Stores original messages that have been analyzed.
    """
    __tablename__ = "messages"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    message_id = Column(String(36), unique=True, default=generate_uuid, index=True)
    text = Column(Text, nullable=False)
    source = Column(String(50), default="api")  # api, dashboard, batch, etc.
    language = Column(String(20), default="en")  # en, kn, mixed
    extra_data = Column(JSON, nullable=True)  # Additional context (renamed from metadata)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    predictions = relationship("Prediction", back_populates="message", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index("idx_messages_created_at", "created_at"),
        Index("idx_messages_source", "source"),
    )
    
    def __repr__(self):
        return f"<Message(id={self.id}, text='{self.text[:30]}...')>"
    
    def to_dict(self):
        return {
            "id": self.id,
            "message_id": self.message_id,
            "text": self.text,
            "source": self.source,
            "language": self.language,
            "extra_data": self.extra_data,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class Prediction(Base):
    """
    Stores model predictions with confidence scores and metadata.
    """
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    prediction_id = Column(String(36), unique=True, default=generate_uuid, index=True)
    message_id = Column(Integer, ForeignKey("messages.id", ondelete="CASCADE"), nullable=False)
    
    # Prediction results
    model_type = Column(String(50), nullable=False)  # bert, mbert, indicbert, baseline
    predicted_label = Column(String(50), nullable=False)  # neutral, harassment, etc.
    confidence = Column(Float, nullable=False)
    is_cyberbullying = Column(Boolean, default=False)
    
    # Full probability distribution
    probabilities = Column(JSON, nullable=True)
    
    # Performance metrics
    inference_time_ms = Column(Float, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    message = relationship("Message", back_populates="predictions")
    alerts = relationship("Alert", back_populates="prediction", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index("idx_predictions_label", "predicted_label"),
        Index("idx_predictions_model", "model_type"),
        Index("idx_predictions_is_bullying", "is_cyberbullying"),
        Index("idx_predictions_created_at", "created_at"),
    )
    
    def __repr__(self):
        return f"<Prediction(id={self.id}, label='{self.predicted_label}', confidence={self.confidence:.2f})>"
    
    def to_dict(self):
        return {
            "id": self.id,
            "prediction_id": self.prediction_id,
            "message_id": self.message_id,
            "model_type": self.model_type,
            "predicted_label": self.predicted_label,
            "confidence": self.confidence,
            "is_cyberbullying": self.is_cyberbullying,
            "probabilities": self.probabilities,
            "inference_time_ms": self.inference_time_ms,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class User(Base):
    """
    User profiles for tracking behavior patterns.
    Used for dashboard authentication and risk profiling.
    """
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(36), unique=True, default=generate_uuid, index=True)
    username = Column(String(100), unique=True, nullable=False)
    email = Column(String(255), unique=True, nullable=True)
    password_hash = Column(String(255), nullable=True)  # For dashboard auth
    
    # Profile
    display_name = Column(String(100), nullable=True)
    role = Column(String(20), default="viewer")  # admin, moderator, viewer
    is_active = Column(Boolean, default=True)
    
    # Risk metrics (for monitored users, not dashboard users)
    risk_score = Column(Float, default=0.0)
    total_messages = Column(Integer, default=0)
    flagged_messages = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    last_active = Column(DateTime, nullable=True)
    
    # Relationships
    alerts = relationship("Alert", back_populates="user", cascade="all, delete-orphan")
    audit_logs = relationship("AuditLog", back_populates="user", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index("idx_users_username", "username"),
        Index("idx_users_risk_score", "risk_score"),
    )
    
    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}')>"
    
    def to_dict(self, include_sensitive=False):
        result = {
            "id": self.id,
            "user_id": self.user_id,
            "username": self.username,
            "display_name": self.display_name,
            "role": self.role,
            "is_active": self.is_active,
            "risk_score": self.risk_score,
            "total_messages": self.total_messages,
            "flagged_messages": self.flagged_messages,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_active": self.last_active.isoformat() if self.last_active else None,
        }
        if include_sensitive:
            result["email"] = self.email
        return result


class Alert(Base):
    """
    Alerts for high-severity predictions requiring attention.
    """
    __tablename__ = "alerts"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    alert_id = Column(String(36), unique=True, default=generate_uuid, index=True)
    
    # References
    prediction_id = Column(Integer, ForeignKey("predictions.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    
    # Alert details
    severity = Column(String(20), default=SeverityLevel.MEDIUM.value)
    status = Column(String(20), default=AlertStatus.PENDING.value)
    reason = Column(Text, nullable=True)  # Why this alert was generated
    
    # Resolution
    resolved_by = Column(String(100), nullable=True)
    resolved_at = Column(DateTime, nullable=True)
    resolution_notes = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    acknowledged_at = Column(DateTime, nullable=True)
    
    # Relationships
    prediction = relationship("Prediction", back_populates="alerts")
    user = relationship("User", back_populates="alerts")
    
    # Indexes
    __table_args__ = (
        Index("idx_alerts_status", "status"),
        Index("idx_alerts_severity", "severity"),
        Index("idx_alerts_created_at", "created_at"),
    )
    
    def __repr__(self):
        return f"<Alert(id={self.id}, severity='{self.severity}', status='{self.status}')>"
    
    def to_dict(self):
        return {
            "id": self.id,
            "alert_id": self.alert_id,
            "prediction_id": self.prediction_id,
            "user_id": self.user_id,
            "severity": self.severity,
            "status": self.status,
            "reason": self.reason,
            "resolved_by": self.resolved_by,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "resolution_notes": self.resolution_notes,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
        }


class AuditLog(Base):
    """
    Audit log for tracking system activities.
    """
    __tablename__ = "audit_logs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    log_id = Column(String(36), unique=True, default=generate_uuid, index=True)
    
    # Actor
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    ip_address = Column(String(45), nullable=True)  # IPv6 compatible
    user_agent = Column(String(255), nullable=True)
    
    # Action details
    action = Column(String(100), nullable=False)  # predict, login, view_alert, etc.
    resource_type = Column(String(50), nullable=True)  # message, prediction, alert, etc.
    resource_id = Column(String(36), nullable=True)
    
    # Request/Response
    request_data = Column(JSON, nullable=True)
    response_status = Column(Integer, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="audit_logs")
    
    # Indexes
    __table_args__ = (
        Index("idx_audit_action", "action"),
        Index("idx_audit_created_at", "created_at"),
        Index("idx_audit_resource", "resource_type", "resource_id"),
    )
    
    def __repr__(self):
        return f"<AuditLog(id={self.id}, action='{self.action}')>"
    
    def to_dict(self):
        return {
            "id": self.id,
            "log_id": self.log_id,
            "user_id": self.user_id,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "action": self.action,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "request_data": self.request_data,
            "response_status": self.response_status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class Feedback(Base):
    """
    User feedback on predictions for model improvement.
    """
    __tablename__ = "feedback"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    feedback_id = Column(String(36), unique=True, default=generate_uuid, index=True)
    
    # Reference to prediction
    prediction_id = Column(String(36), nullable=False, index=True)
    
    # Feedback details
    is_correct = Column(Boolean, nullable=False)
    correct_label = Column(String(50), nullable=True)  # If incorrect, what's the right label?
    comments = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Indexes
    __table_args__ = (
        Index("idx_feedback_prediction", "prediction_id"),
        Index("idx_feedback_is_correct", "is_correct"),
    )
    
    def __repr__(self):
        return f"<Feedback(id={self.id}, is_correct={self.is_correct})>"
    
    def to_dict(self):
        return {
            "id": self.id,
            "feedback_id": self.feedback_id,
            "prediction_id": self.prediction_id,
            "is_correct": self.is_correct,
            "correct_label": self.correct_label,
            "comments": self.comments,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
