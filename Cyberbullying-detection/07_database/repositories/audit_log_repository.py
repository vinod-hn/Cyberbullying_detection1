"""
Audit Log Repository - CRUD operations for audit logs.
"""

from typing import List, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import desc, func

# Handle both relative and absolute imports
try:
    from ..models import AuditLog
except ImportError:
    from models import AuditLog


class AuditLogRepository:
    """Repository for AuditLog CRUD operations."""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create(
        self,
        action: str,
        user_id: Optional[int] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        request_data: Optional[dict] = None,
        response_status: Optional[int] = None
    ) -> AuditLog:
        """Create a new audit log entry."""
        log = AuditLog(
            action=action,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            resource_type=resource_type,
            resource_id=resource_id,
            request_data=request_data,
            response_status=response_status
        )
        self.db.add(log)
        self.db.commit()
        self.db.refresh(log)
        return log
    
    def log_prediction(
        self,
        prediction_id: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        model_type: str = "bert"
    ) -> AuditLog:
        """Convenience method for logging predictions."""
        return self.create(
            action="predict",
            ip_address=ip_address,
            user_agent=user_agent,
            resource_type="prediction",
            resource_id=prediction_id,
            request_data={"model_type": model_type},
            response_status=200
        )
    
    def log_auth(
        self,
        action: str,
        username: str,
        ip_address: Optional[str] = None,
        success: bool = True
    ) -> AuditLog:
        """Convenience method for logging auth events."""
        return self.create(
            action=action,
            ip_address=ip_address,
            resource_type="auth",
            resource_id=username,
            request_data={"username": username},
            response_status=200 if success else 401
        )
    
    def get_by_id(self, log_id: int) -> Optional[AuditLog]:
        """Get audit log by ID."""
        return self.db.query(AuditLog).filter(AuditLog.id == log_id).first()
    
    def get_all(
        self,
        skip: int = 0,
        limit: int = 100,
        action: Optional[str] = None,
        resource_type: Optional[str] = None
    ) -> List[AuditLog]:
        """Get all audit logs with filters and pagination."""
        query = self.db.query(AuditLog)
        
        if action:
            query = query.filter(AuditLog.action == action)
        
        if resource_type:
            query = query.filter(AuditLog.resource_type == resource_type)
        
        return query.order_by(desc(AuditLog.created_at)).offset(skip).limit(limit).all()
    
    def get_by_user(self, user_id: int, limit: int = 100) -> List[AuditLog]:
        """Get audit logs for a specific user."""
        return self.db.query(AuditLog).filter(
            AuditLog.user_id == user_id
        ).order_by(desc(AuditLog.created_at)).limit(limit).all()
    
    def get_by_resource(
        self,
        resource_type: str,
        resource_id: str
    ) -> List[AuditLog]:
        """Get audit logs for a specific resource."""
        return self.db.query(AuditLog).filter(
            AuditLog.resource_type == resource_type,
            AuditLog.resource_id == resource_id
        ).order_by(desc(AuditLog.created_at)).all()
    
    def get_recent(self, hours: int = 24, action: Optional[str] = None) -> List[AuditLog]:
        """Get audit logs from the last N hours."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        query = self.db.query(AuditLog).filter(AuditLog.created_at >= cutoff)
        
        if action:
            query = query.filter(AuditLog.action == action)
        
        return query.order_by(desc(AuditLog.created_at)).all()
    
    def count(self, action: Optional[str] = None) -> int:
        """Count audit logs with optional filter."""
        query = self.db.query(func.count(AuditLog.id))
        
        if action:
            query = query.filter(AuditLog.action == action)
        
        return query.scalar()
    
    def get_action_distribution(self, days: int = 7) -> dict:
        """Get count of each action type."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        results = self.db.query(
            AuditLog.action,
            func.count(AuditLog.id).label('count')
        ).filter(
            AuditLog.created_at >= cutoff
        ).group_by(AuditLog.action).all()
        
        return {r.action: r.count for r in results}
    
    def get_stats(self, days: int = 7) -> dict:
        """Get audit log statistics."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        total = self.count()
        recent = self.db.query(func.count(AuditLog.id)).filter(
            AuditLog.created_at >= cutoff
        ).scalar()
        
        # Unique IPs
        unique_ips = self.db.query(func.count(func.distinct(AuditLog.ip_address))).filter(
            AuditLog.created_at >= cutoff
        ).scalar()
        
        # Error rate
        total_requests = self.db.query(func.count(AuditLog.id)).filter(
            AuditLog.created_at >= cutoff,
            AuditLog.response_status.isnot(None)
        ).scalar()
        
        errors = self.db.query(func.count(AuditLog.id)).filter(
            AuditLog.created_at >= cutoff,
            AuditLog.response_status >= 400
        ).scalar()
        
        return {
            "total_logs": total,
            "logs_last_n_days": recent,
            "unique_ips": unique_ips,
            "action_distribution": self.get_action_distribution(days),
            "error_rate": (errors / total_requests * 100) if total_requests > 0 else 0,
        }
    
    def delete_old(self, days: int = 90) -> int:
        """Delete audit logs older than N days."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        count = self.db.query(AuditLog).filter(AuditLog.created_at < cutoff).delete()
        self.db.commit()
        return count
