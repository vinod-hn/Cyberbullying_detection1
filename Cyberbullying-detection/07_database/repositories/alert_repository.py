"""
Alert Repository - CRUD operations for alerts.
"""

from typing import List, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import desc, func

# Handle both relative and absolute imports
try:
    from ..models import Alert, AlertStatus, SeverityLevel
except ImportError:
    from models import Alert, AlertStatus, SeverityLevel


class AlertRepository:
    """Repository for Alert CRUD operations."""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create(
        self,
        prediction_id: int,
        severity: str = SeverityLevel.MEDIUM.value,
        reason: Optional[str] = None,
        user_id: Optional[int] = None
    ) -> Alert:
        """Create a new alert."""
        alert = Alert(
            prediction_id=prediction_id,
            severity=severity,
            reason=reason,
            user_id=user_id,
            status=AlertStatus.PENDING.value
        )
        self.db.add(alert)
        self.db.commit()
        self.db.refresh(alert)
        return alert
    
    def get_by_id(self, alert_id: int) -> Optional[Alert]:
        """Get alert by ID."""
        return self.db.query(Alert).filter(Alert.id == alert_id).first()
    
    def get_by_uuid(self, alert_uuid: str) -> Optional[Alert]:
        """Get alert by UUID."""
        return self.db.query(Alert).filter(Alert.alert_id == alert_uuid).first()
    
    def get_all(
        self,
        skip: int = 0,
        limit: int = 100,
        status: Optional[str] = None,
        severity: Optional[str] = None
    ) -> List[Alert]:
        """Get all alerts with filters and pagination."""
        query = self.db.query(Alert)
        
        if status:
            query = query.filter(Alert.status == status)
        
        if severity:
            query = query.filter(Alert.severity == severity)
        
        return query.order_by(desc(Alert.created_at)).offset(skip).limit(limit).all()
    
    def get_pending(self, limit: int = 50) -> List[Alert]:
        """Get pending alerts, most severe first."""
        severity_order = {
            SeverityLevel.CRITICAL.value: 0,
            SeverityLevel.HIGH.value: 1,
            SeverityLevel.MEDIUM.value: 2,
            SeverityLevel.LOW.value: 3,
        }
        
        alerts = self.db.query(Alert).filter(
            Alert.status == AlertStatus.PENDING.value
        ).order_by(desc(Alert.created_at)).limit(limit).all()
        
        # Sort by severity
        return sorted(alerts, key=lambda a: severity_order.get(a.severity, 4))
    
    def get_by_prediction(self, prediction_id: int) -> List[Alert]:
        """Get all alerts for a prediction."""
        return self.db.query(Alert).filter(
            Alert.prediction_id == prediction_id
        ).order_by(desc(Alert.created_at)).all()
    
    def get_by_user(self, user_id: int) -> List[Alert]:
        """Get all alerts for a user."""
        return self.db.query(Alert).filter(
            Alert.user_id == user_id
        ).order_by(desc(Alert.created_at)).all()
    
    def acknowledge(self, alert_id: int) -> bool:
        """Mark an alert as acknowledged."""
        alert = self.get_by_id(alert_id)
        if alert:
            alert.status = AlertStatus.ACKNOWLEDGED.value
            alert.acknowledged_at = datetime.utcnow()
            self.db.commit()
            return True
        return False
    
    def resolve(
        self,
        alert_id: int,
        resolved_by: str,
        resolution_notes: Optional[str] = None
    ) -> bool:
        """Mark an alert as resolved."""
        alert = self.get_by_id(alert_id)
        if alert:
            alert.status = AlertStatus.RESOLVED.value
            alert.resolved_by = resolved_by
            alert.resolved_at = datetime.utcnow()
            alert.resolution_notes = resolution_notes
            self.db.commit()
            return True
        return False
    
    def dismiss(self, alert_id: int, resolved_by: str) -> bool:
        """Dismiss an alert."""
        alert = self.get_by_id(alert_id)
        if alert:
            alert.status = AlertStatus.DISMISSED.value
            alert.resolved_by = resolved_by
            alert.resolved_at = datetime.utcnow()
            self.db.commit()
            return True
        return False
    
    def count(
        self,
        status: Optional[str] = None,
        severity: Optional[str] = None
    ) -> int:
        """Count alerts with optional filters."""
        query = self.db.query(func.count(Alert.id))
        
        if status:
            query = query.filter(Alert.status == status)
        
        if severity:
            query = query.filter(Alert.severity == severity)
        
        return query.scalar()
    
    def get_stats(self, days: int = 7) -> dict:
        """Get alert statistics."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        # Status distribution
        status_counts = self.db.query(
            Alert.status,
            func.count(Alert.id).label('count')
        ).group_by(Alert.status).all()
        
        # Severity distribution
        severity_counts = self.db.query(
            Alert.severity,
            func.count(Alert.id).label('count')
        ).group_by(Alert.severity).all()
        
        # Recent counts
        recent = self.db.query(func.count(Alert.id)).filter(
            Alert.created_at >= cutoff
        ).scalar()
        
        # Average resolution time
        resolved = self.db.query(Alert).filter(
            Alert.status == AlertStatus.RESOLVED.value,
            Alert.resolved_at.isnot(None)
        ).all()
        
        avg_resolution = None
        if resolved:
            times = [
                (a.resolved_at - a.created_at).total_seconds() / 3600
                for a in resolved
                if a.resolved_at and a.created_at
            ]
            if times:
                avg_resolution = sum(times) / len(times)
        
        return {
            "total_alerts": self.count(),
            "pending_alerts": self.count(status=AlertStatus.PENDING.value),
            "alerts_last_n_days": recent,
            "status_distribution": {r.status: r.count for r in status_counts},
            "severity_distribution": {r.severity: r.count for r in severity_counts},
            "average_resolution_hours": round(avg_resolution, 2) if avg_resolution else None,
        }
    
    def delete(self, alert_id: int) -> bool:
        """Delete an alert by ID."""
        alert = self.get_by_id(alert_id)
        if alert:
            self.db.delete(alert)
            self.db.commit()
            return True
        return False
    
    def delete_old_resolved(self, days: int = 30) -> int:
        """Delete resolved/dismissed alerts older than N days."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        count = self.db.query(Alert).filter(
            Alert.status.in_([AlertStatus.RESOLVED.value, AlertStatus.DISMISSED.value]),
            Alert.created_at < cutoff
        ).delete()
        self.db.commit()
        return count
