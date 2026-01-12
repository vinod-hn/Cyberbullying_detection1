"""
Message Repository - CRUD operations for messages.
"""

from typing import List, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import desc, func

# Handle both relative and absolute imports
try:
    from ..models import Message
except ImportError:
    from models import Message


class MessageRepository:
    """Repository for Message CRUD operations."""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create(
        self,
        text: str,
        source: str = "api",
        language: str = "en",
        metadata: dict = None
    ) -> Message:
        """Create a new message."""
        message = Message(
            text=text,
            source=source,
            language=language,
            metadata=metadata
        )
        self.db.add(message)
        self.db.commit()
        self.db.refresh(message)
        return message
    
    def get_by_id(self, message_id: int) -> Optional[Message]:
        """Get message by ID."""
        return self.db.query(Message).filter(Message.id == message_id).first()
    
    def get_by_uuid(self, message_uuid: str) -> Optional[Message]:
        """Get message by UUID."""
        return self.db.query(Message).filter(Message.message_id == message_uuid).first()
    
    def get_all(
        self,
        skip: int = 0,
        limit: int = 100,
        source: Optional[str] = None
    ) -> List[Message]:
        """Get all messages with pagination."""
        query = self.db.query(Message)
        
        if source:
            query = query.filter(Message.source == source)
        
        return query.order_by(desc(Message.created_at)).offset(skip).limit(limit).all()
    
    def get_recent(self, hours: int = 24) -> List[Message]:
        """Get messages from the last N hours."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        return self.db.query(Message).filter(
            Message.created_at >= cutoff
        ).order_by(desc(Message.created_at)).all()
    
    def search(self, query: str, limit: int = 50) -> List[Message]:
        """Search messages by text content."""
        return self.db.query(Message).filter(
            Message.text.ilike(f"%{query}%")
        ).order_by(desc(Message.created_at)).limit(limit).all()
    
    def count(self, source: Optional[str] = None) -> int:
        """Count total messages."""
        query = self.db.query(func.count(Message.id))
        if source:
            query = query.filter(Message.source == source)
        return query.scalar()
    
    def count_by_date(self, days: int = 7) -> dict:
        """Count messages per day for the last N days."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        results = self.db.query(
            func.date(Message.created_at).label('date'),
            func.count(Message.id).label('count')
        ).filter(
            Message.created_at >= cutoff
        ).group_by(
            func.date(Message.created_at)
        ).all()
        
        return {str(r.date): r.count for r in results}
    
    def delete(self, message_id: int) -> bool:
        """Delete a message by ID."""
        message = self.get_by_id(message_id)
        if message:
            self.db.delete(message)
            self.db.commit()
            return True
        return False
    
    def delete_old(self, days: int = 90) -> int:
        """Delete messages older than N days. Returns count of deleted."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        count = self.db.query(Message).filter(Message.created_at < cutoff).delete()
        self.db.commit()
        return count
