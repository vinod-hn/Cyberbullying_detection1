"""
User Repository - CRUD operations for users.
"""

from typing import List, Optional
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import desc, func

# Handle both relative and absolute imports
try:
    from ..models import User
except ImportError:
    from models import User


class UserRepository:
    """Repository for User CRUD operations."""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create(
        self,
        username: str,
        email: Optional[str] = None,
        password_hash: Optional[str] = None,
        display_name: Optional[str] = None,
        role: str = "viewer"
    ) -> User:
        """Create a new user."""
        user = User(
            username=username,
            email=email,
            password_hash=password_hash,
            display_name=display_name or username,
            role=role
        )
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)
        return user
    
    def get_by_id(self, user_id: int) -> Optional[User]:
        """Get user by ID."""
        return self.db.query(User).filter(User.id == user_id).first()
    
    def get_by_uuid(self, user_uuid: str) -> Optional[User]:
        """Get user by UUID."""
        return self.db.query(User).filter(User.user_id == user_uuid).first()
    
    def get_by_username(self, username: str) -> Optional[User]:
        """Get user by username."""
        return self.db.query(User).filter(User.username == username).first()
    
    def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        return self.db.query(User).filter(User.email == email).first()
    
    def get_all(
        self,
        skip: int = 0,
        limit: int = 100,
        role: Optional[str] = None,
        is_active: Optional[bool] = None
    ) -> List[User]:
        """Get all users with filters and pagination."""
        query = self.db.query(User)
        
        if role:
            query = query.filter(User.role == role)
        
        if is_active is not None:
            query = query.filter(User.is_active == is_active)
        
        return query.order_by(desc(User.created_at)).offset(skip).limit(limit).all()
    
    def get_high_risk_users(self, min_risk_score: float = 0.5) -> List[User]:
        """Get users with high risk scores."""
        return self.db.query(User).filter(
            User.risk_score >= min_risk_score
        ).order_by(desc(User.risk_score)).all()
    
    def update_last_active(self, user_id: int) -> bool:
        """Update user's last active timestamp."""
        user = self.get_by_id(user_id)
        if user:
            user.last_active = datetime.utcnow()
            self.db.commit()
            return True
        return False
    
    def update_risk_score(self, user_id: int, risk_score: float) -> bool:
        """Update user's risk score."""
        user = self.get_by_id(user_id)
        if user:
            user.risk_score = risk_score
            self.db.commit()
            return True
        return False
    
    def increment_message_count(
        self,
        user_id: int,
        is_flagged: bool = False
    ) -> bool:
        """Increment user's message counts."""
        user = self.get_by_id(user_id)
        if user:
            user.total_messages += 1
            if is_flagged:
                user.flagged_messages += 1
            self.db.commit()
            return True
        return False
    
    def update_password(self, user_id: int, password_hash: str) -> bool:
        """Update user's password hash."""
        user = self.get_by_id(user_id)
        if user:
            user.password_hash = password_hash
            self.db.commit()
            return True
        return False
    
    def deactivate(self, user_id: int) -> bool:
        """Deactivate a user."""
        user = self.get_by_id(user_id)
        if user:
            user.is_active = False
            self.db.commit()
            return True
        return False
    
    def activate(self, user_id: int) -> bool:
        """Activate a user."""
        user = self.get_by_id(user_id)
        if user:
            user.is_active = True
            self.db.commit()
            return True
        return False
    
    def count(self, role: Optional[str] = None, is_active: Optional[bool] = None) -> int:
        """Count users with optional filters."""
        query = self.db.query(func.count(User.id))
        
        if role:
            query = query.filter(User.role == role)
        
        if is_active is not None:
            query = query.filter(User.is_active == is_active)
        
        return query.scalar()
    
    def delete(self, user_id: int) -> bool:
        """Delete a user by ID."""
        user = self.get_by_id(user_id)
        if user:
            self.db.delete(user)
            self.db.commit()
            return True
        return False
    
    def authenticate(self, username: str, password_hash: str) -> Optional[User]:
        """Authenticate user by username and password hash."""
        return self.db.query(User).filter(
            User.username == username,
            User.password_hash == password_hash,
            User.is_active == True
        ).first()
