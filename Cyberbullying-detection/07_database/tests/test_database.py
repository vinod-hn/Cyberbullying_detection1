"""
Database tests for Cyberbullying Detection System.

Tests all repositories and database operations.
"""

import pytest
import sys
import importlib.util
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent.parent
db_path = project_root / "07_database"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(db_path))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Now we can directly import since we added db_path to sys.path
import db_config
import models


class TestDatabaseSetup:
    """Test database configuration and initialization."""
    
    @pytest.fixture
    def test_db(self):
        """Create an in-memory test database."""
        engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        
        db_config.Base.metadata.create_all(bind=engine)
        TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        
        db = TestSessionLocal()
        yield db
        db.close()
    
    def test_tables_created(self, test_db):
        """Test that all tables are created."""
        # Just verify models can be queried
        assert test_db.query(models.Message).count() == 0
        assert test_db.query(models.Prediction).count() == 0
        assert test_db.query(models.User).count() == 0
        assert test_db.query(models.Alert).count() == 0
        assert test_db.query(models.AuditLog).count() == 0
        assert test_db.query(models.Feedback).count() == 0


# Import repositories after models are available
from repositories.message_repository import MessageRepository
from repositories.prediction_repository import PredictionRepository
from repositories.user_repository import UserRepository
from repositories.alert_repository import AlertRepository
from repositories.audit_log_repository import AuditLogRepository
from repositories.feedback_repository import FeedbackRepository


class TestMessageRepository:
    """Test MessageRepository operations."""
    
    @pytest.fixture
    def db_session(self):
        """Create a fresh test database session."""
        engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        
        db_config.Base.metadata.create_all(bind=engine)
        TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        
        db = TestSessionLocal()
        yield db
        db.close()
    
    def test_create_message(self, db_session):
        """Test creating a message."""
        repo = MessageRepository(db_session)
        message = repo.create(
            text="Hello, this is a test message",
            source="test",
            language="en"
        )
        
        assert message.id is not None
        assert message.message_id is not None
        assert message.text == "Hello, this is a test message"
        assert message.source == "test"
    
    def test_get_message_by_id(self, db_session):
        """Test retrieving a message by ID."""
        repo = MessageRepository(db_session)
        created = repo.create(text="Test message")
        
        retrieved = repo.get_by_id(created.id)
        assert retrieved is not None
        assert retrieved.text == "Test message"
    
    def test_count_messages(self, db_session):
        """Test counting messages."""
        repo = MessageRepository(db_session)
        repo.create(text="Message 1", source="api")
        repo.create(text="Message 2", source="api")
        repo.create(text="Message 3", source="batch")
        
        assert repo.count() == 3
        assert repo.count(source="api") == 2
        assert repo.count(source="batch") == 1


class TestPredictionRepository:
    """Test PredictionRepository operations."""
    
    @pytest.fixture
    def db_session(self):
        """Create a fresh test database session."""
        engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        
        db_config.Base.metadata.create_all(bind=engine)
        TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        
        db = TestSessionLocal()
        yield db
        db.close()
    
    def test_create_with_message(self, db_session):
        """Test creating a prediction with its message."""
        repo = PredictionRepository(db_session)
        message, prediction = repo.create_with_message(
            text="You are so stupid",
            model_type="bert",
            predicted_label="harassment",
            confidence=0.95,
            is_cyberbullying=True,
            probabilities={"harassment": 0.95, "neutral": 0.05}
        )
        
        assert message.id is not None
        assert prediction.id is not None
        assert prediction.message_id == message.id
        assert prediction.is_cyberbullying is True
        assert prediction.confidence == 0.95
    
    def test_get_label_distribution(self, db_session):
        """Test getting label distribution."""
        repo = PredictionRepository(db_session)
        
        # Create several predictions
        for label in ["neutral", "neutral", "harassment", "threat"]:
            repo.create_with_message(
                text=f"Test text for {label}",
                model_type="bert",
                predicted_label=label,
                confidence=0.9,
                is_cyberbullying=label != "neutral"
            )
        
        distribution = repo.get_label_distribution()
        assert distribution["neutral"] == 2
        assert distribution["harassment"] == 1
        assert distribution["threat"] == 1
    
    def test_count_cyberbullying(self, db_session):
        """Test counting cyberbullying predictions."""
        repo = PredictionRepository(db_session)
        
        repo.create_with_message("Nice day!", "bert", "neutral", 0.99, False)
        repo.create_with_message("You're ugly!", "bert", "insult", 0.95, True)
        repo.create_with_message("Hello!", "bert", "neutral", 0.98, False)
        
        assert repo.count() == 3
        assert repo.count(is_cyberbullying=True) == 1
        assert repo.count(is_cyberbullying=False) == 2


class TestUserRepository:
    """Test UserRepository operations."""
    
    @pytest.fixture
    def db_session(self):
        """Create a fresh test database session."""
        engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        
        db_config.Base.metadata.create_all(bind=engine)
        TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        
        db = TestSessionLocal()
        yield db
        db.close()
    
    def test_create_user(self, db_session):
        """Test creating a user."""
        repo = UserRepository(db_session)
        user = repo.create(
            username="testuser",
            email="test@example.com",
            role="admin"
        )
        
        assert user.id is not None
        assert user.username == "testuser"
        assert user.role == "admin"
        assert user.is_active is True
    
    def test_get_by_username(self, db_session):
        """Test getting user by username."""
        repo = UserRepository(db_session)
        repo.create(username="john_doe")
        
        user = repo.get_by_username("john_doe")
        assert user is not None
        assert user.username == "john_doe"
    
    def test_update_risk_score(self, db_session):
        """Test updating user risk score."""
        repo = UserRepository(db_session)
        user = repo.create(username="risky_user")
        
        assert user.risk_score == 0.0
        
        repo.update_risk_score(user.id, 0.75)
        
        updated = repo.get_by_id(user.id)
        assert updated.risk_score == 0.75


class TestAlertRepository:
    """Test AlertRepository operations."""
    
    @pytest.fixture
    def db_session(self):
        """Create a fresh test database session."""
        engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        
        db_config.Base.metadata.create_all(bind=engine)
        TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        
        db = TestSessionLocal()
        yield db
        db.close()
    
    def test_create_and_resolve_alert(self, db_session):
        """Test creating and resolving an alert."""
        pred_repo = PredictionRepository(db_session)
        alert_repo = AlertRepository(db_session)
        
        # Create a prediction first
        _, prediction = pred_repo.create_with_message(
            text="Threatening message",
            model_type="bert",
            predicted_label="threat",
            confidence=0.98,
            is_cyberbullying=True
        )
        
        # Create alert
        alert = alert_repo.create(
            prediction_id=prediction.id,
            severity="high",
            reason="High confidence threat detected"
        )
        
        assert alert.status == "pending"
        assert alert.severity == "high"
        
        # Resolve alert
        alert_repo.resolve(alert.id, "moderator1", "False positive")
        
        resolved = alert_repo.get_by_id(alert.id)
        assert resolved.status == "resolved"
        assert resolved.resolved_by == "moderator1"


class TestFeedbackRepository:
    """Test FeedbackRepository operations."""
    
    @pytest.fixture
    def db_session(self):
        """Create a fresh test database session."""
        engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        
        db_config.Base.metadata.create_all(bind=engine)
        TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        
        db = TestSessionLocal()
        yield db
        db.close()
    
    def test_create_feedback(self, db_session):
        """Test creating feedback."""
        repo = FeedbackRepository(db_session)
        feedback = repo.create(
            prediction_id="test-pred-123",
            is_correct=False,
            correct_label="harassment",
            comments="Should have been harassment"
        )
        
        assert feedback.id is not None
        assert feedback.is_correct is False
        assert feedback.correct_label == "harassment"
    
    def test_accuracy_stats(self, db_session):
        """Test getting accuracy statistics."""
        repo = FeedbackRepository(db_session)
        
        # Create mixed feedback
        repo.create("pred-1", is_correct=True)
        repo.create("pred-2", is_correct=True)
        repo.create("pred-3", is_correct=True)
        repo.create("pred-4", is_correct=False, correct_label="threat")
        repo.create("pred-5", is_correct=False, correct_label="harassment")
        
        stats = repo.get_accuracy_stats()
        
        assert stats["total_feedback"] == 5
        assert stats["correct_predictions"] == 3
        assert stats["incorrect_predictions"] == 2
        assert stats["accuracy_percentage"] == 60.0


class TestAuditLogRepository:
    """Test AuditLogRepository operations."""
    
    @pytest.fixture
    def db_session(self):
        """Create a fresh test database session."""
        engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        
        db_config.Base.metadata.create_all(bind=engine)
        TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        
        db = TestSessionLocal()
        yield db
        db.close()
    
    def test_log_prediction(self, db_session):
        """Test logging a prediction."""
        repo = AuditLogRepository(db_session)
        log = repo.log_prediction(
            prediction_id="pred-123",
            ip_address="127.0.0.1",
            model_type="bert"
        )
        
        assert log.action == "predict"
        assert log.resource_type == "prediction"
        assert log.resource_id == "pred-123"
    
    def test_get_action_distribution(self, db_session):
        """Test getting action distribution."""
        repo = AuditLogRepository(db_session)
        
        repo.create(action="predict")
        repo.create(action="predict")
        repo.create(action="login")
        repo.create(action="view_alert")
        
        distribution = repo.get_action_distribution(days=7)
        
        assert distribution["predict"] == 2
        assert distribution["login"] == 1
        assert distribution["view_alert"] == 1


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
