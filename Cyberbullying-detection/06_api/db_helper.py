"""
Database helper for API routes.
Provides clean imports for database functionality.
"""

import sys
from pathlib import Path

# Add project root and database path to path for imports
_project_root = Path(__file__).parent.parent
_db_path = _project_root / "07_database"

if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
if str(_db_path) not in sys.path:
    sys.path.insert(0, str(_db_path))
if str(_db_path / "repositories") not in sys.path:
    sys.path.insert(0, str(_db_path / "repositories"))

# Database imports
_db_module = None


def _get_db_module():
    """Lazy load database module."""
    global _db_module
    if _db_module is None:
        try:
            # Import database modules directly since we added paths
            import db_config
            import models
            
            # Import repositories
            import message_repository
            import prediction_repository
            import user_repository
            import alert_repository
            import audit_log_repository
            import feedback_repository
            
            _db_module = {
                "db_config": db_config,
                "models": models,
                "repositories": {
                    "message": message_repository,
                    "prediction": prediction_repository,
                    "user": user_repository,
                    "alert": alert_repository,
                    "audit_log": audit_log_repository,
                    "feedback": feedback_repository,
                }
            }
        except Exception as e:
            print(f"Failed to load database module: {e}")
            _db_module = {}
    return _db_module


def get_db():
    """Get database session dependency."""
    db_mod = _get_db_module()
    if db_mod and "db_config" in db_mod:
        return db_mod["db_config"].get_db()
    return iter([None])


def get_db_context():
    """Get database context manager."""
    db_mod = _get_db_module()
    if db_mod and "db_config" in db_mod:
        return db_mod["db_config"].get_db_context()
    return None


def init_db():
    """Initialize database."""
    db_mod = _get_db_module()
    if db_mod and "db_config" in db_mod:
        return db_mod["db_config"].init_db()
    return False


def get_db_info():
    """Get database info."""
    db_mod = _get_db_module()
    if db_mod and "db_config" in db_mod:
        return db_mod["db_config"].get_db_info()
    return {"database_type": "unavailable"}


# Repository factory functions
def get_prediction_repository(db):
    """Get prediction repository."""
    db_mod = _get_db_module()
    if db_mod and db and "repositories" in db_mod:
        PredictionRepository = db_mod["repositories"]["prediction"].PredictionRepository
        return PredictionRepository(db)
    return None


def get_message_repository(db):
    """Get message repository."""
    db_mod = _get_db_module()
    if db_mod and db and "repositories" in db_mod:
        MessageRepository = db_mod["repositories"]["message"].MessageRepository
        return MessageRepository(db)
    return None


def get_feedback_repository(db):
    """Get feedback repository."""
    db_mod = _get_db_module()
    if db_mod and db and "repositories" in db_mod:
        FeedbackRepository = db_mod["repositories"]["feedback"].FeedbackRepository
        return FeedbackRepository(db)
    return None


def get_audit_log_repository(db):
    """Get audit log repository."""
    db_mod = _get_db_module()
    if db_mod and db and "repositories" in db_mod:
        AuditLogRepository = db_mod["repositories"]["audit_log"].AuditLogRepository
        return AuditLogRepository(db)
    return None


def get_user_repository(db):
    """Get user repository."""
    db_mod = _get_db_module()
    if db_mod and db and "repositories" in db_mod:
        UserRepository = db_mod["repositories"]["user"].UserRepository
        return UserRepository(db)
    return None


def get_alert_repository(db):
    """Get alert repository."""
    db_mod = _get_db_module()
    if db_mod and db and "repositories" in db_mod:
        AlertRepository = db_mod["repositories"]["alert"].AlertRepository
        return AlertRepository(db)
    return None
