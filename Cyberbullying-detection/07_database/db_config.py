"""
Database configuration for SQLite.

Provides connection management, session handling, and initialization.
"""

import os
from pathlib import Path
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import StaticPool
from contextlib import contextmanager

# Database path - stored in project root
PROJECT_ROOT = Path(__file__).parent.parent
DATABASE_PATH = PROJECT_ROOT / "local.db"
DATABASE_URL = f"sqlite:///{DATABASE_PATH}"

# For testing, use in-memory database
TEST_DATABASE_URL = "sqlite:///:memory:"

# Create engine with SQLite-specific settings
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # Required for SQLite with FastAPI
    poolclass=StaticPool,  # Better for SQLite
    echo=False,  # Set to True for SQL debugging
)

# Enable foreign keys for SQLite
@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for all models
Base = declarative_base()


def get_db():
    """
    Dependency for FastAPI endpoints.
    Yields a database session and ensures cleanup.
    
    Usage:
        @app.get("/items")
        def read_items(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_context():
    """
    Context manager for database sessions outside FastAPI.
    
    Usage:
        with get_db_context() as db:
            db.query(...)
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init_db():
    """
    Initialize the database by creating all tables.
    Call this at application startup.
    """
    # Import models to ensure they're registered with Base
    from . import models  # noqa: F401
    
    Base.metadata.create_all(bind=engine)
    print(f"Database initialized at: {DATABASE_PATH}")
    return True


def drop_db():
    """
    Drop all tables. Use with caution!
    """
    Base.metadata.drop_all(bind=engine)
    print("All database tables dropped.")


def get_test_db():
    """
    Create a test database session (in-memory).
    """
    test_engine = create_engine(
        TEST_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    
    @event.listens_for(test_engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()
    
    TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    
    # Import and create tables
    from . import models  # noqa: F401
    Base.metadata.create_all(bind=test_engine)
    
    return TestSessionLocal


# Database info helper
def get_db_info():
    """Get database information."""
    return {
        "database_type": "SQLite",
        "database_path": str(DATABASE_PATH),
        "exists": DATABASE_PATH.exists(),
        "size_bytes": DATABASE_PATH.stat().st_size if DATABASE_PATH.exists() else 0,
    }
