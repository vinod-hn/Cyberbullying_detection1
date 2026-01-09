"""
Application configuration for Cyberbullying Detection API.
"""

from pydantic_settings import BaseSettings
from typing import List, Optional
from pathlib import Path
import os


class Settings(BaseSettings):
    """API Configuration settings."""
    
    # Application
    app_name: str = "Cyberbullying Detection API"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True  # Hot reload for development
    
    # CORS
    cors_origins: List[str] = ["*"]
    cors_allow_credentials: bool = True
    cors_allow_methods: List[str] = ["*"]
    cors_allow_headers: List[str] = ["*"]
    
    # Authentication
    require_auth: bool = False  # Set True to require auth on all endpoints
    jwt_secret: Optional[str] = None  # Auto-generated if not set
    token_expire_minutes: int = 60
    
    # Model settings
    default_model: str = "bert"
    model_cache_enabled: bool = True
    
    # Logging
    log_level: str = "INFO"
    log_requests: bool = True
    log_responses: bool = False
    
    # Paths
    project_root: Path = Path(__file__).parent.parent
    models_dir: Path = project_root / "03_models" / "saved_models"
    logs_dir: Path = project_root / "17_logs"
    
    # Local config flag
    local_config: bool = True  # True for local development
    
    class Config:
        env_prefix = "CYBERBULLYING_"
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings (use as FastAPI dependency)."""
    return settings


# Ensure directories exist
settings.logs_dir.mkdir(exist_ok=True)
