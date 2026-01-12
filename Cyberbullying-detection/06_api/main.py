"""
Cyberbullying Detection API - Main Application Entry Point

Run with:
    uvicorn 06_api.main:app --reload --host 0.0.0.0 --port 8000
    
Or directly:
    python -m 06_api.main
"""

import sys
from pathlib import Path
from contextlib import asynccontextmanager

# Add parent directory to path for imports
api_dir = Path(__file__).parent
project_dir = api_dir.parent
sys.path.insert(0, str(project_dir))
sys.path.insert(0, str(api_dir))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import base64
import logging

# Handle both relative and absolute imports
try:
    from .app_config import settings
    from .routes import api_router
    from .middleware import LoggingMiddleware, AuthMiddleware, setup_exception_handlers
except ImportError:
    from app_config import settings
    from routes import api_router
    from middleware import LoggingMiddleware, AuthMiddleware, setup_exception_handlers

# Setup logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# Lifespan context manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup and shutdown events."""
    # Startup
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Documentation available at: http://{settings.host}:{settings.port}/docs")
    logger.info(f"Health check at: http://{settings.host}:{settings.port}/health")
    
    # Initialize database
    try:
        try:
            from .db_helper import init_db, get_db_info
        except ImportError:
            from db_helper import init_db, get_db_info
        init_db()
        db_info = get_db_info()
        logger.info(f"Database initialized: {db_info.get('database_path', 'N/A')}")
    except Exception as e:
        logger.warning(f"Database initialization skipped: {e}")
    
    yield
    # Shutdown
    logger.info("Shutting down API...")
    try:
        from .models_loader import clear_model_cache
    except ImportError:
        from models_loader import clear_model_cache
    clear_model_cache()


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    description="""
    API for detecting cyberbullying in text using pre-trained ML models.
    
    ## Features
    - Multiple model options (BERT, mBERT, IndicBERT, Baseline)
    - Single and batch prediction endpoints
    - Conversation-level analysis with escalation detection
    - Confidence scores and probabilities
    - Support for Kannada-English code-mixed text
    
    ## Models Performance (Updated: Colab Training)
    - **BERT**: 100.00% F1 Score (Best - Perfect on test set)
    - **mBERT**: 99.86% F1 Score (Multilingual)
    - **IndicBERT**: 99.93% F1 Score (Indian Languages)
    - **Baseline**: ~95% F1 Score (Fastest)
    
    ## Authentication
    For protected endpoints, use the `/auth/login` endpoint to get a JWT token.
    Default test users: `admin/admin123`, `analyst/analyst123`
    """,
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_allow_methods,
    allow_headers=settings.cors_allow_headers,
)

# Add custom middleware
if settings.log_requests:
    app.add_middleware(LoggingMiddleware)

if settings.require_auth:
    app.add_middleware(AuthMiddleware, require_auth=True)

# Setup exception handlers
setup_exception_handlers(app)

# Include API routes
app.include_router(api_router)

# Minimal transparent 1x1 PNG favicon
FAVICON_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
)


@app.get("/", tags=["Root"])
async def root():
    """API root endpoint with navigation links."""
    return {
        "message": "Cyberbullying Detection API",
        "version": settings.app_version,
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/health",
        "endpoints": {
            "predict": "POST /predict",
            "batch_predict": "POST /predict/batch",
            "conversation": "POST /predict/conversation",
            "stats": "GET /stats",
            "models": "GET /models",
            "feedback": "POST /feedback"
        }
    }


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Serve minimal favicon to avoid 404s."""
    return Response(
        content=base64.b64decode(FAVICON_PNG_BASE64),
        media_type="image/png"
    )


# CLI entry point
if __name__ == "__main__":
    import uvicorn
    
    print(f"\n🚀 Starting {settings.app_name}...")
    print(f"📖 API Docs: http://localhost:{settings.port}/docs")
    print(f"🔄 Health Check: http://localhost:{settings.port}/health\n")
    
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload
    )
