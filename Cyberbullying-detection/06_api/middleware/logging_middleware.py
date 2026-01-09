"""
Request/Response logging middleware.
Logs to 17_logs/ directory.
"""

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
import time
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

# Setup API logger
LOG_DIR = Path(__file__).parent.parent.parent / "17_logs"
LOG_DIR.mkdir(exist_ok=True)

# Configure file handler for API logs
api_logger = logging.getLogger("api")
api_logger.setLevel(logging.INFO)

# File handler
log_file = LOG_DIR / "api.log"
file_handler = logging.FileHandler(log_file, encoding="utf-8")
file_handler.setFormatter(logging.Formatter(
    "%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
))
api_logger.addHandler(file_handler)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(
    "%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
))
api_logger.addHandler(console_handler)


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to log all API requests and responses.
    """
    
    def __init__(self, app, log_request_body: bool = False, log_response_body: bool = False):
        """
        Initialize logging middleware.
        
        Args:
            app: FastAPI application
            log_request_body: Whether to log request bodies (careful with sensitive data)
            log_response_body: Whether to log response bodies
        """
        super().__init__(app)
        self.log_request_body = log_request_body
        self.log_response_body = log_response_body
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Log request details and timing."""
        request_id = datetime.now().strftime("%Y%m%d%H%M%S%f")
        start_time = time.time()
        
        # Log request
        client_ip = request.client.host if request.client else "unknown"
        log_data = {
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "query": str(request.query_params) if request.query_params else None,
            "client_ip": client_ip,
            "user_agent": request.headers.get("user-agent", "unknown")[:100],
        }
        
        api_logger.info(f"REQUEST | {request.method} {request.url.path} | client={client_ip}")
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate processing time
            process_time = (time.time() - start_time) * 1000
            
            # Log response
            api_logger.info(
                f"RESPONSE | {request.method} {request.url.path} | "
                f"status={response.status_code} | time={process_time:.2f}ms"
            )
            
            # Add timing header
            response.headers["X-Process-Time-Ms"] = f"{process_time:.2f}"
            response.headers["X-Request-ID"] = request_id
            
            return response
            
        except Exception as e:
            process_time = (time.time() - start_time) * 1000
            api_logger.error(
                f"ERROR | {request.method} {request.url.path} | "
                f"error={str(e)} | time={process_time:.2f}ms"
            )
            raise


def get_logger() -> logging.Logger:
    """Get the API logger instance."""
    return api_logger
