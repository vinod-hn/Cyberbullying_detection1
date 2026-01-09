"""
Authentication middleware for API request validation.
"""

from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

# Paths that don't require authentication
PUBLIC_PATHS = [
    "/",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/favicon.ico",
    "/health",
    "/auth/login",
]


class AuthMiddleware(BaseHTTPMiddleware):
    """
    Middleware to validate authentication tokens on protected routes.
    """
    
    def __init__(self, app, require_auth: bool = False, public_paths: Optional[List[str]] = None):
        """
        Initialize auth middleware.
        
        Args:
            app: FastAPI application
            require_auth: If True, require authentication on all non-public paths
            public_paths: List of paths that don't require authentication
        """
        super().__init__(app)
        self.require_auth = require_auth
        self.public_paths = public_paths or PUBLIC_PATHS
    
    async def dispatch(self, request: Request, call_next):
        """Process the request and validate authentication if required."""
        path = request.url.path
        
        # Skip auth for public paths
        if not self.require_auth or self._is_public_path(path):
            return await call_next(request)
        
        # Check for Authorization header
        auth_header = request.headers.get("Authorization")
        
        if not auth_header:
            return JSONResponse(
                status_code=401,
                content={"detail": "Authorization header required"}
            )
        
        # Validate Bearer token format
        if not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid authorization header format. Use: Bearer <token>"}
            )
        
        token = auth_header.split(" ")[1]
        
        # Validate token (import here to avoid circular imports)
        from ..routes.auth import verify_token
        username = verify_token(token)
        
        if username is None:
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or expired token"}
            )
        
        # Add user info to request state
        request.state.user = username
        
        logger.debug(f"Authenticated request from user: {username}")
        
        return await call_next(request)
    
    def _is_public_path(self, path: str) -> bool:
        """Check if the path is in the public paths list."""
        for public_path in self.public_paths:
            if path == public_path or path.startswith(public_path + "/"):
                return True
        return False
