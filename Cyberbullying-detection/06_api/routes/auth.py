"""
Authentication endpoint for local JWT.
POST /auth/login, /auth/refresh
"""

from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from datetime import datetime, timedelta
import hashlib
import secrets
from typing import Optional

try:
    from ..schemas import LoginRequest, TokenResponse
except ImportError:
    from schemas import LoginRequest, TokenResponse

router = APIRouter()
security = HTTPBearer(auto_error=False)

# Configuration
SECRET_KEY = secrets.token_hex(32)  # Generated at startup; use env var in production
TOKEN_EXPIRE_MINUTES = 60

# Simple user store (replace with database in production)
_users = {
    "admin": hashlib.sha256("admin123".encode()).hexdigest(),
    "analyst": hashlib.sha256("analyst123".encode()).hexdigest(),
}

# Token store (in-memory; use Redis/DB in production)
_tokens: dict[str, dict] = {}


def hash_password(password: str) -> str:
    """Hash password using SHA256."""
    return hashlib.sha256(password.encode()).hexdigest()


def create_token(username: str) -> tuple[str, datetime]:
    """Create a new access token."""
    token = secrets.token_urlsafe(32)
    expires_at = datetime.now() + timedelta(minutes=TOKEN_EXPIRE_MINUTES)
    
    _tokens[token] = {
        "username": username,
        "expires_at": expires_at
    }
    
    return token, expires_at


def verify_token(token: str) -> Optional[str]:
    """Verify token and return username if valid."""
    if token not in _tokens:
        return None
    
    token_data = _tokens[token]
    if datetime.now() > token_data["expires_at"]:
        del _tokens[token]
        return None
    
    return token_data["username"]


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Optional[str]:
    """Dependency to get current authenticated user."""
    if credentials is None:
        return None
    
    username = verify_token(credentials.credentials)
    return username


@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest):
    """
    Authenticate user and return JWT token.
    
    - **username**: User's username
    - **password**: User's password
    
    Default users for local testing:
    - admin / admin123
    - analyst / analyst123
    """
    # Verify credentials
    if request.username not in _users:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    
    password_hash = hash_password(request.password)
    if _users[request.username] != password_hash:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    
    # Create token
    token, expires_at = create_token(request.username)
    expires_in = int((expires_at - datetime.now()).total_seconds())
    
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in=expires_in
    )


@router.post("/logout")
async def logout(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Invalidate current token.
    """
    if credentials and credentials.credentials in _tokens:
        del _tokens[credentials.credentials]
        return {"message": "Successfully logged out"}
    
    return {"message": "No active session"}


@router.get("/me")
async def get_current_user_info(user: str = Depends(get_current_user)):
    """
    Get current authenticated user info.
    """
    if user is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    return {"username": user, "authenticated": True}
