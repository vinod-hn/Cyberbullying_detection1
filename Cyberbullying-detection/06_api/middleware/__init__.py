"""
API Middleware package.
"""

try:
    from .auth_middleware import AuthMiddleware
    from .logging_middleware import LoggingMiddleware
    from .error_handler import setup_exception_handlers
except ImportError:
    from auth_middleware import AuthMiddleware
    from logging_middleware import LoggingMiddleware
    from error_handler import setup_exception_handlers

__all__ = [
    "AuthMiddleware",
    "LoggingMiddleware", 
    "setup_exception_handlers",
]
