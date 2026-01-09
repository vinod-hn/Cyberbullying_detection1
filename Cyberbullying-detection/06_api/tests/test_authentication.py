"""
Authentication tests for Cyberbullying Detection API.
"""

import pytest
from fastapi.testclient import TestClient

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import app

client = TestClient(app)


class TestLoginEndpoint:
    """Tests for /auth/login endpoint."""
    
    def test_login_valid_credentials(self):
        """Login with valid credentials should return token."""
        response = client.post("/auth/login", json={
            "username": "admin",
            "password": "admin123"
        })
        assert response.status_code == 200
        
        data = response.json()
        assert "access_token" in data
        assert "token_type" in data
        assert "expires_in" in data
        assert data["token_type"] == "bearer"
    
    def test_login_invalid_username(self):
        """Login with invalid username should fail."""
        response = client.post("/auth/login", json={
            "username": "nonexistent",
            "password": "password"
        })
        assert response.status_code == 401
    
    def test_login_invalid_password(self):
        """Login with wrong password should fail."""
        response = client.post("/auth/login", json={
            "username": "admin",
            "password": "wrongpassword"
        })
        assert response.status_code == 401
    
    def test_login_missing_credentials(self):
        """Login without credentials should return validation error."""
        response = client.post("/auth/login", json={})
        assert response.status_code == 422


class TestAuthenticatedEndpoints:
    """Tests for authenticated endpoints."""
    
    def get_auth_token(self) -> str:
        """Helper to get auth token."""
        response = client.post("/auth/login", json={
            "username": "admin",
            "password": "admin123"
        })
        return response.json()["access_token"]
    
    def test_get_current_user_with_token(self):
        """Current user endpoint should work with valid token."""
        token = self.get_auth_token()
        
        response = client.get(
            "/auth/me",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["username"] == "admin"
        assert data["authenticated"] is True
    
    def test_get_current_user_without_token(self):
        """Current user endpoint should fail without token."""
        response = client.get("/auth/me")
        assert response.status_code == 401
    
    def test_logout(self):
        """Logout should invalidate token."""
        token = self.get_auth_token()
        
        # Logout
        response = client.post(
            "/auth/logout",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 200
        
        # Token should no longer work
        response = client.get(
            "/auth/me",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 401


class TestTokenValidation:
    """Tests for token validation."""
    
    def test_invalid_token_format(self):
        """Invalid token format should fail."""
        response = client.get(
            "/auth/me",
            headers={"Authorization": "InvalidFormat token123"}
        )
        assert response.status_code == 401
    
    def test_expired_token(self):
        """Expired token should fail (simulated)."""
        # Using a fake token that doesn't exist
        response = client.get(
            "/auth/me",
            headers={"Authorization": "Bearer fake_expired_token_123"}
        )
        assert response.status_code == 401


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
