"""
API route tests for Cyberbullying Detection API.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Import app
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import app

client = TestClient(app)


class TestHealthEndpoint:
    """Tests for /health endpoint."""
    
    def test_health_check_returns_200(self):
        """Health endpoint should return 200 OK."""
        response = client.get("/health")
        assert response.status_code == 200
    
    def test_health_check_response_structure(self):
        """Health response should have required fields."""
        response = client.get("/health")
        data = response.json()
        
        assert "status" in data
        assert "timestamp" in data
        assert "models_available" in data
        assert "default_model" in data
        assert data["status"] == "healthy"
    
    def test_models_endpoint(self):
        """Models endpoint should list available models."""
        response = client.get("/models")
        assert response.status_code == 200
        
        data = response.json()
        assert "models" in data
        assert "total" in data
        assert len(data["models"]) > 0


class TestPredictEndpoint:
    """Tests for /predict endpoint."""
    
    def test_predict_single_text(self):
        """Single prediction should work with valid input."""
        response = client.post("/predict", json={
            "text": "Hello, this is a test message"
        })
        assert response.status_code == 200
        
        data = response.json()
        assert "text" in data
        assert "prediction" in data
        assert "confidence" in data
        assert "is_cyberbullying" in data
    
    def test_predict_empty_text_fails(self):
        """Empty text should return validation error."""
        response = client.post("/predict", json={
            "text": ""
        })
        assert response.status_code == 422
    
    def test_predict_missing_text_fails(self):
        """Missing text field should return validation error."""
        response = client.post("/predict", json={})
        assert response.status_code == 422
    
    def test_predict_with_model_type(self):
        """Prediction with specific model type should work."""
        response = client.post("/predict", json={
            "text": "Test message",
            "model_type": "bert"
        })
        assert response.status_code == 200


class TestBatchPredictEndpoint:
    """Tests for /predict/batch endpoint."""
    
    def test_batch_predict_multiple_texts(self):
        """Batch prediction should work with multiple texts."""
        response = client.post("/predict/batch", json={
            "texts": ["Hello", "World", "Test"]
        })
        assert response.status_code == 200
        
        data = response.json()
        assert "predictions" in data
        assert "total" in data
        assert "processing_time_ms" in data
        assert data["total"] == 3
        assert len(data["predictions"]) == 3
    
    def test_batch_predict_empty_list_fails(self):
        """Empty texts list should return validation error."""
        response = client.post("/predict/batch", json={
            "texts": []
        })
        assert response.status_code == 422


class TestConversationEndpoint:
    """Tests for /predict/conversation endpoint."""
    
    def test_conversation_predict(self):
        """Conversation prediction should analyze message sequence."""
        response = client.post("/predict/conversation", json={
            "messages": [
                {"text": "Hi there!"},
                {"text": "How are you?"},
                {"text": "I'm doing well"}
            ]
        })
        assert response.status_code == 200
        
        data = response.json()
        assert "messages" in data
        assert "overall_risk_score" in data
        assert "escalation_detected" in data
        assert "flagged_messages" in data


class TestStatsEndpoint:
    """Tests for /stats endpoint."""
    
    def test_get_statistics(self):
        """Statistics endpoint should return dashboard data."""
        response = client.get("/stats")
        assert response.status_code == 200
        
        data = response.json()
        assert "total_predictions" in data
        assert "cyberbullying_count" in data
        assert "severity_distribution" in data


class TestFeedbackEndpoint:
    """Tests for /feedback endpoint."""
    
    def test_submit_feedback(self):
        """Feedback submission should work with valid input."""
        response = client.post("/feedback", json={
            "prediction_id": "test-123",
            "correct_label": "not_cyberbullying",
            "comments": "This was correctly classified"
        })
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
    
    def test_feedback_invalid_label_fails(self):
        """Invalid label should return 400 error."""
        response = client.post("/feedback", json={
            "prediction_id": "test-123",
            "correct_label": "invalid_label"
        })
        assert response.status_code == 400


class TestRootEndpoint:
    """Tests for root endpoint."""
    
    def test_root_returns_info(self):
        """Root endpoint should return API info."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "docs" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
