"""
Performance tests for Cyberbullying Detection API.
"""

import pytest
import time
from fastapi.testclient import TestClient
from concurrent.futures import ThreadPoolExecutor, as_completed

import sys
from pathlib import Path
# Add 'Cyberbullying-detection' dir to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from 06_api.main import app

client = TestClient(app)


class TestResponseTimes:
    """Tests for API response time benchmarks."""
    
    def test_health_response_time(self):
        """Health endpoint should respond within 100ms."""
        start = time.time()
        response = client.get("/health")
        elapsed = (time.time() - start) * 1000
        
        assert response.status_code == 200
        assert elapsed < 100, f"Health check took {elapsed:.2f}ms (expected <100ms)"
    
    def test_single_predict_response_time(self):
        """Single prediction should respond within 500ms (mock mode)."""
        start = time.time()
        response = client.post("/predict", json={
            "text": "This is a test message for performance testing"
        })
        elapsed = (time.time() - start) * 1000
        
        assert response.status_code == 200
        assert elapsed < 500, f"Prediction took {elapsed:.2f}ms (expected <500ms)"
    
    def test_batch_predict_response_time(self):
        """Batch prediction (10 texts) should respond within 1000ms (mock mode)."""
        texts = [f"Test message number {i}" for i in range(10)]
        
        start = time.time()
        response = client.post("/predict/batch", json={
            "texts": texts
        })
        elapsed = (time.time() - start) * 1000
        
        assert response.status_code == 200
        assert elapsed < 1000, f"Batch prediction took {elapsed:.2f}ms (expected <1000ms)"


class TestConcurrency:
    """Tests for concurrent request handling."""
    
    def test_concurrent_health_checks(self):
        """API should handle 10 concurrent health checks."""
        num_requests = 10
        results = []
        
        def make_request():
            response = client.get("/health")
            return response.status_code
        
        with ThreadPoolExecutor(max_workers=num_requests) as executor:
            futures = [executor.submit(make_request) for _ in range(num_requests)]
            results = [f.result() for f in as_completed(futures)]
        
        assert all(status == 200 for status in results)
        assert len(results) == num_requests
    
    def test_concurrent_predictions(self):
        """API should handle 5 concurrent predictions."""
        num_requests = 5
        
        def make_prediction(i):
            response = client.post("/predict", json={
                "text": f"Concurrent test message {i}"
            })
            return response.status_code
        
        with ThreadPoolExecutor(max_workers=num_requests) as executor:
            futures = [executor.submit(make_prediction, i) for i in range(num_requests)]
            results = [f.result() for f in as_completed(futures)]
        
        assert all(status == 200 for status in results)


class TestThroughput:
    """Tests for API throughput."""
    
    def test_predictions_per_second(self):
        """Measure predictions per second (target: >10 in mock mode)."""
        num_requests = 20
        
        start = time.time()
        for i in range(num_requests):
            response = client.post("/predict", json={
                "text": f"Throughput test message {i}"
            })
            assert response.status_code == 200
        
        elapsed = time.time() - start
        throughput = num_requests / elapsed
        
        print(f"\nThroughput: {throughput:.2f} predictions/second")
        assert throughput > 10, f"Throughput too low: {throughput:.2f} pred/s"


class TestLargePayloads:
    """Tests for handling large payloads."""
    
    def test_large_text_prediction(self):
        """API should handle large text inputs."""
        large_text = "This is a test message. " * 500  # ~12KB
        
        response = client.post("/predict", json={
            "text": large_text
        })
        assert response.status_code == 200
    
    def test_large_batch_prediction(self):
        """API should handle large batch requests."""
        texts = [f"Batch test message {i}" for i in range(100)]
        
        response = client.post("/predict/batch", json={
            "texts": texts,
            "batch_size": 32
        })
        assert response.status_code == 200
        assert response.json()["total"] == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
