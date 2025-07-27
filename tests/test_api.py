from unittest.mock import patch

import pytest


class TestAPIEndpoints:
    """Unit tests for API endpoints"""

    @pytest.mark.unit
    @pytest.mark.api
    def test_health_check_success(self, client):
        """Test health check when model is ready"""
        with patch("app.main.predictor") as mock_predictor:
            mock_predictor.model.is_ready.return_value = True
            mock_predictor.model.model_name = "test-model"

            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["model"] == "test-model"

    @pytest.mark.unit
    @pytest.mark.api
    def test_health_check_model_not_ready(self, client):
        """Test health check when model is not ready"""
        with patch("app.main.predictor") as mock_predictor:
            mock_predictor.model.is_ready.return_value = False

            response = client.get("/health")
            assert response.status_code == 503
            data = response.json()
            assert data["detail"] == "Model not ready"

    @pytest.mark.unit
    @pytest.mark.api
    def test_predict_endpoint_success(self, client, sample_prediction_request):
        """Test successful prediction endpoint"""
        with patch("app.main.predictor") as mock_predictor:
            mock_predictor.predict.return_value = {
                "is_backchannel": True,
                "confidence": 0.85,
                "model_used": "test-model",
                "latency_ms": 25.0,
                "metadata": {"test": "data"},
            }
            mock_predictor.model.model_name = "test-model"

            response = client.post("/api/v1/predict", json=sample_prediction_request)
            assert response.status_code == 200
            data = response.json()
            assert data["is_backchannel"] is True
            assert data["confidence"] == 0.85
            assert data["model_used"] == "test-model"
            assert data["latency_ms"] == 25.0

    @pytest.mark.unit
    @pytest.mark.api
    def test_predict_endpoint_model_unavailable(
        self, client, sample_prediction_request
    ):
        """Test prediction endpoint when model is unavailable"""
        with patch("app.main.predictor", None):
            response = client.post("/api/v1/predict", json=sample_prediction_request)
            assert response.status_code == 503

    @pytest.mark.unit
    @pytest.mark.api
    def test_batch_predict_endpoint_success(self, client, sample_batch_request):
        """Test successful batch prediction endpoint"""
        with patch("app.main.predictor") as mock_predictor:
            mock_predictor.predict.return_value = {
                "is_backchannel": True,
                "confidence": 0.85,
                "model_used": "test-model",
                "latency_ms": 25.0,
            }
            mock_predictor.model.model_name = "test-model"

            response = client.post("/api/v1/predict/batch", json=sample_batch_request)
            assert response.status_code == 200
            data = response.json()
            assert "predictions" in data
            assert len(data["predictions"]) == 2
