from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models.predictor import BackchannelPredictor


@pytest.fixture
def client():
    """Test client for FastAPI app"""
    return TestClient(app)


@pytest.fixture
def mock_predictor():
    """Mock predictor for testing"""
    mock = Mock(spec=BackchannelPredictor)
    mock.model = Mock()
    mock.model.model_name = "test-model"
    mock.model.is_ready.return_value = True
    mock.predict.return_value = {
        "is_backchannel": True,
        "confidence": 0.85,
        "model_used": "test-model",
        "latency_ms": 25.0,
        "metadata": {"test": "data"},
    }
    return mock


@pytest.fixture
def sample_prediction_request():
    """Sample prediction request data"""
    return {
        "utterance": "Yeah, that's right",
        "previous_utterance": "The meeting is at 3 PM",
    }


@pytest.fixture
def sample_batch_request():
    """Sample batch prediction request data"""
    return {
        "utterances": [
            {
                "utterance": "Yeah, that's right",
                "previous_utterance": "The meeting is at 3 PM",
            },
            {"utterance": "I see", "previous_utterance": "The project is delayed"},
        ]
    }


@pytest.fixture
def mock_model():
    """Mock model for testing"""
    mock = Mock()
    mock.model_name = "test-model"
    mock.is_ready.return_value = True
    mock.predict.return_value = {
        "is_backchannel": True,
        "confidence": 0.85,
        "model_used": "test-model",
        "latency_ms": 25.0,
    }
    return mock
