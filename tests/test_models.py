from unittest.mock import Mock, patch

import pytest

from app.models.base import BackchannelModel, PredictionResult
from app.models.factory import ModelFactory
from app.models.predictor import BackchannelPredictor


class TestBackchannelPredictor:
    """Unit tests for BackchannelPredictor"""

    @pytest.mark.unit
    def test_predict_single_utterance(self):
        """Test prediction with single utterance"""
        mock_model = Mock(spec=BackchannelModel)
        mock_model.model_name = "test-model"
        mock_model.predict.return_value = PredictionResult(
            is_backchannel=True,
            confidence=0.85,
            model_name="test-model",
            metadata={"test": "data"},
        )

        predictor = BackchannelPredictor(mock_model)
        result = predictor.predict("Yeah, that's right")

        assert result["is_backchannel"] is True
        assert result["confidence"] == 0.85
        assert result["model_used"] == "test-model"
        assert "latency_ms" in result
        mock_model.predict.assert_called_once()

    @pytest.mark.unit
    def test_predict_measures_latency(self):
        """Test that prediction measures latency"""
        mock_model = Mock(spec=BackchannelModel)
        mock_model.model_name = "test-model"
        mock_model.predict.return_value = PredictionResult(
            is_backchannel=True,
            confidence=0.85,
            model_name="test-model",
            metadata={"test": "data"},
        )

        predictor = BackchannelPredictor(mock_model)
        result = predictor.predict("test utterance")

        assert "latency_ms" in result
        assert isinstance(result["latency_ms"], (int, float))
        assert result["latency_ms"] >= 0

    @pytest.mark.unit
    def test_from_config_method(self):
        """Test predictor creation from config"""
        mock_model = Mock(spec=BackchannelModel)
        mock_model.model_name = "test-model"

        with patch("app.models.factory.ModelFactory.create_model") as mock_factory:
            mock_factory.return_value = mock_model

            predictor = BackchannelPredictor.from_config({"type": "distilbert"})

            assert predictor.model == mock_model
            mock_factory.assert_called_once_with({"type": "distilbert"})


class TestModelFactory:
    """Unit tests for ModelFactory"""

    @pytest.mark.unit
    def test_create_distilbert_model(self):
        """Test creating DistilBERT model"""
        with patch("app.models.factory.DistilBertModel") as mock_distilbert_class:
            mock_model = Mock()
            mock_distilbert_class.create_from_weights.return_value = mock_model

            model = ModelFactory.create_model({"type": "distilbert"})

            assert model == mock_model
            mock_distilbert_class.create_from_weights.assert_called_once_with(
                threshold=0.5
            )

    @pytest.mark.unit
    def test_create_baseline_model(self):
        """Test creating baseline model"""
        with patch("app.models.factory.BaselineModel") as mock_baseline_class:
            mock_model = Mock()
            mock_baseline_class.create_with_default_terms.return_value = mock_model

            model = ModelFactory.create_model({"type": "baseline"})

            assert model == mock_model
            mock_baseline_class.create_with_default_terms.assert_called_once()

    @pytest.mark.unit
    def test_create_model_invalid_type(self):
        """Test creating model with invalid type"""
        with pytest.raises(ValueError, match="Unknown model type"):
            ModelFactory.create_model({"type": "invalid_model_type"})

    @pytest.mark.unit
    def test_create_model_missing_type(self):
        """Test creating model without type"""
        with pytest.raises(ValueError, match="Unknown model type: None"):
            ModelFactory.create_model({})
