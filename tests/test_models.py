from unittest.mock import Mock, patch

import pytest

from app.models.base import BackchannelModel, PredictionResult
from app.models.distilbert_onnx import DistilBertOnnxModel
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
    def test_create_distilbert_onnx_model(self):
        """Test creating ONNX DistilBERT model"""
        with patch("app.models.factory.DistilBertOnnxModel") as mock_onnx_class:
            mock_model = Mock()
            mock_onnx_class.create_from_weights.return_value = mock_model

            model = ModelFactory.create_model({"type": "distilbert-onnx"})

            assert model == mock_model
            mock_onnx_class.create_from_weights.assert_called_once_with(threshold=0.5)

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


class TestDistilBertOnnxModel:
    """Test the ONNX DistilBERT model"""

    @pytest.fixture
    def mock_model(self):
        """Mock the ONNX model and tokenizer"""
        with patch("app.models.distilbert_onnx.AutoTokenizer") as mock_tokenizer, patch(
            "app.models.distilbert_onnx.ORTModelForSequenceClassification"
        ) as mock_ort_model, patch("pathlib.Path.exists") as mock_exists:

            # Mock path exists
            mock_exists.return_value = True

            # Mock tokenizer
            mock_tokenizer_instance = Mock()
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

            # Mock ONNX model
            mock_model_instance = Mock()
            mock_ort_model.from_pretrained.return_value = mock_model_instance

            # Mock tokenizer output
            mock_tokenizer_instance.return_value = {
                "input_ids": Mock(),
                "attention_mask": Mock(),
            }

            # Mock model output
            mock_logits = Mock()
            mock_logits.shape = (1, 2)
            mock_model_instance.return_value.logits = mock_logits

            yield DistilBertOnnxModel(model_path="test_path")

    def test_model_name(self, mock_model):
        """Test that the model name is correct"""
        assert mock_model.model_name == "distilbert-onnx-backchannel"

    def test_preprocess_with_context(self, mock_model):
        """Test preprocessing with previous utterance"""
        result = mock_model.preprocess("hello", "how are you")
        assert result == "how are you [SEP] hello"

    def test_preprocess_without_context(self, mock_model):
        """Test preprocessing without previous utterance"""
        result = mock_model.preprocess("hello")
        assert result == "hello"

    def test_is_ready(self, mock_model):
        """Test that the model reports ready state correctly"""
        assert mock_model.is_ready() is True

    def test_uses_onnx_model_type(self, mock_model):
        """Test that the model uses ONNX-specific components"""
        # Check that it uses AutoTokenizer instead of DistilBertTokenizer
        assert mock_model.tokenizer is not None
        # Check that it uses ORTModelForSequenceClassification
        assert mock_model.model is not None
