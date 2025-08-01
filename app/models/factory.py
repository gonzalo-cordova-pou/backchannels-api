from typing import Any, Dict

from app.models.base import BackchannelModel
from app.models.baseline import BaselineModel
from app.models.distilbert import DistilBertModel
from app.models.distilbert_onnx import DistilBertOnnxModel


class ModelFactory:
    """Factory for creating different model instances"""

    @staticmethod
    def create_model(model_config: Dict[str, Any]) -> BackchannelModel:
        """
        Create a model instance based on configuration

        Args:
            model_config: Dictionary with model type and parameters

        Returns:
            Instance of BackchannelModel
        """
        model_type = model_config.get("type")
        threshold = model_config.get("threshold", 0.5)

        if model_type == "baseline":
            return BaselineModel.create_with_default_terms()
        elif model_type == "distilbert":
            return DistilBertModel.create_from_weights(threshold=threshold)
        elif model_type == "distilbert-onnx":
            return DistilBertOnnxModel.create_from_weights(threshold=threshold)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
