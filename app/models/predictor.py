import time
from typing import Optional

from app.config import settings
from app.models.base import BackchannelModel
from app.models.factory import ModelFactory


class BackchannelPredictor:
    """Main predictor class that wraps any BackchannelModel implementation"""

    def __init__(self, model: Optional[BackchannelModel] = None):
        if model is None:
            # Use threshold from settings
            model = ModelFactory.create_default_model(
                threshold=settings.model_threshold
            )

        self.model = model

        if not self.model.is_ready():
            raise RuntimeError(f"Model {self.model.model_name} is not ready")

    def predict(self, text: str, previous_utterance: Optional[str] = None) -> dict:
        """Predict with timing and formatting for API response"""
        start_time = time.perf_counter()

        result = self.model.predict(text, previous_utterance)

        latency_ms = (time.perf_counter() - start_time) * 1000

        return {
            "is_backchannel": result.is_backchannel,
            "confidence": result.confidence,
            "model_used": result.model_name,
            "latency_ms": latency_ms,
            "metadata": result.metadata,
        }

    def predict_batch(self, texts: list[str]) -> list[dict]:
        """Batch prediction"""
        start_time = time.perf_counter()

        results = self.model.predict_batch(texts)

        total_latency_ms = (time.perf_counter() - start_time) * 1000
        avg_latency_ms = total_latency_ms / len(texts)

        return [
            {
                "is_backchannel": result.is_backchannel,
                "confidence": result.confidence,
                "model_used": result.model_name,
                "latency_ms": avg_latency_ms,
                "metadata": result.metadata,
            }
            for result in results
        ]

    @classmethod
    def from_config(cls, config: dict) -> "BackchannelPredictor":
        """Create predictor from configuration"""
        model = ModelFactory.create_model(config)
        return cls(model)
