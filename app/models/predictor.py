import asyncio
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from app.models.base import BackchannelModel
from app.models.baseline import BaselineModel
from app.models.factory import ModelFactory

logger = logging.getLogger(__name__)


class BackchannelPredictor:
    """Main predictor class that wraps any BackchannelModel implementation"""

    def __init__(self, model: Optional[BackchannelModel] = None):
        if model is None:
            # Use baseline model as default if no model is provided
            model = BaselineModel.create_with_default_terms()

        self.model = model

        # Optimize thread pool for containerized environments
        cpu_count = os.cpu_count() or 4
        # Use conservative thread count for containers
        max_workers = min(cpu_count * 2, 8)  # Conservative for containers

        logger.info(
            f"Using {max_workers} thread pool workers for model inference "
            f"(CPU count: {cpu_count})"
        )
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

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

    async def predict_async(
        self, text: str, previous_utterance: Optional[str] = None
    ) -> dict:
        """Async version of predict that runs in thread pool"""
        start_time = time.perf_counter()

        # Run the model prediction in a thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self._executor, self.model.predict, text, previous_utterance
        )

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

    @classmethod
    def with_fallback(
        cls, preferred_config: dict, fallback_config: Optional[dict] = None
    ) -> "BackchannelPredictor":
        """
        Create predictor with fallback mechanism.

        Args:
            preferred_config: Configuration for the preferred model
            fallback_config: Configuration for the fallback model (defaults to baseline)

        Returns:
            BackchannelPredictor with the successfully loaded model
        """
        if fallback_config is None:
            fallback_config = {"type": "baseline"}

        try:
            # Try to load the preferred model
            preferred_type = preferred_config.get("type", "unknown")
            logger.info(f"Attempting to load preferred model: {preferred_type}")
            model = ModelFactory.create_model(preferred_config)

            # Check if the model is ready
            if model.is_ready():
                logger.info(f"Successfully loaded preferred model: {model.model_name}")
                return cls(model)
            else:
                raise RuntimeError(f"Preferred model {model.model_name} is not ready")

        except Exception as e:
            preferred_type = preferred_config.get("type", "unknown")
            logger.warning(f"Failed to load preferred model {preferred_type}: {e}")
            logger.info("Falling back to baseline model")

            try:
                # Try to load the fallback model
                fallback_model = ModelFactory.create_model(fallback_config)

                if fallback_model.is_ready():
                    logger.info(
                        "Successfully loaded"
                        f"fallback model: {fallback_model.model_name}"
                    )
                    return cls(fallback_model)
                else:
                    raise RuntimeError(
                        f"Fallback model {fallback_model.model_name} is not ready"
                    )

            except Exception as fallback_error:
                logger.error(
                    "Failed to load both preferred and "
                    f"fallback models: {fallback_error}"
                )
                raise RuntimeError(
                    f"Unable to load any model. Preferred model failed: {e}. "
                    f"Fallback model failed: {fallback_error}"
                )

    def __del__(self):
        """Cleanup thread pool executor"""
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=False)
