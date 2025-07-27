from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class PredictionResult:
    is_backchannel: bool
    confidence: float
    model_name: str
    metadata: Optional[Dict[str, Any]] = None


class BackchannelModel(ABC):
    """Abstract base class for backchannel detection models"""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the name/identifier of the model"""
        pass

    @abstractmethod
    def predict(
        self, text: str, previous_utterance: Optional[str] = None
    ) -> PredictionResult:
        """
        Predict if the given text is a backchannel

        Args:
            text: Input text to classify
            previous_utterance:
                Previous utterance (optional, for context-aware models)

        Returns:
            PredictionResult with prediction, confidence, and metadata
        """
        pass

    @abstractmethod
    def predict_batch(self, texts: list[str]) -> list[PredictionResult]:
        """
        Predict for multiple texts (for efficiency)

        Args:
            texts: List of input texts

        Returns:
            List of PredictionResult objects
        """
        pass

    @abstractmethod
    def is_ready(self) -> bool:
        """Check if model is loaded and ready for inference"""
        pass

    def preprocess(self, text: str) -> str:
        """Default preprocessing - can be overridden"""
        return text.strip().lower()
