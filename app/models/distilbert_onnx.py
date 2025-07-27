import time
from pathlib import Path
from typing import List, Optional

import torch
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

from app.models.base import BackchannelModel, PredictionResult
from app.models.constants import DISTILBERT_ONNX_WEIGHTS_DIR, validate_weights_directory
from app.models.s3_loader import S3ModelLoader


class DistilBertOnnxModel(BackchannelModel, S3ModelLoader):
    """ONNX-based DistilBERT backchannel detection model"""

    def __init__(
        self, model_path: str = DISTILBERT_ONNX_WEIGHTS_DIR, threshold: float = 0.5
    ):
        """
        Initialize the ONNX DistilBERT model

        Args:
            model_path: Path to the ONNX model weights and tokenizer
            threshold: Classification threshold (0.0 to 1.0). Defaults to 0.5
        """
        self.model_path = Path(model_path)
        self.model: Optional[ORTModelForSequenceClassification] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = max(0.0, min(1.0, threshold))  # Clip between 0.0 and 1.0
        super().__init__()  # Initialize S3ModelLoader
        self._load_model()

    def _load_model(self):
        """Load the ONNX model and tokenizer from disk"""
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

            # Load ONNX model
            self.model = ORTModelForSequenceClassification.from_pretrained(
                self.model_path
            )

        except Exception as e:
            raise RuntimeError(
                f"Failed to load ONNX DistilBERT model from {self.model_path}: {e}"
            )

    def load_from_s3(self, model_type: str = "distilbert-onnx") -> bool:
        """
        Load model from S3 if local weights are not available

        Args:
            model_type: Type of model for S3 key prefix

        Returns:
            True if model was successfully loaded from S3, False otherwise
        """
        # Check if local weights exist and are valid
        if validate_weights_directory(self.model_path):
            return True

        # Try to download from S3
        if self.download_model_from_s3(model_type, self.model_path):
            # Reload the model after downloading
            self._load_model()
            return True

        return False

    @property
    def model_name(self) -> str:
        """Return the name/identifier of the model"""
        return "distilbert-onnx-backchannel"

    def preprocess(self, text: str, previous_utterance: Optional[str] = None) -> str:
        """
        Preprocess the input text according to the specified format

        Args:
            text: The utterance to classify
            previous_utterance: Previous utterance for context

        Returns:
            Preprocessed text in the format: previous_utterance + "[SEP]" + utterance
        """
        # Apply lowercase preprocessing
        text = text.strip().lower()

        if previous_utterance:
            previous_utterance = previous_utterance.strip().lower()
            # Combine with SEP token
            return f"{previous_utterance} [SEP] {text}"
        else:
            return text

    def predict(
        self, text: str, previous_utterance: Optional[str] = None
    ) -> PredictionResult:
        """
        Predict if the given text is a backchannel

        Args:
            text: Input text to classify
            previous_utterance: Previous utterance (optional, for context-aware models)

        Returns:
            PredictionResult with prediction, confidence, and metadata
        """
        if not self.is_ready():
            raise RuntimeError("Model is not loaded and ready for inference")

        start_time = time.time()

        # Preprocess the input
        processed_text = self.preprocess(text, previous_utterance)

        # Tokenize
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer is not loaded")

        inputs = self.tokenizer(
            processed_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )

        # Get prediction using ONNX model
        with torch.no_grad():
            if self.model is None:
                raise RuntimeError("Model is not loaded")

            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)

            # Assuming binary classification: [not_backchannel, backchannel]
            # Adjust indices based on your model's training setup
            backchannel_prob = probabilities[0][1].item()  # Probability of backchannel
            is_backchannel = backchannel_prob > self.threshold

            confidence = backchannel_prob if is_backchannel else 1 - backchannel_prob

        latency_ms = (time.time() - start_time) * 1000

        metadata = {
            "latency_ms": latency_ms,
            "processed_text": processed_text,
            "raw_probabilities": probabilities[0].tolist(),
            "threshold_used": self.threshold,
            "model_type": "onnx",
        }

        return PredictionResult(
            is_backchannel=is_backchannel,
            confidence=confidence,
            model_name=self.model_name,
            metadata=metadata,
        )

    def predict_batch(self, texts: List[str]) -> List[PredictionResult]:
        """
        Predict for multiple texts (for efficiency)

        Args:
            texts: List of input texts

        Returns:
            List of PredictionResult objects
        """
        if not self.is_ready():
            raise RuntimeError("Model is not loaded and ready for inference")

        results = []

        # Process each text individually for now
        # Could be optimized for true batch processing if needed
        for text in texts:
            result = self.predict(text)
            results.append(result)

        return results

    def is_ready(self) -> bool:
        """Check if model is loaded and ready for inference"""
        return (
            self.model is not None
            and self.tokenizer is not None
            and self.model_path.exists()
        )

    @classmethod
    def create_from_weights(
        cls, model_path: str = "app/models/weights_onnx", threshold: float = 0.5
    ) -> "DistilBertOnnxModel":
        """
        Create an ONNX DistilBERT model instance from the weights directory

        Args:
            model_path: Path to the ONNX model weights
            threshold: Classification threshold (0.0 to 1.0). Defaults to 0.5

        Returns:
            DistilBertOnnxModel instance
        """
        return cls(model_path, threshold)
