from typing import List, Optional

from .base import BackchannelModel, PredictionResult


class BaselineModel(BackchannelModel):
    """
    Perfect match baseline that checks if the ENTIRE text
    exactly equals one of the terms. Always case insensitive.
    """

    # Default terms for backchannel detection
    DEFAULT_TERMS = [
        "yeah",
        "yes",
        "uh-huh",
        "mhmm",
        "mm-hmm",
        "hmm",
        "oh",
        "ah",
        "uhhuh",
        "uh",
        "um",
        "mmmm",
        "yep",
        "wow",
        "right",
        "okay",
        "ok",
        "sure",
        "alright",
        "gotcha",
        "mmhmm",
        "great",
        "sweet",
        "ma'am",
        "awesome",
        "good morning",
        "i see",
        "got it",
        "that makes sense",
        "i hear you",
        "i understand",
        "good afternoon",
        "hey there",
        "perfect",
        "that's true",
        "good point",
        "exactly",
        "makes sense",
        "no problem",
        "indeed",
        "certainly",
        "very well",
        "absolutely",
        "correct",
        "of course",
        "hey",
        "hello",
        "hi",
        "yo",
    ]

    @property
    def model_name(self) -> str:
        """Return the name/identifier of the model"""
        return "perfect_match_baseline"

    @classmethod
    def create_with_default_terms(cls):
        """
        Create a baseline model with the default terms.

        Returns:
            BaselineModel: A new instance with default terms
        """
        return cls(cls.DEFAULT_TERMS)

    def __init__(self, terms: List[str]):
        """
        Initialize perfect match baseline.

        Args:
            terms: List of terms to match exactly (entire text
                    must equal one of these)
        """
        # Convert all terms to lowercase and store in set for O(1) lookup
        self.terms_set = set(term.lower().strip() for term in terms)

    def predict(
        self, text: str, previous_utterance: Optional[str] = None
    ) -> PredictionResult:
        """
        Predict if the given text is a backchannel.

        Args:
            text: Input text to classify
            previous_utterance: Previous utterance (ignored by baseline model)

        Returns:
            PredictionResult with prediction, confidence, and metadata
        """
        if not isinstance(text, str):
            raise ValueError("Input text must be a string")

        processed_text = self.preprocess(text)
        is_backchannel = processed_text in self.terms_set

        # For baseline model, confidence is 1.0 if it matches, 0.0 otherwise
        confidence = 1.0 if is_backchannel else 0.0

        return PredictionResult(
            is_backchannel=is_backchannel,
            confidence=confidence,
            model_name=self.model_name,
            metadata={"matched_term": processed_text if is_backchannel else None},
        )

    def predict_batch(self, texts: list[str]) -> list[PredictionResult]:
        """
        Predict for multiple texts (for efficiency).

        Args:
            texts: List of input texts

        Returns:
            List of PredictionResult objects
        """
        if not isinstance(texts, (list, tuple)):
            raise ValueError("Input texts must be a list or tuple")

        results = []
        for text in texts:
            if not isinstance(text, str):
                raise ValueError("All items in texts must be strings")
            results.append(self.predict(text))
        return results

    def is_ready(self) -> bool:
        """Check if model is loaded and ready for inference"""
        return hasattr(self, "terms_set") and len(self.terms_set) > 0
