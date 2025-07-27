from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    utterance: str = Field(
        ..., min_length=1, max_length=500, description="Utterance to classify"
    )
    previous_utterance: Optional[str] = Field(
        None, max_length=500, description="Previous utterance"
    )


class PredictionResponse(BaseModel):
    is_backchannel: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    model_used: str
    latency_ms: float
    metadata: Optional[Dict[str, Any]] = None


class BatchPredictionRequest(BaseModel):
    utterances: List[PredictionRequest]


class BatchPredictionResponse(BaseModel):
    predictions: List[Dict[str, Any]]


class ModelInfoResponse(BaseModel):
    model_name: str
    is_ready: bool
    model_type: str
    threshold: float = Field(
        ..., ge=0.0, le=1.0, description="Classification threshold"
    )
