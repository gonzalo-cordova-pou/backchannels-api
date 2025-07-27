import time
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request

from app.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)
from app.config import settings
from app.core.logging import (
    RequestLogger,
    get_logger,
    log_batch_metrics,
    log_performance_warning,
    log_prediction,
)
from app.models.predictor import BackchannelPredictor

logger = get_logger(__name__)
router = APIRouter()


def get_predictor() -> BackchannelPredictor:
    """Dependency to get the global predictor instance"""
    from app.main import predictor

    if not predictor:
        logger.error("Predictor not available")
        raise HTTPException(status_code=503, detail="Model not loaded")
    return predictor


def generate_request_id() -> str:
    """Generate unique request ID for tracking"""
    return str(uuid.uuid4())[:8]


@router.post("/predict", response_model=PredictionResponse)
async def predict_backchannel(
    request: PredictionRequest,
    http_request: Request,
    predictor: BackchannelPredictor = Depends(get_predictor),  # noqa: B008
):
    """Predict if a single text is a backchannel"""
    request_id = generate_request_id()

    with RequestLogger(request_id, "POST /predict", predictor.model.model_name):
        try:
            # Log the incoming request
            logger.info(
                "Prediction request received",
                extra={
                    "request_id": request_id,
                    "input_length": len(request.utterance),
                    "client_ip": (
                        http_request.client.host if http_request.client else "unknown"
                    ),
                    "user_agent": http_request.headers.get("user-agent", "unknown"),
                },
            )

            # Make prediction
            result = predictor.predict(request.utterance, request.previous_utterance)

            # Log the prediction details
            log_prediction(
                logger=logger,
                text=request.utterance,
                prediction=result["is_backchannel"],
                confidence=result["confidence"],
                model_name=result["model_used"],
                latency_ms=result["latency_ms"],
                request_id=request_id,
            )

            # Check for performance issues
            log_performance_warning(
                logger=logger,
                operation="single_prediction",
                latency_ms=result["latency_ms"],
                threshold_ms=50,  # Your target latency
            )

            return PredictionResponse(**result)

        except Exception as e:
            logger.error(
                f"Prediction failed for request {request_id}",
                extra={
                    "request_id": request_id,
                    "error_type": type(e).__name__,
                    "input_text": request.utterance[
                        :100
                    ],  # Log partial text for debugging
                },
                exc_info=True,
            )
            raise HTTPException(status_code=500, detail="Prediction failed")


@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_backchannel_batch(
    request: BatchPredictionRequest,
    http_request: Request,
    predictor: BackchannelPredictor = Depends(get_predictor),  # noqa: B008
):
    """Predict for multiple texts"""
    request_id = generate_request_id()

    # Validation
    if len(request.utterances) > settings.max_batch_size:
        logger.warning(
            f"Batch size too large: {len(request.utterances)}",
            extra={
                "request_id": request_id,
                "requested_size": len(request.utterances),
                "max_size": settings.max_batch_size,
            },
        )
        raise HTTPException(
            status_code=400,
            detail=f"Batch size too large. Maximum: {settings.max_batch_size}",
        )

    with RequestLogger(request_id, "POST /predict/batch", predictor.model.model_name):
        try:
            logger.info(
                "Batch prediction request received",
                extra={
                    "request_id": request_id,
                    "batch_size": len(request.utterances),
                    "client_ip": (
                        http_request.client.host if http_request.client else "unknown"
                    ),
                },
            )

            # Extract utterances and previous utterances from the request objects
            utterances = []
            previous_utterances = []
            for req in request.utterances:
                utterances.append(req.utterance)
                previous_utterances.append(req.previous_utterance)

            start_time = time.perf_counter()
            # For now, we'll process each prediction individually to handle
            # previous_utterance. This could be optimized for true batch processing
            # in the future
            results = []
            for utterance, prev_utterance in zip(utterances, previous_utterances):
                result = predictor.predict(utterance, prev_utterance)
                results.append(result)
            total_latency_ms = (time.perf_counter() - start_time) * 1000

            # Log batch metrics
            log_batch_metrics(
                logger=logger,
                batch_size=len(request.utterances),
                total_latency_ms=total_latency_ms,
                model_name=predictor.model.model_name,
                request_id=request_id,
            )

            return BatchPredictionResponse(predictions=results)

        except Exception as e:
            logger.error(
                f"Batch prediction failed for request {request_id}",
                extra={
                    "request_id": request_id,
                    "batch_size": len(request.utterances),
                    "error_type": type(e).__name__,
                },
                exc_info=True,
            )
            raise HTTPException(status_code=500, detail="Batch prediction failed")


@router.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info(
    predictor: BackchannelPredictor = Depends(get_predictor),  # noqa: B008
):
    """Get information about the current model"""
    logger.info(
        "Model info requested",
        extra={
            "model_name": predictor.model.model_name,
            "model_type": type(predictor.model).__name__,
            "is_ready": predictor.model.is_ready(),
        },
    )

    # Get threshold from model if available, otherwise use settings default
    threshold = getattr(predictor.model, "threshold", 0.5)

    return ModelInfoResponse(
        model_name=predictor.model.model_name,
        is_ready=predictor.model.is_ready(),
        model_type=type(predictor.model).__name__,
        threshold=threshold,
    )
