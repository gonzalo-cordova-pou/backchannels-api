import json
import logging
import logging.config
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

from app.config import settings


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add extra fields if present
        if hasattr(record, "request_id"):
            log_entry["request_id"] = record.request_id

        if hasattr(record, "model_name"):
            log_entry["model_name"] = record.model_name

        if hasattr(record, "latency_ms"):
            log_entry["latency_ms"] = record.latency_ms

        if hasattr(record, "prediction"):
            log_entry["prediction"] = record.prediction

        if hasattr(record, "user_id"):
            log_entry["user_id"] = record.user_id

        # Add exception info if present
        if record.exc_info:
            exc_type, exc_value, exc_traceback = record.exc_info
            log_entry["exception"] = {
                "type": exc_type.__name__ if exc_type else "Unknown",
                "message": str(exc_value) if exc_value else "Unknown error",
                "traceback": traceback.format_exception(*record.exc_info),
            }

        return json.dumps(log_entry)


class PerformanceFilter(logging.Filter):
    """Filter to log performance metrics"""

    def filter(self, record: logging.LogRecord) -> bool:
        # Only log performance records or warnings/errors
        return hasattr(record, "latency_ms") or record.levelno >= logging.WARNING


def setup_logging() -> None:
    """
    Setup logging configuration for the application
    """

    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Determine log level
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    # Base logging configuration
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "detailed": {
                "format": (
                    "%(asctime)s - %(name)s - %(levelname)s - "
                    "%(module)s:%(funcName)s:%(lineno)d - %(message)s"
                ),
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "simple": {"format": "%(levelname)s - %(message)s"},
            "json": {
                "()": JSONFormatter,
            },
        },
        "filters": {
            "performance": {
                "()": PerformanceFilter,
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": log_level,
                "formatter": "detailed",
                "stream": sys.stdout,
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": logging.INFO,
                "formatter": "json",
                "filename": "logs/backchannel_api.log",
                "maxBytes": 50_000_000,  # 50MB
                "backupCount": 5,
                "encoding": "utf8",
            },
            "error_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": logging.ERROR,
                "formatter": "json",
                "filename": "logs/errors.log",
                "maxBytes": 10_000_000,  # 10MB
                "backupCount": 3,
                "encoding": "utf8",
            },
            "performance_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": logging.INFO,
                "formatter": "json",
                "filename": "logs/performance.log",
                "maxBytes": 20_000_000,  # 20MB
                "backupCount": 3,
                "encoding": "utf8",
                "filters": ["performance"],
            },
        },
        "loggers": {
            # Root logger
            "": {"level": log_level, "handlers": ["console", "file", "error_file"]},
            # App loggers
            "app": {
                "level": log_level,
                "handlers": ["console", "file", "error_file"],
                "propagate": False,
            },
            "app.models": {
                "level": log_level,
                "handlers": ["console", "file", "performance_file"],
                "propagate": False,
            },
            "app.api": {
                "level": log_level,
                "handlers": ["console", "file", "performance_file"],
                "propagate": False,
            },
            # Third-party loggers
            "uvicorn": {"level": logging.INFO, "handlers": ["console", "file"]},
            "uvicorn.access": {
                "level": logging.INFO,
                "handlers": ["file"],
                "propagate": False,
            },
            "fastapi": {"level": logging.INFO, "handlers": ["console", "file"]},
            # Suppress noisy loggers
            "httpx": {"level": logging.WARNING},
            "urllib3": {"level": logging.WARNING},
        },
    }

    # Apply configuration
    logging.config.dictConfig(config)

    # Log startup message
    logger = logging.getLogger("app")
    logger.info(
        "Logging initialized",
        extra={"log_level": settings.log_level, "log_dir": str(log_dir.absolute())},
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name

    Args:
        name: Logger name (usually __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


class LoggerMixin:
    """Mixin class to add logging capabilities to any class"""

    @property
    def logger(self) -> logging.Logger:
        return get_logger(self.__class__.__module__ + "." + self.__class__.__name__)


# Context manager for request logging
class RequestLogger:
    """Context manager for request-level logging with metrics"""

    def __init__(
        self, request_id: str, endpoint: str, model_name: Optional[str] = None
    ):
        self.request_id = request_id
        self.endpoint = endpoint
        self.model_name = model_name
        self.logger = get_logger("app.api.requests")
        self.start_time = None

    def __enter__(self):
        import time

        self.start_time = time.perf_counter()
        self.logger.info(
            f"Request started: {self.endpoint}",
            extra={
                "request_id": self.request_id,
                "endpoint": self.endpoint,
                "model_name": self.model_name,
            },
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import time

        latency_ms = (time.perf_counter() - self.start_time) * 1000

        if exc_type is None:
            self.logger.info(
                f"Request completed: {self.endpoint}",
                extra={
                    "request_id": self.request_id,
                    "endpoint": self.endpoint,
                    "model_name": self.model_name,
                    "latency_ms": latency_ms,
                    "status": "success",
                },
            )
        else:
            self.logger.error(
                f"Request failed: {self.endpoint}",
                extra={
                    "request_id": self.request_id,
                    "endpoint": self.endpoint,
                    "model_name": self.model_name,
                    "latency_ms": latency_ms,
                    "status": "error",
                    "error_type": exc_type.__name__ if exc_type else None,
                    "error_message": str(exc_val) if exc_val else None,
                },
                exc_info=True,
            )


def log_prediction(
    logger: logging.Logger,
    text: str,
    prediction: bool,
    confidence: float,
    model_name: str,
    latency_ms: float,
    request_id: Optional[str] = None,
) -> None:
    """
    Log a prediction with structured data

    Args:
        logger: Logger instance
        text: Input text
        prediction: Prediction result
        confidence: Confidence score
        model_name: Name of the model used
        latency_ms: Prediction latency in milliseconds
        request_id: Optional request ID
    """
    logger.info(
        f"Prediction: '{text[:50]}...' -> {prediction}",
        extra={
            "request_id": request_id,
            "prediction": prediction,
            "confidence": confidence,
            "model_name": model_name,
            "latency_ms": latency_ms,
            "input_length": len(text),
            "input_preview": text[:100],  # First 100 chars for debugging
        },
    )


def log_model_load(
    logger: logging.Logger,
    model_name: str,
    model_type: str,
    load_time_ms: float,
    success: bool = True,
    error: Optional[str] = None,
) -> None:
    """
    Log model loading events

    Args:
        logger: Logger instance
        model_name: Name of the model
        model_type: Type of the model
        load_time_ms: Time taken to load the model
        success: Whether loading was successful
        error: Error message if loading failed
    """
    if success:
        logger.info(
            f"Model loaded successfully: {model_name}",
            extra={
                "model_name": model_name,
                "model_type": model_type,
                "load_time_ms": load_time_ms,
                "status": "loaded",
            },
        )
    else:
        logger.error(
            f"Model loading failed: {model_name}",
            extra={
                "model_name": model_name,
                "model_type": model_type,
                "load_time_ms": load_time_ms,
                "status": "failed",
                "error": error,
            },
        )


# Example usage functions for different log levels
def log_performance_warning(
    logger: logging.Logger, operation: str, latency_ms: float, threshold_ms: float = 50
):
    """Log performance warnings when latency exceeds threshold"""
    if latency_ms > threshold_ms:
        logger.warning(
            f"Performance warning: {operation} took {latency_ms:.1f}ms "
            f"(threshold: {threshold_ms}ms)",
            extra={
                "operation": operation,
                "latency_ms": latency_ms,
                "threshold_ms": threshold_ms,
                "performance_issue": True,
            },
        )


def log_batch_metrics(
    logger: logging.Logger,
    batch_size: int,
    total_latency_ms: float,
    model_name: str,
    request_id: Optional[str] = None,
):
    """Log batch processing metrics"""
    avg_latency = total_latency_ms / batch_size if batch_size > 0 else 0

    logger.info(
        f"Batch processed: {batch_size} items in {total_latency_ms:.1f}ms",
        extra={
            "request_id": request_id,
            "batch_size": batch_size,
            "total_latency_ms": total_latency_ms,
            "avg_latency_ms": avg_latency,
            "model_name": model_name,
            "throughput_per_second": (
                (batch_size / total_latency_ms * 1000) if total_latency_ms > 0 else 0
            ),
        },
    )
