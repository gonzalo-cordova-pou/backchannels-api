import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException

from app.api.routes import router
from app.core.logging import setup_logging
from app.models.predictor import BackchannelPredictor

logger = logging.getLogger(__name__)

# Global predictor instance
predictor: Optional[BackchannelPredictor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    global predictor

    # Startup
    setup_logging()
    logger.info("Starting Backchannel Detection API")

    try:
        # Use ONNX DistilBERT model for better performance
        predictor = BackchannelPredictor.from_config({"type": "distilbert-onnx"})

        logger.info(f"Model loaded: {predictor.model.model_name}")

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down Backchannel Detection API")


app = FastAPI(
    title="Backchannel Detection API",
    version="1.0.0",
    description="Low-latency backchannel detection for voice AI agents",
    lifespan=lifespan,
)

app.include_router(router, prefix="/api/v1")


@app.get("/")
async def root():
    return {
        "message": "Backchannel Detection API",
        "model": predictor.model.model_name if predictor else "Not loaded",
        "status": "ready" if predictor and predictor.model.is_ready() else "not ready",
    }


@app.get("/health")
async def health_check():
    if not predictor or not predictor.model.is_ready():
        raise HTTPException(status_code=503, detail="Model not ready")

    return {"status": "healthy", "model": predictor.model.model_name}
