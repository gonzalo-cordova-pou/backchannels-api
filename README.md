# Backchannel Detection API

A FastAPI-based service for detecting backchannels in conversational AI systems. This API provides low-latency backchannel detection using machine learning models.

## Table of Contents

- [Quick Start](#quick-start)
  - [Prerequisites](#prerequisites)
  - [Using Docker (Recommended)](#using-docker-recommended)
  - [Using Poetry](#using-poetry)
- [Features](#features)
- [API Usage Examples](#api-usage-examples)
  - [Single Prediction](#single-prediction)
  - [Batch Prediction](#batch-prediction)
  - [Get Model Information](#get-model-information)
- [Models](#models)
  - [Available Models](#available-models)
- [Configuration](#configuration)
  - [Environment Variables](#environment-variables)
- [API Endpoints](#api-endpoints)
- [Docker Deployment](#docker-deployment)
  - [Using Docker Compose (Recommended)](#using-docker-compose-recommended)
  - [Using Docker Directly](#using-docker-directly)
  - [Production Considerations](#production-considerations)
- [Model Requirements](#model-requirements)
- [Testing](#testing)
  - [Run Test Suite](#run-test-suite)
  - [Performance Testing Scripts](#performance-testing-scripts)
- [Performance](#performance)

## Quick Start

### Prerequisites

Before getting started, ensure you have the following installed:

- **Python 3.12+**
- **Poetry** (for dependency management)

#### Installing Poetry

```bash
# Update pip first
pip install --upgrade pip
```

```bash
pip install poetry
```

### Using Docker (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd backchannels-api

# Start the API with Docker Compose
docker-compose up --build

# Test the API
curl http://localhost:8000/health
```

### Using Poetry

```bash
# Clone the repository
git clone <repository-url>
cd backchannels-api

# Install dependencies
poetry install

# Start the server
poetry run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Features

- **DistilBERT Model**: Fine-tuned DistilBERT for backchannel detection
- **Baseline Model**: Simple keyword-based baseline model
- **RESTful API**: FastAPI-based endpoints for single and batch predictions
- **Context Awareness**: Support for previous utterance context
- **Performance Monitoring**: Built-in latency tracking and logging
- **Configurable Threshold**: Adjustable classification threshold for model sensitivity

## API Usage Examples

### Single Prediction

```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "utterance": "Yeah, I see what you mean",
    "previous_utterance": "The weather is really nice today"
  }'
```

**Response:**
```json
{
  "is_backchannel": true,
  "confidence": 0.92,
  "model_used": "distilbert-onnx",
  "latency_ms": 31.5,
  "metadata": null
}
```

### Batch Prediction

```bash
curl -X POST "http://localhost:8000/api/v1/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "utterances": [
      {
        "utterance": "Yeah, I see what you mean",
        "previous_utterance": "The weather is really nice today"
      },
      {
        "utterance": "What time is the meeting?",
        "previous_utterance": "We have a meeting tomorrow"
      }
    ]
  }'
```

### Get Model Information

```bash
curl http://localhost:8000/api/v1/model/info
```

**Response:**
```json
{
  "model_name": "distilbert-onnx",
  "is_ready": true,
  "model_type": "transformer",
  "threshold": 0.5
}
```

## Models

The API uses an abstract base class (`BackchannelModel`) that allows any model to be implemented by following the interface. All models must implement:

- `predict(text, previous_utterance=None)` - Single prediction
- `predict_batch(texts)` - Batch predictions
- `is_ready()` - Model readiness check
- `model_name` - Model identifier

### Available Models

#### DistilBERT ONNX (Default)
Optimized DistilBERT model using ONNX Runtime for high performance:
- **55-60% faster** than original PyTorch model
- **Configurable threshold** (0.0 to 1.0)
- **Context-aware** with previous utterance support
- **Low latency**: ~31ms average for 10 concurrent requests

```python
from app.models.distilbert_onnx import DistilBertOnnxModel

model = DistilBertOnnxModel(threshold=0.5)
result = model.predict("Yeah, I see what you mean", "The weather is really nice today")
```

#### DistilBERT (Legacy)
Original PyTorch-based DistilBERT model (slower but available for comparison).

#### Baseline Model
Simple keyword-based perfect match model for comparison and fallback.

## Configuration

### Environment Variables

The API supports configuration via environment variables:

```bash
# Model configuration
MODEL_THRESHOLD=0.9  # Classification threshold (0.0 to 1.0)

# API configuration
MAX_BATCH_SIZE=32

# Logging
LOG_LEVEL=INFO
```

## API Endpoints

- `POST /api/v1/predict` - Single prediction
- `POST /api/v1/predict/batch` - Batch predictions
- `GET /api/v1/model/info` - Model information (includes current threshold)
- `GET /health` - Health check

## Docker Deployment

### Using Docker Compose (Recommended)

```bash
# Start the API
docker-compose up -d

# View logs
docker-compose logs -f backchannels-api

# Stop the API
docker-compose down
```

### Using Docker Directly

```bash
# Build the image
docker build -t backchannels-api .

# Run the container
docker run -p 8000:8000 \
  -e MODEL_THRESHOLD=0.5 \
  -e LOG_LEVEL=INFO \
  backchannels-api
```

### Production Considerations

#### Resource Limits
For production deployments, consider adding resource limits to `docker-compose.yml`:

```yaml
services:
  backchannels-api:
    # ... existing config ...
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
        reservations:
          memory: 2G
          cpus: '1.0'
```

#### Model Weights Volume Mounting
If your model weights are large (>100MB), mount them as a volume:

```yaml
volumes:
  - ./app/models/weights:/app/app/models/weights
```

#### Health Checks
The container includes built-in health checks. Monitor with:

```bash
# Check container health
docker ps

# View logs
docker-compose logs -f backchannels-api
```



## Model Requirements

The API requires pre-trained model weights in the standard HuggingFace format:
- **DistilBERT**: `app/models/weights/` (PyTorch format)
- **DistilBERT ONNX**: `app/models/weights_onnx/` (ONNX format)

Both directories should contain: `config.json`, `model.onnx`/`pytorch_model.bin`, `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, and `vocab.txt`.

### S3 Model Loading (Production)
For production deployments, models can be loaded from S3:

```bash
# Future S3 configuration (not currently used)
export S3_BUCKET_NAME="your-models-bucket"
export S3_MODEL_PREFIX="models"
export S3_REGION="us-east-1"
export S3_ACCESS_KEY_ID="your-access-key"
export S3_SECRET_ACCESS_KEY="your-secret-key"
```

## Testing

### Run Test Suite

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=app --cov-report=html

# Run specific test categories
poetry run pytest -m unit      # Unit tests
poetry run pytest -m api       # API tests
poetry run pytest -m integration  # Integration tests
```

### Performance Testing Scripts

The project includes comprehensive inference testing scripts in the `scripts/` directory:

#### Simple Inference Test
```bash
# Simple model testing with sample inputs
python scripts/simple_inference.py

# Direct model performance analysis (bypasses HTTP layer)
python scripts/stress_test.py

# Full API stack testing (production-like with HTTP overhead)
python scripts/prod_stress_test.py
```

These scripts provide different levels of testing:
- **Simple Inference**: Tests model with sample inputs and provides detailed output
- **Stress Test**: Pure performance analysis with sequential, concurrent, and batch testing
- **Production Test**: Realistic HTTP testing with async requests and error handling



## Performance

#### Before ONNX (Original Model):
- **8 concurrent**: ~63ms average latency
- **16 concurrent**: ~123ms average latency

#### After ONNX Optimization:
- **8 concurrent**: 28.4ms average latency (**55% faster**)
- **16 concurrent**: 49.6ms average latency (**60% faster**)

### Key Metrics for Our Use Case

**Target Load (10 concurrent calls)**:
- **Average latency**: ~31ms (well under 50ms target)
- **P95 latency**: ~41ms
- **Throughput**: 314 req/s


#### Deployment Recommendation
Single instance is more than sufficient:
- Handles 10 concurrent requests at 31ms avg latency
- P99 latency (85ms) still reasonable for outliers

### Key Insights

#### 1. ONNX Benefits Realized
- **Faster inference**: Model optimization working perfectly
- **Better concurrency**: Less CPU contention per request
- **Consistent performance**: Lower variance in latency

#### 2. P99 Outliers to Monitor
- **16 concurrent**: P99 = 148ms (some requests still slow)
- Likely GC pauses or OS scheduling - monitor in production
