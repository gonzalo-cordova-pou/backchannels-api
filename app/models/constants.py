import os
from pathlib import Path

# Model weights directories
DISTILBERT_WEIGHTS_DIR = "app/models/weights"
DISTILBERT_ONNX_WEIGHTS_DIR = "app/models/weights_onnx"

# S3 configuration (for future production use)
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "backchannels-models")
S3_MODEL_PREFIX = os.getenv("S3_MODEL_PREFIX", "models")

# Model file names (assuming standard HuggingFace format)
MODEL_FILES = [
    "config.json",
    "pytorch_model.bin",  # For regular DistilBERT
    "model.onnx",  # For ONNX model
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
]

# Validation functions


def validate_weights_directory(weights_dir: str) -> bool:
    """
    Validate that a weights directory contains the necessary model files

    Args:
        weights_dir: Path to the weights directory

    Returns:
        True if directory contains required files, False otherwise
    """
    weights_path = Path(weights_dir)

    if not weights_path.exists():
        return False

    # Check for essential files (at least config and tokenizer)
    essential_files = ["config.json", "tokenizer.json"]
    for file in essential_files:
        if not (weights_path / file).exists():
            return False

    return True


def get_model_path(model_type: str) -> str:
    """
    Get the appropriate weights directory for a given model type

    Args:
        model_type: Type of model ("distilbert" or "distilbert-onnx")

    Returns:
        Path to the weights directory
    """
    if model_type == "distilbert":
        return DISTILBERT_WEIGHTS_DIR
    elif model_type == "distilbert-onnx":
        return DISTILBERT_ONNX_WEIGHTS_DIR
    else:
        raise ValueError(f"Unknown model type: {model_type}")
