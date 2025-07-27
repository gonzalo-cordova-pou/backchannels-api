from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    # Model selection
    model_type: str = "auto"  # "auto", "baseline"

    # Model configuration
    model_threshold: float = (
        0.9  # Threshold for backchannel classification (0.0 to 1.0)
    )

    # API configuration
    max_batch_size: int = 32

    # Logging
    log_level: str = "INFO"

    # S3 configuration (for future production use)
    s3_bucket_name: str = "backchannels-models"
    s3_model_prefix: str = "models"
    s3_region: str = "us-east-1"
    s3_access_key_id: str = ""
    s3_secret_access_key: str = ""

    class Config:
        env_file = ".env"
        extra = "ignore"  # Ignore extra environment variables


settings = Settings()
