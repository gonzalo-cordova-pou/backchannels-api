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

    class Config:
        env_file = ".env"
        extra = "ignore"  # Ignore extra environment variables


settings = Settings()
