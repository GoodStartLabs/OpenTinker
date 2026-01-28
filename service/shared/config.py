"""
Shared configuration for OpenTinker service components.

Environment variable configuration and constants shared across
API gateway and worker services.
"""
import os
from typing import Optional


class Config:
    """Shared configuration for OpenTinker services."""

    # RabbitMQ / Celery Broker
    CELERY_BROKER_URL: str = os.getenv(
        "CELERY_BROKER_URL",
        "amqp://admin:admin@rabbitmq:5672//"
    )

    # API Configuration
    API_KEY: str = os.getenv("API_KEY", "opentinker-api-key-2025")
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))

    # Scheduler Configuration (us-east4-a)
    SCHEDULER_URL: str = os.getenv(
        "SCHEDULER_URL",
        "http://opentinker-scheduler:8780"
    )

    # Worker Configuration
    POLL_INTERVAL: int = int(os.getenv("POLL_INTERVAL", "30"))  # seconds
    MAX_TRAINING_TIME: int = int(os.getenv("MAX_TRAINING_TIME", "7200"))  # 2 hours

    # Docker Image Registry
    IMAGE_REGISTRY: str = os.getenv(
        "IMAGE_REGISTRY",
        "us-central1-docker.pkg.dev/development-472321/opentinker"
    )
    TRAINING_IMAGE: str = os.getenv(
        "TRAINING_IMAGE",
        "gsl-opentinker_v1:latest"
    )

    # GCP Configuration
    GCP_PROJECT: str = os.getenv("GCP_PROJECT", "development-472321")
    GCP_REGION_CLUSTER: str = os.getenv("GCP_REGION_CLUSTER", "us-east4-a")

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    @classmethod
    def get_training_image_url(cls) -> str:
        """Get full training image URL."""
        return f"{cls.IMAGE_REGISTRY}/{cls.TRAINING_IMAGE}"

    @classmethod
    def is_production(cls) -> bool:
        """Check if running in production environment."""
        return os.getenv("ENV", "dev").lower() == "production"

    @classmethod
    def get_celery_queue_name(cls) -> str:
        """Get Celery queue name for training tasks."""
        return "opentinker-training"


# Export config instance
config = Config()
