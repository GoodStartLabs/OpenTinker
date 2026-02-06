"""
Celery application configuration for OpenTinker worker.

Configures message broker (RabbitMQ), serialization, and task routing.
"""
from celery import Celery
import os

# RabbitMQ broker URL
CELERY_BROKER = os.getenv(
    "CELERY_BROKER_URL",
    "amqp://admin:admin@rabbitmq:5672//"
)

# Initialize Celery app
app = Celery(
    "opentinker",
    broker=CELERY_BROKER,
    include=['tasks']  # Import tasks module
)

# Celery configuration
app.conf.update(
    # Serialization
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',

    # Timezone
    timezone='UTC',
    enable_utc=True,

    # Task execution
    task_track_started=True,
    task_acks_late=True,  # Acknowledge after task completes
    worker_prefetch_multiplier=1,  # Only fetch one task at a time

    # Timeouts (7 days for long training jobs)
    task_time_limit=604800,  # Hard limit: 7 days
    task_soft_time_limit=604600,  # Soft limit: 7 days - 200s

    # Results (use RPC backend via RabbitMQ)
    result_backend='rpc://',  # Use RabbitMQ for results
    result_expires=3600,  # Expire results after 1 hour

    # Task routing
    task_routes={
        'tasks.run_training': {'queue': 'opentinker-training'},
        'tasks.health_check': {'queue': 'opentinker-health'},
    },

    # Worker configuration
    worker_log_format='[%(asctime)s: %(levelname)s/%(processName)s] %(message)s',
    worker_task_log_format='[%(asctime)s: %(levelname)s/%(processName)s][%(task_name)s(%(task_id)s)] %(message)s',
)

if __name__ == '__main__':
    app.start()
