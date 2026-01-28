"""
OpenTinker Training API Gateway.

FastAPI service that receives training requests from Argo Workflows (us-central1)
and queues them to RabbitMQ for processing by workers in us-east4-a.
"""
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from celery import Celery
from models import TrainingRequest, TrainingResponse
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="OpenTinker Training API",
    description="HTTP API for submitting cross-region GPU training jobs",
    version="1.0.0"
)

# Security
security = HTTPBearer()

# Celery configuration
CELERY_BROKER = os.getenv("CELERY_BROKER_URL", "amqp://admin:admin@rabbitmq:5672//")
celery_app = Celery("opentinker", broker=CELERY_BROKER)

# API key authentication
API_KEY = os.getenv("API_KEY", "opentinker-api-key-2025")


def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Verify API key from Bearer token."""
    if credentials.credentials != API_KEY:
        logger.warning(f"Invalid API key attempt: {credentials.credentials[:10]}...")
        raise HTTPException(status_code=403, detail="Invalid API key")
    return credentials.credentials


@app.get("/health")
async def health():
    """
    Public health check endpoint.

    Verifies:
    - FastAPI service is running
    - RabbitMQ connectivity (Celery broker)

    Returns:
        dict: Health status with Celery connection state
    """
    try:
        # Test Celery/RabbitMQ connectivity
        celery_app.control.inspect().ping()
        return {
            "status": "healthy",
            "celery": "connected",
            "broker": CELERY_BROKER.split("@")[1] if "@" in CELERY_BROKER else "unknown"
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "degraded",
            "celery": "disconnected",
            "error": str(e)
        }


@app.post("/train", response_model=TrainingResponse, status_code=202)
async def submit_training(
    request: TrainingRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Submit training job to GPU cluster queue.

    The job will be:
    1. Enqueued to RabbitMQ (us-central1)
    2. Consumed by Celery worker (us-east4-a)
    3. Submitted to OpenTinker scheduler for GPU allocation
    4. Trained on H100 GPUs with latest image

    Args:
        request: Training job configuration
        api_key: Bearer token for authentication

    Returns:
        TrainingResponse: Job ID and queued status

    Raises:
        HTTPException: If job submission fails
    """
    try:
        logger.info(f"Submitting training job: task={request.task}, gpus={request.num_gpus}")

        # Send task to Celery queue
        task = celery_app.send_task(
            'tasks.run_training',
            args=[request.dict()],
            queue='opentinker-training'
        )

        logger.info(f"Job queued successfully: job_id={task.id}")

        return TrainingResponse(
            status="queued",
            job_id=task.id,
            message=f"Training job queued for H100 GPU cluster in us-east4-a (task={request.task}, gpus={request.num_gpus})"
        )

    except Exception as e:
        logger.error(f"Failed to submit training job: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to queue training job: {str(e)}"
        )


@app.get("/status/{job_id}")
async def get_status(job_id: str, api_key: str = Depends(verify_api_key)):
    """
    Get training job status by ID.

    Args:
        job_id: Celery task ID from /train response
        api_key: Bearer token for authentication

    Returns:
        dict: Job status and result (if completed)
    """
    try:
        result = celery_app.AsyncResult(job_id)

        response = {
            "job_id": job_id,
            "status": result.state,
        }

        # Include result if task is ready
        if result.ready():
            response["result"] = result.result

        # Include task info if available
        if result.info:
            response["info"] = result.info

        return response

    except Exception as e:
        logger.error(f"Failed to get job status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve job status: {str(e)}"
        )


@app.get("/")
async def root():
    """API root endpoint with service information."""
    return {
        "service": "OpenTinker Training API",
        "version": "1.0.0",
        "endpoints": {
            "health": "GET /health",
            "submit_training": "POST /train",
            "get_status": "GET /status/{job_id}"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
