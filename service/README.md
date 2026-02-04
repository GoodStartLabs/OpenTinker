# OpenTinker Service Layer

Cross-region GPU training service for OpenTinker. Enables Argo Workflows in us-central1 to trigger H100 GPU training jobs in us-east4-a.

## Architecture

```
Argo Workflows (us-central1)
    ↓ HTTP POST /train
FastAPI Gateway (api/)
    ↓ Enqueue to RabbitMQ
Celery Worker (worker/)
    ↓ Consumes queue
    ↓ Calls OpenTinker Scheduler API
OpenTinker Scheduler (us-east4-a)
    ↓ Allocates H100 GPUs
    ↓ Launches HTTP Training Server
Training (H100 GPUs)
```

## Directory Structure

```
OpenTinker/
├── Dockerfile        # Runtime image (volume mount based)
├── Dockerfile.worker # GPU worker image (self-contained, builds from root)
│
└── service/
    ├── api/              # FastAPI gateway (us-central1)
    │   ├── main.py       # FastAPI app with /health, /train, /status endpoints
    │   ├── models.py     # Pydantic request/response models
    │   ├── requirements.txt
    │   └── Dockerfile
    │
    ├── worker/           # Celery worker (us-east4-a)
    │   ├── tasks.py      # run_training task
    │   ├── scheduler_client.py  # OpenTinker scheduler HTTP client
    │   ├── celery_app.py # Celery configuration
    │   └── requirements.txt
    │
    └── shared/           # Shared utilities
        ├── config.py     # Environment configuration
        └── models.py     # Shared Pydantic models
```

## Components

### API Gateway (`api/`)

FastAPI service that receives training requests and queues them to RabbitMQ.

**Endpoints:**
- `GET /health` - Health check (public, no auth)
- `POST /train` - Submit training job (requires API key)
- `GET /status/{job_id}` - Get job status (requires API key)

**Authentication:**
- Bearer token authentication
- API key from `API_KEY` environment variable

**Build:**
```bash
cd api/
docker build -t us-central1-docker.pkg.dev/development-472321/opentinker/gsl-opentinker-api:latest .
docker push us-central1-docker.pkg.dev/development-472321/opentinker/gsl-opentinker-api:latest
```

### Worker (`worker/`)

Celery worker that orchestrates training jobs by calling OpenTinker scheduler.

**Tasks:**
- `run_training` - Main training orchestration task
- `health_check` - Worker health check

**Features:**
- Automatic retries on transient errors
- Job status polling (30s intervals)
- Timeout protection (2 hour max)

**Build:**
```bash
# Worker Dockerfile is at OpenTinker root (self-contained image)
cd ~/GSL/OpenTinker
docker build -f Dockerfile.worker -t us-central1-docker.pkg.dev/development-472321/opentinker/gsl-opentinker-worker:latest .
docker push us-central1-docker.pkg.dev/development-472321/opentinker/gsl-opentinker-worker:latest
```

### Shared (`shared/`)

Common configuration and models used by both API and worker.

- `config.py` - Environment variable configuration
- `models.py` - Pydantic models for training requests, job results, metrics

## Usage

### Submit Training Job

```bash
curl -X POST http://api.opentinker/train \
  -H "Authorization: Bearer opentinker-api-key-2025" \
  -H "Content-Type: application/json" \
  -d '{
    "task": "gomoku",
    "num_gpus": 4,
    "config": {
      "experiment_name": "gomoku_v1",
      "batch_size": 32,
      "num_epochs": 10
    }
  }'
```

**Response:**
```json
{
  "status": "queued",
  "job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "message": "Training job queued for H100 GPU cluster in us-east4-a (task=gomoku, gpus=4)"
}
```

### Check Job Status

```bash
curl -X GET http://api.opentinker/status/a1b2c3d4-e5f6-7890-abcd-ef1234567890 \
  -H "Authorization: Bearer opentinker-api-key-2025"
```

**Response:**
```json
{
  "job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "RUNNING",
  "info": {
    "status": "running",
    "scheduler_job_id": "job-12345",
    "gpus_allocated": 4
  }
}
```

## Environment Variables

### API Gateway

- `CELERY_BROKER_URL` - RabbitMQ connection URL (default: `amqp://admin:admin@rabbitmq:5672//`)
- `API_KEY` - Bearer token for authentication (default: `opentinker-api-key-2025`)
- `API_HOST` - Host to bind (default: `0.0.0.0`)
- `API_PORT` - Port to listen on (default: `8000`)

### Worker

- `CELERY_BROKER_URL` - RabbitMQ connection URL
- `SCHEDULER_URL` - OpenTinker scheduler URL (default: `http://opentinker-scheduler:8780`)
- `POLL_INTERVAL` - Job status poll interval in seconds (default: `30`)
- `MAX_TRAINING_TIME` - Maximum training time in seconds (default: `7200`)

## Deployment

See [../deploy/README.md](../deploy/README.md) for Kubernetes deployment instructions.

## Development

### Local Testing with Docker Compose

```yaml
version: '3.8'
services:
  rabbitmq:
    image: rabbitmq:4.0.9-management-alpine
    ports:
      - "5672:5672"
      - "15672:15672"
    environment:
      RABBITMQ_DEFAULT_USER: admin
      RABBITMQ_DEFAULT_PASS: admin

  api:
    build: ./api
    ports:
      - "8000:8000"
    environment:
      CELERY_BROKER_URL: amqp://admin:admin@rabbitmq:5672//
      API_KEY: test-api-key
    depends_on:
      - rabbitmq

  worker:
    build: ./worker
    environment:
      CELERY_BROKER_URL: amqp://admin:admin@rabbitmq:5672//
      SCHEDULER_URL: http://opentinker-scheduler:8780
    depends_on:
      - rabbitmq
```

```bash
docker-compose up -d
```

## Monitoring

- **RabbitMQ Management UI:** http://localhost:15672 (admin/admin)
- **API Health:** http://localhost:8000/health
- **Worker Logs:** `docker logs -f worker`
- **Queue Depth:** Check RabbitMQ management UI → Queues → `opentinker-training`

## Troubleshooting

### API returns 403 Forbidden
- Check `API_KEY` environment variable matches Bearer token
- Ensure `Authorization: Bearer <token>` header is set

### Worker not consuming tasks
- Check RabbitMQ connectivity: `CELERY_BROKER_URL` correct
- Verify queue name: `opentinker-training`
- Check worker logs for errors

### Training jobs timeout
- Increase `MAX_TRAINING_TIME` environment variable
- Check scheduler connectivity: `SCHEDULER_URL` accessible
- Verify H100 GPU nodes are available in us-east4-a cluster

### Cross-region latency issues
- Normal latency us-central1 ↔ us-east4-a: 20-50ms
- If > 100ms, check network configuration
- RabbitMQ provides fault tolerance for transient issues
