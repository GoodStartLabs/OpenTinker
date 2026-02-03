# OpenTinker Kubernetes Deployment

Kubernetes manifests for deploying OpenTinker cross-region GPU training service using Kustomize.

## Architecture Overview

```                              
                               FastAPI Training Entry Point (us-central1)
                                          ↓
                            RabbitMQ (us-central1) ← Central Control Plane
                                          ↓
        ┌─────────────────────────────────┴──────────────────────────────┐
        ↓                                 ↓                              ↓
        ↓                                 ↓                              ↓
Workers (us-central1)             Workers (us-east4)                 Workers (...) 
  T4 GPUs                              H100 GPUs                         H200 GPUs

Existing Autopilot Clusters:
- dev-autopilot-cluster (us-central1)
- us-east4-autopilot-cluster (us-east4)
```

**Key Features:**
- ✅ Enabling workers deployment to any region based on GPU availability
- ✅ Single RabbitMQ message queue for all regions
- ✅ Cross-region communication via GCP Internal Load Balancer
- ✅ KEDA auto-scaling per region
- ✅ Independent worker scaling in each region
- TODO: Worker logs pushed to internal monitoring tool or external (weights & biases)
- TODO: Prometheus metrics for training jobs
- TODO: Grafana dashboards
- TODO: Job result storage in GCS

## Architecture

- **API Gateway**: FastAPI service in us-central1 (Argo cluster)
- **RabbitMQ**: Message broker in us-central1
- **Worker**: Celery workers in us-east4-a (GPU cluster)
- **Scheduler**: OpenTinker job scheduler in us-east4-a with H100 GPUs

## Directory Structure

```
deploy/
├── base/                           # Base Kustomize resources
│   ├── namespace.yaml              # opentinker namespace
│   ├── api.yaml                    # FastAPI deployment + service
│   ├── worker.yaml                 # Celery worker deployment
│   ├── worker-serviceaccount.yaml  # Worker service account (Workload Identity)
│   ├── worker-keda-autoscaler.yaml # KEDA autoscaling config
│   ├── rabbitmq.yaml               # RabbitMQ StatefulSet + service
│   ├── scheduler.yaml              # OpenTinker scheduler with H100 GPU
│   ├── secret-store.yaml           # External Secrets Operator config
│   └── kustomization.yaml          # Base kustomization
│
└── overlays/
    ├── dev/                        # Development environment
    │   ├── namespace.yaml          # opentinker-dev
    │   ├── external-secret.yaml    # Secrets from GCP Secret Manager
    │   └── kustomization.yaml      # Dev overrides
    │
    └── production/                 # Production environment
        ├── namespace.yaml          # opentinker-prod
        ├── external-secret.yaml    # Prod secrets
        ├── ingress-route.yaml      # HTTPS ingress
        └── kustomization.yaml      # Prod overrides (3 API replicas, etc.)
```

## Prerequisites

### 1. GKE Clusters

**us-central1 Cluster** (Argo cluster):
```bash
gcloud container clusters create main-cluster \
  --zone=us-central1-a \
  --num-nodes=3 \
  --machine-type=e2-standard-4 \
  --enable-autoscaling \
  --min-nodes=1 \
  --max-nodes=10
```

**us-east4-a Cluster** (GPU cluster):
```bash
gcloud container clusters create gpu-cluster \
  --zone=us-east4-a \
  --num-nodes=1 \
  --machine-type=e2-standard-4 \
  --enable-autoscaling \
  --min-nodes=1 \
  --max-nodes=5
```

### 2. H100 GPU Node Pool (us-east4-a)

```bash
gcloud container node-pools create h100-pool \
  --cluster=gpu-cluster \
  --zone=us-east4-a \
  --machine-type=a3-highgpu-8g \
  --accelerator=type=nvidia-h100-80gb,count=8 \
  --num-nodes=0 \
  --enable-autoscaling \
  --min-nodes=0 \
  --max-nodes=8 \
  --disk-size=200 \
  --disk-type=pd-balanced
```

### 3. Install Required Operators

**KEDA (Kubernetes Event-Driven Autoscaling):**
```bash
kubectl apply -f https://github.com/kedacore/keda/releases/download/v2.12.0/keda-2.12.0.yaml
```

**External Secrets Operator:**
```bash
helm repo add external-secrets https://charts.external-secrets.io
helm install external-secrets external-secrets/external-secrets -n external-secrets-system --create-namespace
```

**NVIDIA GPU Operator (on GPU cluster):**
```bash
kubectl create ns gpu-operator
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm install gpu-operator nvidia/gpu-operator -n gpu-operator --wait
```

### 4. Workload Identity Setup

**Worker Service Account:**
```bash
# Create GCP service account
gcloud iam service-accounts create opentinker-worker-dev \
  --display-name="OpenTinker Worker Dev"

# Grant permissions
gcloud projects add-iam-policy-binding development-472321 \
  --member="serviceAccount:opentinker-worker-dev@development-472321.iam.gserviceaccount.com" \
  --role="roles/artifactregistry.reader"

gcloud projects add-iam-policy-binding development-472321 \
  --member="serviceAccount:opentinker-worker-dev@development-472321.iam.gserviceaccount.com" \
  --role="roles/logging.logWriter"

# Bind to Kubernetes service account
gcloud iam service-accounts add-iam-policy-binding \
  opentinker-worker-dev@development-472321.iam.gserviceaccount.com \
  --role roles/iam.workloadIdentityUser \
  --member "serviceAccount:development-472321.svc.id.goog[opentinker-dev/worker]"
```

### 5. Create Secrets in GCP Secret Manager

```bash
# API key
echo -n "opentinker-api-key-$(openssl rand -hex 16)" | gcloud secrets create opentinker-api-key-dev --data-file=-

# RabbitMQ password
echo -n "$(openssl rand -base64 32)" | gcloud secrets create opentinker-rabbitmq-password-dev --data-file=-
```

## Deployment

### Deploy to Development

```bash
cd deploy/

# Preview changes
kubectl kustomize overlays/dev

# Apply
kubectl apply -k overlays/dev

# Verify deployment
kubectl get all -n opentinker-dev

# Check logs
kubectl logs -n opentinker-dev deployment/api -f
kubectl logs -n opentinker-dev deployment/worker -f
```

### Deploy to Production

```bash
# Update image tags in overlays/production/kustomization.yaml
# Then apply
kubectl apply -k overlays/production

# Verify
kubectl get all -n opentinker-prod
```

## Build and Push Images

```bash
# API Gateway
cd ../service/api
docker build -t us-central1-docker.pkg.dev/development-472321/opentinker/gsl-opentinker-api:dev-latest .
docker push us-central1-docker.pkg.dev/development-472321/opentinker/gsl-opentinker-api:dev-latest

# Worker
cd ../worker
docker build -t us-central1-docker.pkg.dev/development-472321/opentinker/gsl-opentinker-worker:dev-latest .
docker push us-central1-docker.pkg.dev/development-472321/opentinker/gsl-opentinker-worker:dev-latest
```

## Verification

### Test End-to-End Flow

1. **Port-forward API service:**
```bash
kubectl port-forward -n opentinker-dev service/api 8000:80
```

2. **Submit training job:**
```bash
curl -X POST http://localhost:8000/train \
  -H "Authorization: Bearer opentinker-api-key-2025" \
  -H "Content-Type: application/json" \
  -d '{
    "task": "gomoku",
    "num_gpus": 4,
    "config": {
      "experiment_name": "test_gomoku",
      "batch_size": 32
    }
  }'
```

3. **Check RabbitMQ queue:**
```bash
kubectl port-forward -n opentinker-dev statefulset/rabbitmq 15672:15672
# Open http://localhost:15672 (admin/admin)
# Check Queues → opentinker-training
```

4. **Watch worker scaling:**
```bash
kubectl get scaledobject -n opentinker-dev -w
kubectl get pods -n opentinker-dev -l app=opentinker-worker -w
```

5. **Monitor H100 nodes:**
```bash
kubectl get nodes -l cloud.google.com/gke-accelerator=nvidia-h100-80gb -w
```

6. **Check scheduler logs:**
```bash
kubectl logs -n opentinker-dev deployment/opentinker-scheduler -f
```

## Monitoring

### Check API Health

```bash
kubectl exec -n opentinker-dev deployment/api -- curl http://localhost:8000/health
```

### Check Worker Status

```bash
kubectl exec -n opentinker-dev deployment/worker -it -- celery -A celery_app inspect active
```

### RabbitMQ Queue Depth

```bash
kubectl exec -n opentinker-dev rabbitmq-0 -- rabbitmqctl list_queues name messages consumers
```

### GPU Utilization

```bash
kubectl exec -n opentinker-dev deployment/opentinker-scheduler -- nvidia-smi
```

## Scaling Behavior

### Worker Auto-Scaling (KEDA)

- **Trigger**: RabbitMQ queue depth > 5 messages/worker
- **Min replicas**: 1 (dev), 2 (prod)
- **Max replicas**: 20
- **Cooldown**: 5 minutes

### GPU Node Auto-Scaling (GKE)

- **Trigger**: Scheduler requests GPU resources
- **Provisioning time**: 3-5 minutes
- **Scale-down**: 10 minutes idle
- **Max nodes**: 8 (64 H100 GPUs total)

## Troubleshooting

### Workers not consuming tasks

```bash
# Check RabbitMQ connectivity
kubectl logs -n opentinker-dev deployment/worker | grep "Connected to amqp"

# Check KEDA scaler
kubectl describe scaledobject -n opentinker-dev opentinker-worker-scaler

# Manually scale for testing
kubectl scale deployment worker --replicas=3 -n opentinker-dev
```

### Scheduler not accessible

```bash
# Check scheduler health
kubectl port-forward -n opentinker-dev service/opentinker-scheduler 8780:8780
curl http://localhost:8780/health

# Check GPU allocation
kubectl describe pod -n opentinker-dev -l app=opentinker-scheduler | grep nvidia.com/gpu
```

### Training jobs failing

```bash
# Check worker logs for errors
kubectl logs -n opentinker-dev deployment/worker -f

# Check scheduler logs
kubectl logs -n opentinker-dev deployment/opentinker-scheduler -f

# Verify image pull
kubectl get events -n opentinker-dev --sort-by='.lastTimestamp' | grep -i pull
```

### Cross-region latency issues

```bash
# Test latency from worker to scheduler
kubectl exec -n opentinker-dev deployment/worker -- curl -w "@-" -o /dev/null -s http://opentinker-scheduler:8780/health <<'EOF'
time_total: %{time_total}
EOF
```

Expected latency: 20-50ms (us-central1 ↔ us-east4-a)

## Cleanup

```bash
# Delete dev environment
kubectl delete -k overlays/dev

# Delete production environment
kubectl delete -k overlays/production

# Delete GPU node pool
gcloud container node-pools delete h100-pool --cluster=gpu-cluster --zone=us-east4-a --quiet
```

## Cost Optimization

- H100 GPUs are expensive (~$30/hour per node)
- Ensure aggressive scale-down (10 min idle)
- Monitor GPU utilization with Prometheus/Grafana
- Use preemptible nodes for development (lower cost)
- Set budget alerts in GCP

## Security Considerations

- API key authentication (Bearer tokens)
- Workload Identity (no static credentials)
- Non-root containers (UID 1000)
- Read-only root filesystems where possible
- Network policies (future enhancement)
- TLS/HTTPS for production (via ingress)

## Future Enhancements

- [ ] Multi-region RabbitMQ cluster (HA)
- [ ] Prometheus metrics for training jobs
- [ ] Grafana dashboards
- [ ] Cost tracking per training job
- [ ] Job result storage in GCS
- [ ] Webhook notifications on completion
- [ ] Priority queues for urgent jobs
- [ ] Spot/preemptible GPU nodes for dev
