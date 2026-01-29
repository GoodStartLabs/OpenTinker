# Multi-Region OpenTinker Deployment Guide

Deploy OpenTinker training workers across 2 regions (us-central1, us-east4) for dynamic H100 GPU availability.

## Architecture Overview

```
                    Argo Workflows (us-central1)
                              ↓
                    FastAPI Gateway (us-central1)
                              ↓
                    RabbitMQ (us-central1) ← Control Plane
                              ↓
        ┌─────────────────────┴─────────────────────┐
        ↓                                           ↓
Workers (us-central1)                      Workers (us-east4)
  4x H100 GPUs                               4x H100 GPUs

Existing Autopilot Clusters:
- dev-autopilot-cluster (us-central1)
- us-east4-autopilot-cluster (us-east4)
```

**Key Features:**
- ✅ Workers deployed to any region based on H100 availability
- ✅ Single RabbitMQ message queue for all regions
- ✅ Cross-region communication via GCP Internal Load Balancer
- ✅ KEDA auto-scaling per region
- ✅ Independent worker scaling in each region

## Prerequisites

### 1. Existing GKE Autopilot Clusters

This guide uses your existing Autopilot clusters:
- **dev-autopilot-cluster** (us-central1) - Control plane + workers
- **us-east4-autopilot-cluster** (us-east4) - Workers only

**Note**: Autopilot automatically provisions GPU nodes when requested, no manual node pool creation needed.

### 2. GCP Project Setup

```bash
export PROJECT_ID=development-472321
gcloud config set project $PROJECT_ID

# Get credentials for both clusters
gcloud container clusters get-credentials dev-autopilot-cluster \
  --region=us-central1 \
  --project=$PROJECT_ID

gcloud container clusters get-credentials us-east4-autopilot-cluster \
  --region=us-east4 \
  --project=$PROJECT_ID

# Create context aliases for easier switching
kubectl config rename-context gke_${PROJECT_ID}_us-central1_dev-autopilot-cluster central
kubectl config rename-context gke_${PROJECT_ID}_us-east4_us-east4-autopilot-cluster east4
```

### 3. Build and Push Docker Images

```bash
cd OpenTinker

# Build API image
docker build -t us-central1-docker.pkg.dev/$PROJECT_ID/opentinker/api:latest \
  -f service/api/Dockerfile \
  service/api/

# Build worker image with OpenTinker + CUDA + Celery
docker build -t us-central1-docker.pkg.dev/$PROJECT_ID/opentinker/worker:latest \
  -f service/worker/Dockerfile \
  .

# Push to Artifact Registry
gcloud auth configure-docker us-central1-docker.pkg.dev
docker push us-central1-docker.pkg.dev/$PROJECT_ID/opentinker/api:latest
docker push us-central1-docker.pkg.dev/$PROJECT_ID/opentinker/worker:latest
```

## Step 1: Deploy Control Plane (us-central1)

### 1.1 Switch to us-central1 Cluster

```bash
kubectl config use-context central

# Verify cluster access
kubectl get nodes
```

### 1.2 Install KEDA (Kubernetes Event-Driven Autoscaling)

```bash
kubectl apply -f https://github.com/kedacore/keda/releases/download/v2.15.1/keda-2.15.1.yaml

# Verify KEDA is running
kubectl get pods -n keda
```

### 1.3 Deploy Base Resources (API Gateway + RabbitMQ)

```bash
cd OpenTinker

# Deploy all base resources using Kustomize
kubectl apply -k deploy/base

# Wait for services to be ready
kubectl wait --for=condition=ready pod -l app=rabbitmq -n opentinker --timeout=300s

# Check all resources
kubectl get all -n opentinker
```

### 1.4 Get RabbitMQ External IP

```bash
# Wait for Load Balancer to assign IP (takes 2-3 minutes)
kubectl get svc rabbitmq-external -n opentinker -w

# Once EXTERNAL-IP shows (e.g., 10.128.0.50):
export RABBITMQ_EXTERNAL_IP=$(kubectl get svc rabbitmq-external -n opentinker -o jsonpath='{.status.loadBalancer.ingress[0].ip}')

echo "RabbitMQ External IP: $RABBITMQ_EXTERNAL_IP"
# Save this IP - you'll need it for worker deployments
```

## Step 2: Deploy Workers in us-central1

**Note**: Using the same cluster (dev-autopilot-cluster) for both control plane and workers.

### 2.1 Deploy Workers

```bash
# Already in central context
kubectl config use-context central

# Deploy workers (RabbitMQ is in same cluster, uses local service name)
kubectl apply -k deploy/overlays/workers-us-central1

# Verify deployment
kubectl get all -n opentinker-workers-us-central1

# Watch for pods to become ready
kubectl get pods -n opentinker-workers-us-central1 -w
```

**Expected**: You'll see 1 worker pod running initially. Autopilot will automatically provision H100 GPU nodes when jobs require them (3-8 min delay).

## Step 3: Deploy Workers in us-east4

### 3.1 Switch to us-east4 Cluster

```bash
kubectl config use-context east4

# Verify cluster access
kubectl get nodes
```

### 3.2 Install KEDA on us-east4 Cluster

```bash
kubectl apply -f https://github.com/kedacore/keda/releases/download/v2.15.1/keda-2.15.1.yaml

# Verify KEDA is running
kubectl get pods -n keda
```

### 3.3 Get RabbitMQ External IP from us-central1

```bash
# Switch back to us-central1
kubectl config use-context central

# Get the Internal Load Balancer IP
export RABBITMQ_EXTERNAL_IP=$(kubectl get svc -n opentinker rabbitmq-external -o jsonpath='{.status.loadBalancer.ingress[0].ip}')

echo "RabbitMQ External IP: $RABBITMQ_EXTERNAL_IP"
# Save this IP - you'll need it for the next step
```

### 3.4 Update Worker Configuration with RabbitMQ IP

```bash
cd OpenTinker

# Update us-east4 kustomization with RabbitMQ IP
sed -i.bak "s/RABBITMQ_EXTERNAL_IP/$RABBITMQ_EXTERNAL_IP/g" \
  deploy/overlays/workers-us-east4/kustomization.yaml

# Verify the update
grep CELERY_BROKER_URL deploy/overlays/workers-us-east4/kustomization.yaml
```

### 3.5 Deploy Workers

```bash
# Switch to us-east4 context
kubectl config use-context east4

kubectl apply -k deploy/overlays/workers-us-east4

# Verify deployment
kubectl get all -n opentinker-workers-us-east4

# Watch for pods
kubectl get pods -n opentinker-workers-us-east4 -w
```

**Expected**: You'll see 1 worker pod running initially. Autopilot will automatically provision H100 GPU nodes when KEDA scales workers based on queue depth.

## Verification

### Check All Deployments

```bash
# Control plane (us-central1)
kubectl config use-context central
kubectl get all -n opentinker

# Workers us-central1 (same cluster)
kubectl get all -n opentinker-workers-us-central1

# Workers us-east4
kubectl config use-context east4
kubectl get all -n opentinker-workers-us-east4
```

### Test Cross-Region Connectivity

```bash
# Check RabbitMQ from us-central1
kubectl config use-context central
kubectl exec -n opentinker rabbitmq-0 -- rabbitmqctl list_queues

# You should see: opentinker-training queue with 0 messages
```

### Submit Test Training Job

```bash
# Port-forward API from control plane
kubectl port-forward -n opentinker service/api 8000:80 &

# Submit job
curl -X POST http://localhost:8000/train \
  -H "Authorization: Bearer opentinker-api-key-2025" \
  -H "Content-Type: application/json" \
  -d '{
    "task": "gomoku",
    "num_gpus": 4,
    "config": {
      "experiment_name": "multi_region_test",
      "batch_size": 32
    }
  }'

# Response should include job_id
```

### Monitor Job Execution

```bash
# Check which region picked up the job
# Check us-central1
kubectl config use-context central
kubectl logs -n opentinker-workers-us-central1 -l app=opentinker-worker --tail=50 -f

# Check us-east4
kubectl config use-context east4
kubectl logs -n opentinker-workers-us-east4 -l app=opentinker-worker --tail=50 -f
```

### Monitor GPU Node Provisioning

```bash
# Watch for H100 nodes being created (3-5 min delay)
kubectl get nodes -l cloud.google.com/gke-accelerator=nvidia-h100-80gb -w
```

## Workload Identity Setup (Optional but Recommended)

Set up GCP service accounts for each region's workers:

```bash
# Create service accounts
gcloud iam service-accounts create opentinker-worker-us-central1 \
  --display-name="OpenTinker Worker US-Central1"

gcloud iam service-accounts create opentinker-worker-us-east4 \
  --display-name="OpenTinker Worker US-East4"

# Grant permissions (example: Artifact Registry reader)
for region in us-central1 us-east4; do
  gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:opentinker-worker-${region}@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role="roles/artifactregistry.reader"

  gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:opentinker-worker-${region}@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role="roles/logging.logWriter"
done

# Bind to Kubernetes service accounts
for region in us-central1 us-east4 us-west1; do
  gcloud iam service-accounts add-iam-policy-binding \
    opentinker-worker-${region}@${PROJECT_ID}.iam.gserviceaccount.com \
    --role roles/iam.workloadIdentityUser \
    --member "serviceAccount:${PROJECT_ID}.svc.id.goog[opentinker-workers-${region}/worker]"
done
```

## Scaling Configuration

### Per-Region Worker Scaling

Each region scales independently based on RabbitMQ queue depth:

```yaml
# KEDA ScaledObject (same for all regions)
minReplicaCount: 1        # Min workers when queue has messages
maxReplicaCount: 20       # Max workers per region
idleReplicaCount: 1       # Workers when queue is empty
cooldownPeriod: 300       # 5 minutes before scaling down
queueLength: "5"          # Scale up when >5 messages per worker
```

### GPU Node Auto-Scaling

Each region's H100 node pool scales independently:

```yaml
min-nodes: 0              # Scale to zero when no workers need GPUs
max-nodes: 8              # Max 8 H100 nodes per region (64 GPUs)
```

**Total capacity:** 3 regions × 8 nodes × 8 GPUs = **192 H100 GPUs**

## Cost Estimates

### Per Region (Idle)

| Component | Cost/Month |
|-----------|------------|
| Control cluster (3 e2-standard-4) | ~$150 |
| Worker cluster (1 e2-standard-4) | ~$50 |
| Internal Load Balancer | ~$20 |
| **Subtotal per region** | **~$220** |

### Per Region (Under Load)

| Resource | Cost/Hour |
|----------|-----------|
| 1 H100 node (8x H100 GPUs) | ~$30-40 |
| 1 Worker pod (4 GPUs) | ~$15-20 |

**Example:** 10 concurrent training jobs across 3 regions = 10 workers × $20/hour = **$200/hour**

## Monitoring & Observability

### RabbitMQ Management UI

```bash
# Port-forward to RabbitMQ management
gcloud container clusters get-credentials opentinker-control --zone=us-central1-a
kubectl port-forward -n opentinker rabbitmq-0 15672:15672

# Open http://localhost:15672
# Login: admin / admin
```

### Worker Logs

```bash
# View logs from all workers in a region
kubectl logs -n opentinker-workers-us-central1 -l app=opentinker-worker -f

# View specific worker pod logs
kubectl logs -n opentinker-workers-us-central1 <pod-name> -f
```

### KEDA Metrics

```bash
# Check KEDA scaling metrics
kubectl get scaledobject -n opentinker-workers-us-central1
kubectl describe scaledobject opentinker-worker-scaler -n opentinker-workers-us-central1
```

### GPU Utilization

```bash
# Get H100 node
NODE=$(kubectl get nodes -l cloud.google.com/gke-accelerator=nvidia-h100-80gb -o name | head -1)

# Check GPU utilization
kubectl debug $NODE -it --image=nvidia/cuda:12.2.0-base-ubuntu22.04 -- nvidia-smi
```

## Troubleshooting

### Workers Not Connecting to RabbitMQ

```bash
# Check RabbitMQ external service
kubectl get svc rabbitmq-external -n opentinker

# Test connectivity from worker pod
kubectl exec -n opentinker-workers-us-east4 <worker-pod> -- \
  curl -v telnet://$RABBITMQ_EXTERNAL_IP:5672
```

### No GPU Nodes Provisioning

```bash
# Check GPU quota
gcloud compute project-info describe --project=$PROJECT_ID | grep -A5 H100

# Check node pool autoscaling
gcloud container node-pools describe h100-pool \
  --cluster=opentinker-workers-us-east4 \
  --zone=us-east4-a
```

### Worker Image Pull Errors

```bash
# Check Artifact Registry permissions
gcloud artifacts repositories get-iam-policy opentinker \
  --location=us-central1

# Manually pull image to test
docker pull us-central1-docker.pkg.dev/$PROJECT_ID/opentinker/worker:latest
```

### High Cross-Region Latency

```bash
# Test latency between regions
# From us-east4 worker to RabbitMQ in us-central1
kubectl exec -n opentinker-workers-us-east4 <worker-pod> -- \
  ping -c 10 $RABBITMQ_EXTERNAL_IP

# Expected: 20-50ms for us-central1 ↔ us-east4
```

## Cleanup

### Delete Worker Clusters

```bash
gcloud container clusters delete opentinker-workers-us-central1 --zone=us-central1-a --quiet
gcloud container clusters delete opentinker-workers-us-east4 --zone=us-east4-a --quiet
gcloud container clusters delete opentinker-workers-us-west1 --zone=us-west1-a --quiet
```

### Delete Control Plane

```bash
gcloud container clusters delete opentinker-control --zone=us-central1-a --quiet
```

### Delete Service Accounts

```bash
gcloud iam service-accounts delete opentinker-worker-us-central1@$PROJECT_ID.iam.gserviceaccount.com --quiet
gcloud iam service-accounts delete opentinker-worker-us-east4@$PROJECT_ID.iam.gserviceaccount.com --quiet
gcloud iam service-accounts delete opentinker-worker-us-west1@$PROJECT_ID.iam.gserviceaccount.com --quiet
```

## Next Steps

1. **Add more regions** as H100 availability expands (europe-west4, asia-southeast1)
2. **Set up Prometheus/Grafana** for metrics and dashboards
3. **Implement priority queues** for urgent training jobs
4. **Add checkpoint storage** to GCS for training results
5. **Configure spot/preemptible nodes** for cost savings on non-critical workloads
