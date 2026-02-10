#!/bin/bash
# Startup script for OpenTinker Celery worker
# Starts scheduler and environment server before launching Celery

set -e

# ============================================
# vLLM Configuration for multi-GPU stability
# ============================================
export VLLM_USE_V1=1
export VLLM_DEVICE_MEM_ALLOCATOR=cuda

echo "=========================================="
echo "OpenTinker Worker Startup"
echo "=========================================="
echo ""
echo "vLLM Config: VLLM_USE_V1=$VLLM_USE_V1, VLLM_DEVICE_MEM_ALLOCATOR=$VLLM_DEVICE_MEM_ALLOCATOR"
echo ""

# Change to OpenTinker root
cd /workspace/OpenTinker

# Detect available GPUs
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    GPU_LIST=$(seq -s, 0 $((GPU_COUNT-1)))
    echo "Detected $GPU_COUNT GPUs: $GPU_LIST"
else
    GPU_COUNT=8
    GPU_LIST="0,1,2,3,4,5,6,7"
    echo "nvidia-smi not available, assuming $GPU_COUNT GPUs"
fi

# Start scheduler in background
echo "Starting Job Scheduler..."
python3 opentinker/scheduler/launch_scheduler_kill.py \
    available_gpus=[$GPU_LIST] \
    port_range=null \
    num_ports=200 \
    scheduler_port=8780 \
    > /logs/scheduler.log 2>&1 &

SCHEDULER_PID=$!
echo "Scheduler started (PID: $SCHEDULER_PID)"

# Wait for scheduler to initialize
sleep 5

# Start ALFWorld environment server in background
echo "Starting ALFWorld Environment Server..."
python3 -m opentinker.environment.alfworld.alfworld_server \
    --port 8092 \
    --max_steps 50 \
    --split train \
    --shards 32 \
    --num_games 5 \
    > /logs/alfworld_env.log 2>&1 &

ALFWORLD_PID=$!
echo "ALFWorld server started (PID: $ALFWORLD_PID)"

# Start Diplomacy environment server in background
echo "Starting Diplomacy Environment Server..."
python3 -m opentinker.environment.diplomacy.diplomacy_server \
    --port 8093 \
    --opponent_model "${DIPLOMACY_OPPONENT_MODEL:-x-ai/grok-4-fast}" \
    > /logs/diplomacy_env.log 2>&1 &

DIPLOMACY_PID=$!
echo "Diplomacy server started (PID: $DIPLOMACY_PID)"

# Wait for env servers to initialize
sleep 3

echo ""
echo "=========================================="
echo "Background services started!"
echo "=========================================="
echo "Scheduler: http://localhost:8780 (log: /logs/scheduler.log)"
echo "ALFWorld Environment: http://localhost:8092 (log: /logs/alfworld_env.log)"
echo "Diplomacy Environment: http://localhost:8093 (log: /logs/diplomacy_env.log)"
echo ""
echo "Starting Celery worker..."
echo "=========================================="
echo ""

# Change to worker directory and start Celery
cd /workspace/OpenTinker/service/worker

# Run Celery worker in foreground (handles SIGTERM properly)
exec celery -A celery_app worker \
    --loglevel=info \
    --concurrency=1 \
    --pool=solo \
    -Q opentinker-training
