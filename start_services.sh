#!/bin/bash
# Startup script to automatically start OpenTinker services

set -e

# ============================================
# vLLM Configuration for multi-GPU stability
# ============================================
# Use vLLM V1 for async rollout (required for agent_loop)
export VLLM_USE_V1=1
# Use CUDA allocator instead of cumem to avoid "invalid argument" errors on multi-GPU wake_up
export VLLM_DEVICE_MEM_ALLOCATOR=cuda
# NOTE: Do NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True - incompatible with vLLM memory pool

echo "=========================================="
echo "OpenTinker Services Startup"
echo "=========================================="
echo ""
echo "vLLM Config: VLLM_USE_V1=$VLLM_USE_V1, VLLM_DEVICE_MEM_ALLOCATOR=$VLLM_DEVICE_MEM_ALLOCATOR"
echo ""

# Change to workspace directory
cd /workspace/OpenTinker

# Start scheduler in background
echo "Starting Job Scheduler..."
python3 opentinker/scheduler/launch_scheduler_kill.py \
    available_gpus=[0,1,2,3,4,5,6,7] \
    port_range=null \
    num_ports=200 \
    scheduler_port=8780 \
    > /workspace/OpenTinker/scheduler_docker.log 2>&1 &

SCHEDULER_PID=$!
echo "Scheduler started (PID: $SCHEDULER_PID)"

# Wait a bit for scheduler to initialize
sleep 3

# Start ALFWorld environment server in background
echo "Starting ALFWorld Environment Server..."
python3 -m opentinker.environment.alfworld.alfworld_server \
    --port 8082 \
    --max_steps 50 \
    --split train \
    --shards 32 \
    --num_games 5 \
    > /workspace/OpenTinker/alfworld_env_docker.log 2>&1 &

ALFWORLD_PID=$!
echo "ALFWorld server started (PID: $ALFWORLD_PID)"

echo ""
echo "=========================================="
echo "Services started successfully!"
echo "=========================================="
echo "Scheduler: http://localhost:8780"
echo "ALFWorld Environment: http://localhost:8082"
echo ""
echo "Press Ctrl+C to stop all services and exit"
echo "=========================================="
echo ""

# Wait for both processes
wait $SCHEDULER_PID $ALFWORLD_PID

