# OpenTinker GPU Training Worker
# Based on OpenTinker base image with CUDA 12.8, PyTorch, and vLLM
# Runs Celery worker that executes training directly on H100 GPUs

FROM nvcr.io/nvidia/pytorch:25.03-py3

LABEL maintainer="OpenTinker Community"
LABEL description="GPU-enabled Celery worker for OpenTinker training on H100 GPUs"

# Environment variables for vLLM and CUDA
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn \
    VLLM_USE_V1=1 \
    VLLM_DEVICE_MEM_ALLOCATOR=cuda \
    DEBIAN_FRONTEND=noninteractive \
    PIP_ROOT_USER_ACTION=ignore \
    HF_HUB_ENABLE_HF_TRANSFER="1" \
    CUDA_HOME=/usr/local/cuda

# Set working directory
WORKDIR /workspace

# Install uv package manager (much faster than pip)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.cargo/bin/uv /usr/local/bin/

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        wget \
        curl \
        build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pyarrow for datasets compatibility
RUN uv pip install --system --no-cache "pyarrow==21.0.0"

# Install core OpenTinker dependencies
RUN uv pip install --system --no-cache \
    "transformers>=4.35.0" \
    "datasets>=4.1.0" \
    "ray[default]>=2.9.0" \
    "fastapi>=0.104.0" \
    "uvicorn>=0.24.0" \
    "requests>=2.31.0" \
    "omegaconf>=2.3.0" \
    "hydra-core>=1.3.0"

# Install vLLM for inference
RUN uv pip install --system --no-cache "vllm==0.11.0"

# Install verl dependencies
RUN uv pip install --system --no-cache \
    accelerate \
    codetiming \
    liger-kernel \
    peft \
    pylatexenc \
    pybind11 \
    "tensordict>=0.8.0,<=0.10.0,!=0.9.0" \
    torchdata

# Install ALFWorld for game environments
RUN uv pip install --system --no-cache alfworld

# Install wandb for experiment tracking
RUN uv pip install --system --no-cache wandb

# Install Celery for task queue
RUN uv pip install --system --no-cache \
    "celery>=5.3.0" \
    "kombu>=5.3.0"

# Create directories
RUN mkdir -p /workspace/OpenTinker /logs && \
    ln -sf /logs /workspace/logs

# Set log directory
ENV LOG_DIR=/logs

# Copy OpenTinker source code
COPY ../../ /workspace/OpenTinker/

# Copy worker service code
COPY . /workspace/worker/

# Set working directory to worker
WORKDIR /workspace/worker

# Run Celery worker with 1 concurrent training job per worker
# (Each worker gets its own H100 GPU allocation via Kubernetes)
CMD ["celery", "-A", "celery_app", "worker", \
     "--loglevel=info", \
     "--concurrency=1", \
     "--pool=solo"]
