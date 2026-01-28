# OpenTinker Minimal Runtime Dockerfile
# Based on NVIDIA PyTorch image with CUDA 12.8 and Python 3.12
# Includes only dependencies needed to run OpenTinker scheduler and ALFWorld environment

FROM nvcr.io/nvidia/pytorch:25.03-py3

# Define environment variables for runtime
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn
ENV VLLM_USE_V1=1
ENV VLLM_DEVICE_MEM_ALLOCATOR=cuda
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_ROOT_USER_ACTION=ignore
ENV HF_HUB_ENABLE_HF_TRANSFER="1"
ENV CUDA_HOME=/usr/local/cuda

# Set working directory
WORKDIR /workspace

# Upgrade pip
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Install minimal system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        wget \
        curl \
        build-essential \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Note: Base image includes PyTorch 2.7.0 with CUDA 12.8 - using that instead of reinstalling
# Base image also includes: numpy 1.26.4, pandas 2.2.3, pyarrow 19.0.1, dill 0.3.9

# Temporarily disable constraints to upgrade pyarrow for datasets compatibility
RUN PIP_CONSTRAINT="" pip install --no-cache-dir --upgrade "pyarrow==21.0.0"

# Install core OpenTinker dependencies (from pyproject.toml)
# Disable PIP_CONSTRAINT to allow pyarrow 21.0.0
RUN PIP_CONSTRAINT="" pip install --no-cache-dir \
    "transformers>=4.35.0" \
    "datasets>=4.1.0" \
    "ray[default]>=2.9.0" \
    "fastapi>=0.104.0" \
    "uvicorn>=0.24.0" \
    "requests>=2.31.0" \
    "omegaconf>=2.3.0" \
    "hydra-core>=1.3.0"

# Install vLLM for inference (required by verl submodule)
RUN PIP_CONSTRAINT="" pip install --no-cache-dir "vllm==0.11.0"

# Install verl dependencies (skip packages already in base image)
RUN PIP_CONSTRAINT="" pip install --no-cache-dir \
    accelerate \
    codetiming \
    liger-kernel \
    peft \
    pylatexenc \
    pybind11 \
    "tensordict>=0.8.0,<=0.10.0,!=0.9.0" \
    torchdata

# Install ALFWorld for environment server
RUN PIP_CONSTRAINT="" pip install --no-cache-dir alfworld

# Optional: Install wandb for logging
RUN PIP_CONSTRAINT="" pip install --no-cache-dir wandb

###############################################################################
# Runtime layout: mount OpenTinker source as a volume, logs under /logs
###############################################################################

# Create expected OpenTinker workdir (will be populated via volume mount)
RUN mkdir -p /workspace/OpenTinker

# Create a central logs directory and backward-compatible symlink
RUN mkdir -p /logs && ln -sf /logs /workspace/logs

# Expose log directory as an environment variable
ENV LOG_DIR=/logs

# Set working directory to where the OpenTinker repo will be mounted
WORKDIR /workspace/OpenTinker

# Container will expect start_services.sh inside the mounted repo
ENTRYPOINT ["/bin/bash", "/workspace/OpenTinker/start_services.sh"]

# Default command (can be overridden)
CMD []

