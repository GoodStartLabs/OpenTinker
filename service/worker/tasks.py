"""
Celery tasks for executing OpenTinker training jobs directly on H100 GPUs.

Worker runs training in the same container with GPU access, no separate scheduler needed.
"""
from celery import shared_task
import logging
import subprocess
import os
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# OpenTinker paths
OPENTINKER_ROOT = Path("/workspace/OpenTinker")
CLIENT_PATH = OPENTINKER_ROOT / "opentinker" / "client"

# Task configuration
TASK_SCRIPTS = {
    "gomoku": "gomoku_rl.py",
    "alfworld": "alfworld_rl.py",
    "math": "math_rl.py",
    "math_tool": "math_tool_rl.py",
    "geo3k": "geo3k_rl.py",
}


@shared_task(bind=True, max_retries=3, default_retry_delay=60)
def run_training(self, training_request: dict):
    """
    Execute training job directly on H100 GPUs.

    Flow:
    1. Parse training configuration
    2. Set up environment (CUDA, vLLM)
    3. Launch training script directly
    4. Monitor training progress
    5. Return results

    Args:
        training_request: Training configuration dict with:
            - task: Training task name ('gomoku', 'alfworld', etc.)
            - config: Hydra config overrides
            - num_gpus: Number of H100 GPUs (1-8)
            - checkpoint_path: Optional checkpoint for resuming

    Returns:
        dict: Training results with job_id, status, checkpoint_path, metrics

    Raises:
        Exception: If training fails permanently
    """
    task_name = training_request['task']
    config = training_request.get('config', {})
    num_gpus = training_request.get('num_gpus', 4)
    checkpoint_path = training_request.get('checkpoint_path')

    request_id = self.request.id
    logger.info(f"[{request_id}] Starting training job: task={task_name}, gpus={num_gpus}")

    # Validate task
    if task_name not in TASK_SCRIPTS:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(TASK_SCRIPTS.keys())}")

    script_name = TASK_SCRIPTS[task_name]
    script_path = CLIENT_PATH / script_name

    if not script_path.exists():
        raise FileNotFoundError(f"Training script not found: {script_path}")

    try:
        # Set up environment
        env = os.environ.copy()
        env.update({
            "VLLM_USE_V1": "1",
            "VLLM_DEVICE_MEM_ALLOCATOR": "cuda",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "CUDA_VISIBLE_DEVICES": ",".join(str(i) for i in range(num_gpus)),
        })

        # Build Hydra config overrides
        hydra_overrides = []
        for key, value in config.items():
            hydra_overrides.append(f"{key}={value}")

        # Add num_gpus override
        hydra_overrides.append(f"num_gpus={num_gpus}")

        # Add checkpoint path if provided
        if checkpoint_path:
            hydra_overrides.append(f"checkpoint_path={checkpoint_path}")

        # Build command
        cmd = [
            sys.executable,  # Use same Python interpreter
            str(script_path),
        ] + hydra_overrides

        logger.info(f"[{request_id}] Executing: {' '.join(cmd)}")
        logger.info(f"[{request_id}] Working directory: {OPENTINKER_ROOT}")
        logger.info(f"[{request_id}] GPUs: {num_gpus} (CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']})")

        # Execute training
        result = subprocess.run(
            cmd,
            cwd=str(OPENTINKER_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=7200  # 2 hour timeout
        )

        # Log output
        if result.stdout:
            logger.info(f"[{request_id}] Training stdout:\n{result.stdout}")
        if result.stderr:
            logger.warning(f"[{request_id}] Training stderr:\n{result.stderr}")

        # Check result
        if result.returncode != 0:
            error_msg = f"Training failed with exit code {result.returncode}"
            if result.stderr:
                error_msg += f"\nError: {result.stderr[-1000:]}"  # Last 1000 chars
            logger.error(f"[{request_id}] {error_msg}")
            raise Exception(error_msg)

        logger.info(f"[{request_id}] Training completed successfully")

        # Parse checkpoint path from output (if available)
        # Look for common checkpoint patterns in output
        checkpoint_path_result = None
        for line in result.stdout.split('\n'):
            if 'checkpoint' in line.lower() and ('saved' in line.lower() or 'path' in line.lower()):
                # Try to extract path
                parts = line.split()
                for part in parts:
                    if 'checkpoint' in part.lower() and ('/' in part or '\\' in part):
                        checkpoint_path_result = part
                        break

        return {
            "status": "completed",
            "job_id": request_id,
            "task": task_name,
            "num_gpus": num_gpus,
            "checkpoint_path": checkpoint_path_result,
            "exit_code": result.returncode
        }

    except subprocess.TimeoutExpired:
        error_msg = f"Training timeout after 2 hours"
        logger.error(f"[{request_id}] {error_msg}")
        raise Exception(error_msg)

    except Exception as e:
        logger.error(f"[{request_id}] Training error: {str(e)}")

        # Retry on transient errors
        error_str = str(e).lower()
        if any(keyword in error_str for keyword in ['timeout', 'connection', 'network', 'oom', 'out of memory']):
            logger.info(f"[{request_id}] Transient error detected, will retry")
            raise self.retry(exc=e)

        # Permanent failure
        logger.error(f"[{request_id}] Permanent failure, not retrying")
        raise


@shared_task
def health_check():
    """
    Health check task for worker monitoring.

    Verifies:
    - GPU availability
    - OpenTinker installation
    - Training scripts accessible

    Returns:
        dict: Worker health status
    """
    try:
        # Check GPU availability
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5
        )

        gpu_info = result.stdout.strip() if result.returncode == 0 else "No GPUs detected"

        # Check OpenTinker installation
        opentinker_exists = OPENTINKER_ROOT.exists()

        # Check training scripts
        scripts_found = []
        for task, script in TASK_SCRIPTS.items():
            if (CLIENT_PATH / script).exists():
                scripts_found.append(task)

        return {
            "worker": "healthy",
            "gpus": gpu_info,
            "opentinker_installed": opentinker_exists,
            "training_scripts": scripts_found,
            "opentinker_root": str(OPENTINKER_ROOT)
        }

    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "worker": "unhealthy",
            "error": str(e)
        }
