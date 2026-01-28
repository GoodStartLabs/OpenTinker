#!/usr/bin/env python3
"""
Patch vLLM's gpu_worker.py to respect VLLM_DEVICE_MEM_ALLOCATOR environment variable.

The vLLM v1 API in version 0.11.0 has a bug where the wake_up() method is hardcoded
to use CuMemAllocator, ignoring the VLLM_DEVICE_MEM_ALLOCATOR environment variable.

This script patches the gpu_worker.py file to check the environment variable first.
"""

import os
import sys


def patch_vllm_gpu_worker():
    """Patch vLLM's gpu_worker.py to respect VLLM_DEVICE_MEM_ALLOCATOR."""

    # Find vLLM installation path
    try:
        import vllm
        vllm_path = os.path.dirname(vllm.__file__)
    except ImportError:
        print("ERROR: vLLM is not installed")
        return False

    gpu_worker_path = os.path.join(vllm_path, "v1", "worker", "gpu_worker.py")

    if not os.path.exists(gpu_worker_path):
        print(f"ERROR: Could not find {gpu_worker_path}")
        return False

    print(f"Found vLLM gpu_worker.py at: {gpu_worker_path}")

    # Read the file
    with open(gpu_worker_path, 'r') as f:
        content = f.read()

    # Check if already patched
    if "VLLM_DEVICE_MEM_ALLOCATOR" in content and "OpenTinker patch" in content:
        print("✓ File is already patched")
        return True

    # Find the wake_up method and patch it
    original_wake_up = """    def wake_up(self, tags: Optional[list[str]] = None) -> None:
        from vllm.device_allocator.cumem import CuMemAllocator

        allocator = CuMemAllocator.get_instance()
        allocator.wake_up(tags)

        # Restore the buffers after level 2 sleep
        if len(self._sleep_saved_buffers):
            model = self.model_runner.model
            for name, buffer in model.named_buffers():
                if name in self._sleep_saved_buffers:
                    buffer.data.copy_(self._sleep_saved_buffers[name].data)
            self._sleep_saved_buffers = {}"""

    patched_wake_up = """    def wake_up(self, tags: Optional[list[str]] = None) -> None:
        # OpenTinker patch: Respect VLLM_DEVICE_MEM_ALLOCATOR environment variable
        import os
        import logging
        allocator_type = os.environ.get("VLLM_DEVICE_MEM_ALLOCATOR", "cumem")

        logger = logging.getLogger(__name__)
        logger.info(f"[OPENTINKER_PATCH] wake_up called with allocator_type={allocator_type}, tags={tags}")

        if allocator_type == "cuda":
            # Use standard CUDA allocator - no sleep/wake mechanism needed
            # Skip both allocator wake_up and buffer restoration
            logger.info("[OPENTINKER_PATCH] Skipping wake_up for CUDA allocator")
            return
        else:
            # Use cumem allocator (original behavior with sleep/wake)
            logger.info("[OPENTINKER_PATCH] Using cumem allocator wake_up")
            from vllm.device_allocator.cumem import CuMemAllocator
            allocator = CuMemAllocator.get_instance()
            allocator.wake_up(tags)

            # Restore the buffers after level 2 sleep
            if len(self._sleep_saved_buffers):
                model = self.model_runner.model
                for name, buffer in model.named_buffers():
                    if name in self._sleep_saved_buffers:
                        buffer.data.copy_(self._sleep_saved_buffers[name].data)
                self._sleep_saved_buffers = {}"""

    if original_wake_up not in content:
        print("WARNING: Could not find the expected wake_up method signature")
        print("The vLLM version might be different than expected")
        return False

    # Apply the patch
    patched_content = content.replace(original_wake_up, patched_wake_up)

    # Backup original file
    backup_path = gpu_worker_path + ".backup"
    if not os.path.exists(backup_path):
        with open(backup_path, 'w') as f:
            f.write(content)
        print(f"✓ Backed up original to: {backup_path}")

    # Write patched version
    with open(gpu_worker_path, 'w') as f:
        f.write(patched_content)

    print("✓ Successfully patched vLLM gpu_worker.py")
    print("  The wake_up() method now respects VLLM_DEVICE_MEM_ALLOCATOR")
    return True


if __name__ == "__main__":
    success = patch_vllm_gpu_worker()
    sys.exit(0 if success else 1)
