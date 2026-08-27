"""
GPU Detection and Distributed Computing Utilities

Provides automatic GPU detection, CPU fallback, and distributed training support
for single-node multi-GPU and multi-node multi-GPU scenarios.
"""

import torch
import torch.distributed as dist
import os
import logging
from typing import Tuple, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DeviceStrategy:
    """Device strategy information."""

    device_type: str  # 'cuda' or 'cpu'
    device_count: int  # Number of devices
    primary_device: torch.device  # Primary device for this process
    is_distributed: bool  # Whether running in distributed mode
    world_size: int  # Total number of processes
    rank: int  # Current process rank
    local_rank: int  # Local rank on this node


def detect_gpus() -> Tuple[int, List[str]]:
    """
    Detect available GPUs.

    Returns:
        Tuple of (gpu_count, device_list)
    """
    if not torch.cuda.is_available():
        logger.info("CUDA not available. No GPUs detected.")
        return 0, []

    gpu_count = torch.cuda.device_count()
    device_list = [f"cuda:{i}" for i in range(gpu_count)]

    logger.info(f"Detected {gpu_count} GPU(s): {device_list}")

    # Log GPU details
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        logger.info(
            f"  GPU {i}: {props.name}, "
            f"{props.total_memory / 1024**3:.2f} GB memory"
        )

    return gpu_count, device_list


def is_distributed() -> bool:
    """
    Check if running in distributed mode.

    Returns:
        True if torch.distributed is initialized
    """
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    """
    Get total number of processes across all nodes.

    Returns:
        World size (total processes)
    """
    if is_distributed():
        return dist.get_world_size()
    return 1


def get_rank() -> int:
    """
    Get current process rank.

    Returns:
        Process rank (0 if not distributed)
    """
    if is_distributed():
        return dist.get_rank()
    return 0


def get_local_rank() -> int:
    """
    Get local rank on this node.

    Returns:
        Local rank (0 if not distributed)
    """
    if is_distributed():
        # Try to get local rank from environment
        local_rank = os.environ.get("LOCAL_RANK")
        if local_rank is not None:
            return int(local_rank)
        # Fallback: assume single node
        return get_rank()
    return 0


def setup_distributed(backend: str = "nccl") -> bool:
    """
    Initialize distributed training environment.

    This function detects if running under torchrun and initializes
    torch.distributed accordingly.

    Args:
        backend: Distributed backend ('nccl' for GPU, 'gloo' for CPU)

    Returns:
        True if distributed was initialized, False if already initialized or not needed
    """
    # Check if already initialized
    if is_distributed():
        logger.info("torch.distributed already initialized")
        return False

    # Check for distributed environment variables set by torchrun
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        logger.info(
            f"Initializing distributed training: "
            f"rank={rank}, world_size={world_size}, local_rank={local_rank}"
        )

        # Initialize process group
        dist.init_process_group(backend=backend)

        # Set device for this process
        if backend == "nccl" and torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

        logger.info(f"Distributed training initialized successfully (backend={backend})")
        return True

    logger.info("Not running in distributed mode (no RANK/WORLD_SIZE env vars)")
    return False


def cleanup_distributed():
    """Cleanup distributed training environment."""
    if is_distributed():
        logger.info("Cleaning up distributed environment")
        dist.destroy_process_group()


def get_device_strategy(auto_detect: bool = True, fallback_cpu: bool = True) -> DeviceStrategy:
    """
    Determine device strategy for this process.

    Args:
        auto_detect: Whether to auto-detect GPUs
        fallback_cpu: Whether to fallback to CPU on errors

    Returns:
        DeviceStrategy object
    """
    distributed = is_distributed()
    world_size = get_world_size()
    rank = get_rank()
    local_rank = get_local_rank()

    # Detect GPUs
    if auto_detect and torch.cuda.is_available():
        gpu_count, device_list = detect_gpus()

        if gpu_count > 0:
            # Determine which GPU to use
            if distributed:
                # In distributed mode, use local rank to select GPU
                device_idx = local_rank % gpu_count
                primary_device = torch.device(f"cuda:{device_idx}")
            else:
                # Not distributed, use first GPU
                primary_device = torch.device("cuda:0")

            return DeviceStrategy(
                device_type="cuda",
                device_count=gpu_count,
                primary_device=primary_device,
                is_distributed=distributed,
                world_size=world_size,
                rank=rank,
                local_rank=local_rank,
            )

    # Fallback to CPU
    logger.info("Using CPU (no GPUs detected or auto_detect=False)")
    return DeviceStrategy(
        device_type="cpu",
        device_count=0,
        primary_device=torch.device("cpu"),
        is_distributed=distributed,
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
    )


def try_load_model_on_device(
    model: torch.nn.Module,
    device: torch.device,
    fallback_cpu: bool = True,
) -> Tuple[torch.nn.Module, torch.device]:
    """
    Try to load model on specified device with CPU fallback.

    Args:
        model: Model to load
        device: Target device
        fallback_cpu: Whether to fallback to CPU on OOM

    Returns:
        Tuple of (model, actual_device)
    """
    try:
        logger.info(f"Loading model on {device}")
        model = model.to(device)

        # Test with dummy forward pass
        if device.type == "cuda":
            torch.cuda.empty_cache()

        logger.info(f"Model loaded successfully on {device}")
        return model, device

    except RuntimeError as e:
        if "out of memory" in str(e).lower() and fallback_cpu:
            logger.warning(
                f"CUDA out of memory on {device}. Falling back to CPU."
            )

            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Load on CPU
            cpu_device = torch.device("cpu")
            model = model.to(cpu_device)
            logger.info("Model loaded on CPU")

            return model, cpu_device
        else:
            # Re-raise if not OOM or fallback disabled
            raise


def gather_objects(obj, world_size: int, rank: int):
    """
    Gather objects from all processes to rank 0.

    Args:
        obj: Object to gather
        world_size: Total number of processes
        rank: Current process rank

    Returns:
        List of gathered objects (only on rank 0, None elsewhere)
    """
    if not is_distributed():
        return [obj]

    # Prepare list to gather into (only on rank 0)
    gathered = [None for _ in range(world_size)] if rank == 0 else None

    # Gather
    dist.gather_object(obj, gathered, dst=0)

    return gathered


def all_gather_objects(obj):
    """
    All-gather objects from all processes to all processes.

    Args:
        obj: Object to gather

    Returns:
        List of gathered objects from all processes
    """
    if not is_distributed():
        return [obj]

    world_size = get_world_size()
    gathered = [None for _ in range(world_size)]

    dist.all_gather_object(gathered, obj)

    return gathered


def barrier():
    """Synchronize all processes."""
    if is_distributed():
        dist.barrier()


def is_main_process() -> bool:
    """
    Check if this is the main process (rank 0).

    Returns:
        True if rank 0, False otherwise
    """
    return get_rank() == 0


def print_rank0(message: str):
    """
    Print message only on rank 0.

    Args:
        message: Message to print
    """
    if is_main_process():
        print(message)


def log_rank0(message: str, level: str = "info"):
    """
    Log message only on rank 0.

    Args:
        message: Message to log
        level: Log level ('info', 'warning', 'error', 'debug')
    """
    if is_main_process():
        log_func = getattr(logger, level, logger.info)
        log_func(message)
