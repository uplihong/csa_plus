import os
import random
import warnings
from typing import Optional, Dict

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from deepspeed.accelerator import get_accelerator
import deepspeed
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from .metric import AverageMeter


def seed_everything(
    seed: Optional[int],
    deterministic: Optional[bool] = None,
    cudnn_benchmark: Optional[bool] = None,
) -> None:
    """Sets the seed and CUDNN behavior for reproducibility/performance tradeoffs."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        if torch.cuda.device_count() > 1:
            torch.cuda.manual_seed_all(seed)

    if deterministic is None:
        deterministic = True if seed is not None else cudnn.deterministic
    if cudnn_benchmark is None:
        cudnn_benchmark = False if seed is not None else cudnn.benchmark

    if deterministic and cudnn_benchmark:
        warnings.warn(
            "cudnn_benchmark=True conflicts with deterministic=True. "
            "Disabling cudnn_benchmark."
        )
        cudnn_benchmark = False

    cudnn.benchmark = cudnn_benchmark
    cudnn.deterministic = deterministic

    if deterministic:
        warnings.warn(
            'You have chosen to enable CUDNN deterministic mode. '
            'This can slow down your training considerably! '
            'You may see unexpected behavior when restarting '
            'from checkpoints.'
        )


def is_aml() -> bool:
    """Checks if running inside an Azure Machine Learning (AML) environment."""
    return 'AZUREML_EXPERIMENT_ID' in os.environ


def is_rank_0() -> bool:
    """
    Check whether the current process is the main rank (rank 0).
    Handles both torch.distributed and DeepSpeed environment variables.
    """
    if dist.is_initialized():
        if dist.get_rank() == 0:
            return True
        # AML specific check for rank 0 on a node
        if is_aml() and dist.get_rank() % get_accelerator().device_count() == 0:
            return True
        return False
    else:
        # Fallback to checking env vars if dist not initialized yet
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        rank = int(os.environ.get("RANK", -1))

        # Single-process fallback when launcher env vars are absent.
        if rank == -1 and local_rank == -1:
            return True
        
        # If standard rank var is present
        if rank == 0:
            return True
        # If only local rank is present (single node multi-GPU pre-init)
        if local_rank == 0 and rank == -1:
            return True
            
        return False


def get_world_size() -> int:
    if dist.is_initialized():
        return dist.get_world_size()
    return int(os.environ.get("WORLD_SIZE", 1))


def get_rank() -> int:
    if dist.is_initialized():
        return dist.get_rank()
    return int(os.environ.get("RANK", 0))


def reduce_meters(meters, rank, cfg):
    """Sync and flush meters."""
    assert isinstance(meters, dict), "collect AverageMeters into a dict"
    if not dist.is_initialized() or dist.get_world_size() == 1:
        for name in sorted(meters.keys()):
            meter = meters[name]
            if not isinstance(meter, AverageMeter):
                raise TypeError("meter should be AverageMeter type")
            meter.update_reduce_v2(meter.avg, meter.sum, meter.count)
        return

    backend = dist.get_backend()
    if backend == "nccl" and torch.cuda.is_available() and rank is not None and rank >= 0:
        device = torch.device("cuda", rank)
    else:
        device = torch.device("cpu")
    for name in sorted(meters.keys()):
        meter = meters[name]
        if not isinstance(meter, AverageMeter):
            raise TypeError("meter should be AverageMeter type")
        stats = torch.tensor([meter.sum, meter.count], dtype=torch.float64, device=device)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        if is_rank_0():
            sum_reduce = stats[0].item()
            count_reduce = stats[1].item()
            avg_reduce = sum_reduce / max(count_reduce, 1e-12)
            meter.update_reduce_v2(avg_reduce, sum_reduce, count_reduce)

def _z3_params_to_fetch(param_list):
    return [
        p for p in param_list
        if hasattr(p, 'ds_id') and p.ds_status == ZeroParamStatus.NOT_AVAILABLE
    ]


def save_zero_three_model(model: torch.nn.Module, global_rank: int, save_dir: str, zero_stage: int = 0):
    """
    Saves the model checkpoint, handling DeepSpeed ZeRO Stage 3 parameter gathering.
    """
    zero_stage_3 = (zero_stage == 3)
    os.makedirs(save_dir, exist_ok=True)
    weights_name = "pytorch_model.bin"
    output_model_file = os.path.join(save_dir, weights_name)

    model_to_save = model.module if hasattr(model, 'module') else model

    if not zero_stage_3:
        if global_rank == 0:
            torch.save(model_to_save.state_dict(), output_model_file)
    else:
        # For ZeRO-3, we need to gather parameters
        output_state_dict = {}
        for k, v in model_to_save.named_parameters():
            if hasattr(v, 'ds_id'):
                with deepspeed.zero.GatheredParameters(_z3_params_to_fetch([v]), enabled=zero_stage_3):
                    v_p = v.data.cpu()
            else:
                v_p = v.cpu()
            
            # TODO: 
            # changes: Lora handling removed as it wasn't in imports, 
            # Original: if global_rank == 0 and "lora" not in k:
            if global_rank == 0 and "lora" not in k:
                output_state_dict[k] = v_p
        
        if global_rank == 0:
            torch.save(output_state_dict, output_model_file)
        del output_state_dict
