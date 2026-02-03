import torch
import torch.distributed as dist
from torch.utils.data import DistributedSampler, SequentialSampler, DataLoader
from typing import Iterable, TypeVar, List, Tuple
import numpy as np

T = TypeVar("T")

# Collate function
def collate_fn(batch: List[Tuple[np.ndarray, str]]):
    """
    Collate function for LibriSpeech.
    Returns:
        audios: List of audio arrays (not padded yet, done by processor)
        transcripts: List of transcript strings
    """
    audios = [item[0] for item in batch]
    transcripts = [item[1] for item in batch]
    return [audios, transcripts]


def build_train_librispeech_loader(cfg, dataset: torch.utils.data.Dataset, shuffle: bool = True, drop_last: bool = False) -> DataLoader:
    # Handle distributed settings
    if dist.is_initialized():
        num_tasks = dist.get_world_size()
        global_rank = dist.get_rank()
    else:
        num_tasks = 1
        global_rank = 0

    # Determine batch size
    # Check if we are using DeepSpeed config or standard config
    if hasattr(cfg, 'deepspeed_config_yaml'):
        ds_cfg = cfg.deepspeed_config_yaml
        micro_batch_size = ds_cfg.get("train_micro_batch_size_per_gpu", None)
        if micro_batch_size is None:
            micro_batch_size = cfg.train.batch_size // num_tasks
    else:
        micro_batch_size = cfg.train.batch_size // num_tasks

    micro_batch_size = int(micro_batch_size)
    if micro_batch_size <= 0:
        raise ValueError(f"Invalid train micro batch size: {micro_batch_size}")

    sampler = DistributedSampler(
        dataset,
        num_replicas=num_tasks,
        rank=global_rank,
        shuffle=shuffle, # seed is handled by DistributedSampler via set_epoch
        seed=cfg.train.seed if hasattr(cfg.train, 'seed') else 0
    )

    data_loader = DataLoader(
        dataset,
        batch_size=micro_batch_size,
        sampler=sampler,
        num_workers=cfg.train.data.num_workers,
        pin_memory=cfg.train.data.pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn
    )
    return data_loader


def build_test_librispeech_loader(cfg, dataset: torch.utils.data.Dataset, shuffle: bool = False, drop_last: bool = False) -> DataLoader:
    # Handle distributed settings
    if dist.is_initialized():
        num_tasks = dist.get_world_size()
        global_rank = dist.get_rank()
    else:
        num_tasks = 1
        global_rank = 0

    eval_batch_size = int(cfg.train.evaluation.eval_batch_size)
    # Per GPU
    eval_micro_batch_size = max(1, eval_batch_size // num_tasks)

    if cfg.train.evaluation.sequential:
        # SequentialSampler doesn't support distributed split usually, assume run on rank 0 or full eval
        # Logic from original code: eval_micro_batch_size = cfg.train.evaluation.eval_batch_size
        # This implies running on single node or duplicated?
        # Let's keep original logic roughly but fix it.
        # If sequential is True, we probably want SequentialSampler.
        sampler = SequentialSampler(dataset)
        # But if distributed, we technically process same data on all ranks?
        # Original code did this. Let's assume user wants this behavior or we fix it.
        # Safe bet: If distributed, usually we want DistributedSampler(shuffle=False).
        # Let's respect the flag.
    else:
        sampler = DistributedSampler(
            dataset,
            num_replicas=num_tasks,
            rank=global_rank,
            shuffle=shuffle
        )

    data_loader = DataLoader(
        dataset,
        batch_size=eval_micro_batch_size,
        sampler=sampler,
        num_workers=cfg.train.data.num_workers,
        pin_memory=cfg.train.data.pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn
    )
    return data_loader


class InfiniteIterator:
    """
    A wrapper around a dataloader to return an infinite iterator.
    """
    def __init__(self, loader: DataLoader):
        self.loader = loader
        self.iterator = iter(self.loader)
        self.epoch = 0
        if hasattr(self.loader.sampler, "set_epoch"):
            self.loader.sampler.set_epoch(self.epoch)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.epoch += 1
            if hasattr(self.loader.sampler, "set_epoch"):
                self.loader.sampler.set_epoch(self.epoch)
            self.iterator = iter(self.loader)
            batch = next(self.iterator)
        return batch
