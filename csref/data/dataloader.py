from bisect import bisect_right
import torch
import torch.distributed as dist
from torch.utils.data import DistributedSampler, SequentialSampler, DataLoader
from typing import Iterable, TypeVar, List, Tuple
import numpy as np

from ..utils.logging_utils import setup_logger

T = TypeVar("T")
logger = setup_logger(__name__)

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


class LengthBucketDistributedSampler(DistributedSampler):
    """Distributed sampler that groups samples by length buckets when metadata is available."""

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        bucket_boundaries: List[int],
        num_replicas: int,
        rank: int,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ):
        super().__init__(
            dataset=dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )
        self.bucket_boundaries = sorted(int(x) for x in bucket_boundaries if int(x) > 0)

    def _get_length(self, index: int):
        if not hasattr(self.dataset, "get_num_samples"):
            return None
        return self.dataset.get_num_samples(index)

    def __iter__(self):
        if not self.shuffle:
            return super().__iter__()

        indices = list(range(len(self.dataset)))
        if len(indices) == 0 or len(self.bucket_boundaries) == 0:
            return super().__iter__()

        lengths = []
        for idx in indices:
            num_samples = self._get_length(idx)
            if num_samples is None:
                # Fallback when metadata is missing.
                return super().__iter__()
            lengths.append(int(num_samples))

        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)

        buckets = {}
        for idx, num_samples in zip(indices, lengths):
            bucket_id = bisect_right(self.bucket_boundaries, num_samples)
            buckets.setdefault(bucket_id, []).append(idx)

        ordered = []
        bucket_ids = list(buckets.keys())
        if len(bucket_ids) > 1:
            bucket_perm = torch.randperm(len(bucket_ids), generator=generator).tolist()
            bucket_ids = [bucket_ids[i] for i in bucket_perm]

        for bucket_id in bucket_ids:
            bucket_indices = buckets[bucket_id]
            if len(bucket_indices) > 1:
                idx_perm = torch.randperm(len(bucket_indices), generator=generator).tolist()
                bucket_indices = [bucket_indices[i] for i in idx_perm]
            ordered.extend(bucket_indices)

        if not self.drop_last:
            padding_size = self.total_size - len(ordered)
            if padding_size > 0:
                repeats = (padding_size + len(ordered) - 1) // len(ordered)
                ordered += (ordered * repeats)[:padding_size]
            else:
                ordered = ordered[:self.total_size]
        else:
            ordered = ordered[:self.total_size]

        indices = ordered[self.rank:self.total_size:self.num_replicas]
        return iter(indices)


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

    data_cfg = cfg.train.data
    seed = cfg.train.seed if hasattr(cfg.train, "seed") else 0
    use_length_bucket = bool(getattr(data_cfg, "use_length_bucket", False))
    bucket_boundaries_sec = getattr(data_cfg, "bucket_boundaries_second", None)
    bucket_boundaries_samples: List[int] = []
    if use_length_bucket and bucket_boundaries_sec is not None:
        target_sr = int(cfg.dataset.target_sample_rate)
        for sec in bucket_boundaries_sec:
            try:
                sec_float = float(sec)
            except (TypeError, ValueError):
                continue
            if sec_float > 0:
                bucket_boundaries_samples.append(int(sec_float * target_sr))
        bucket_boundaries_samples = sorted(set(bucket_boundaries_samples))

    if use_length_bucket and bucket_boundaries_samples:
        sampler = LengthBucketDistributedSampler(
            dataset=dataset,
            bucket_boundaries=bucket_boundaries_samples,
            num_replicas=num_tasks,
            rank=global_rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )
        if global_rank == 0:
            logger.info(f"Train sampler: LengthBucketDistributedSampler boundaries(samples)={bucket_boundaries_samples}")
    else:
        sampler = DistributedSampler(
            dataset,
            num_replicas=num_tasks,
            rank=global_rank,
            shuffle=shuffle, # seed is handled by DistributedSampler via set_epoch
            seed=seed
        )

    num_workers = int(data_cfg.num_workers)
    pin_memory = bool(data_cfg.pin_memory)
    prefetch_factor = getattr(data_cfg, "prefetch_factor", None)
    persistent_workers = bool(getattr(data_cfg, "persistent_workers", False))

    loader_kwargs = dict(
        dataset=dataset,
        batch_size=micro_batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn,
    )
    if num_workers > 0:
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = int(prefetch_factor)
        loader_kwargs["persistent_workers"] = persistent_workers

    data_loader = DataLoader(**loader_kwargs)
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

    num_workers = int(cfg.train.data.num_workers)
    pin_memory = bool(cfg.train.data.pin_memory)
    prefetch_factor = getattr(cfg.train.data, "prefetch_factor", None)
    persistent_workers = bool(getattr(cfg.train.data, "persistent_workers", False))

    loader_kwargs = dict(
        dataset=dataset,
        batch_size=eval_micro_batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn,
    )
    if num_workers > 0:
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = int(prefetch_factor)
        loader_kwargs["persistent_workers"] = persistent_workers

    data_loader = DataLoader(**loader_kwargs)
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
