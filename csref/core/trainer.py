import os
import time
import math
from collections import deque
import torch
import torch.distributed as dist
import deepspeed
from typing import Optional, Any, Dict
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from torch.utils.data import DataLoader

from ..utils.logging_utils import setup_logger
from ..utils.distributed import is_rank_0, save_zero_three_model, reduce_meters, get_rank
from ..utils.metric import AverageMeter
from ..data.dataloader import InfiniteIterator
from ..data.transforms import RandomAudioSlice

logger = setup_logger(__name__)

class Trainer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.local_rank = int(os.environ.get("LOCAL_RANK", -1))
        
        # Setup model variables
        self.model_engine = None
        self.optimizer = None
        self.train_dataloader = None
        self.val_dataloader = None
        
        # State
        self.global_step = 0
        self.start_epoch = 0
        self.timing_window = int(getattr(cfg.train, "timing_window", 100))
        self.enable_cuda_sync_timing = bool(getattr(cfg.train, "enable_cuda_sync_timing", False))
        self.timing_rank_scope = str(getattr(cfg.train, "timing_rank_scope", "rank0")).lower()
        if self.timing_rank_scope not in {"rank0", "all"}:
            raise ValueError("train.timing_rank_scope must be one of: rank0, all")
        self.enable_eta_logging = bool(getattr(cfg.train, "enable_eta_logging", True))
        self.eta_distributed_mode = str(getattr(cfg.train, "eta_distributed_mode", "rank0")).lower()
        if self.eta_distributed_mode not in {"rank0", "global_max"}:
            raise ValueError("train.eta_distributed_mode must be one of: rank0, global_max")
        self.eta_min_samples = int(getattr(cfg.train, "eta_min_samples", 10))
        if self.eta_min_samples <= 0:
            raise ValueError("train.eta_min_samples must be > 0")
        self.enable_torch_compile = bool(getattr(cfg.train, "enable_torch_compile", False))
        self.torch_compile_mode = str(getattr(cfg.train, "torch_compile_mode", "max-autotune"))
        self.torch_compile_dynamic = bool(getattr(cfg.train, "torch_compile_dynamic", True))
        fixed_slice = getattr(cfg.train, "fixed_slice_seconds", None)
        self.fixed_slice_seconds: Optional[float] = None
        if fixed_slice is not None:
            try:
                self.fixed_slice_seconds = float(fixed_slice)
            except (TypeError, ValueError) as exc:
                raise ValueError("train.fixed_slice_seconds must be float or null") from exc
            if self.fixed_slice_seconds <= 0:
                raise ValueError("train.fixed_slice_seconds must be > 0")
        self.step_timing_history = {
            "data_wait_ms": deque(maxlen=self.timing_window),
            "preprocess_ms": deque(maxlen=self.timing_window),
            "fwd_ms": deque(maxlen=self.timing_window),
            "bwd_ms": deque(maxlen=self.timing_window),
            "step_ms": deque(maxlen=self.timing_window),
            "iter_ms": deque(maxlen=self.timing_window),
        }
        
        # Tools
        self.audio_preprocessor = instantiate(cfg.audio_preprocessor)
        self.text_tokenizer = instantiate(cfg.text_preprocessor)
        
        # Tensorboard
        self.writer = None
        if is_rank_0() and hasattr(cfg.train, 'tensorboard_dir'):
            from tensorboardX import SummaryWriter
            self.writer = SummaryWriter(log_dir=cfg.train.tensorboard_dir)
            
        # Determine Training Precision
        self.dtype = torch.float32
        ds_conf = OmegaConf.to_container(self.cfg.deepspeed_config_yaml, resolve=True) if hasattr(self.cfg, 'deepspeed_config_yaml') else {}
        if ds_conf.get('fp16', {}).get('enabled', False):
            self.dtype = torch.float16
        elif ds_conf.get('bf16', {}).get('enabled', False):
            self.dtype = torch.bfloat16

    def setup(self):
        """Initializes model, optimizer, dataloaders, and DeepSpeed."""
        # 1. Dataset
        logger.info("Setting up datasets...")
        self.cfg.dataset.train_split = "train"
        train_set = instantiate(self.cfg.dataset)

        target_sr = self.cfg.dataset.target_sample_rate
        if self.fixed_slice_seconds is not None:
            fixed_len = int(self.fixed_slice_seconds * target_sr)
            train_set.transform = RandomAudioSlice(fixed_len, fixed_len, sample_rate=target_sr)
            if is_rank_0():
                logger.info(f"Apply fixed RandomAudioSlice: {self.fixed_slice_seconds:.3f}s @ {target_sr}Hz")
        elif hasattr(self.cfg.train, "min_max_audio_second") and self.cfg.train.min_max_audio_second:
            min_max = self.cfg.train.min_max_audio_second
            if len(min_max) != 2:
                raise ValueError("train.min_max_audio_second must be a 2-element list [min_sec, max_sec]")
            min_sec, max_sec = min_max
            if max_sec < min_sec:
                raise ValueError("train.min_max_audio_second max must be >= min")
            min_len = int(min_sec * target_sr)
            max_len = int(max_sec * target_sr)
            if min_len <= 0 or max_len <= 0:
                raise ValueError("train.min_max_audio_second must be > 0")
            train_set.transform = RandomAudioSlice(min_len, max_len, sample_rate=target_sr)
            if is_rank_0():
                logger.info(f"Apply RandomAudioSlice: {min_sec}-{max_sec}s @ {target_sr}Hz")
        
        self.cfg.dataset.train_split = "val"
        val_set = instantiate(self.cfg.dataset)

        if len(train_set) == 0:
            raise RuntimeError(f"Train dataset is empty. Please check dataset.root_dir: {self.cfg.dataset.root_dir}")
        if len(val_set) == 0:
            raise RuntimeError(f"Validation dataset is empty. Please check dataset.root_dir: {self.cfg.dataset.root_dir}")
        
        # 2. Dataloaders - Built later during DeepSpeed init or manually
        # Note: DeepSpeed initialize can take training_data but we might want custom loader
        # functionality (like build_train_librispeech_loader).
        # We will pass the dataset to deepspeed if we want it to handle it, or pass None 
        # and handle loader creation ourselves. Original code did both.
        # Original: deepspeed.initialize(..., training_data=train_set, collate_fn=collate_fn)
        # But also: trainloader = build_train_librispeech_loader(...)
        
        # 3. Model
        logger.info("Setting up model...")
        model = instantiate(self.cfg.model)
        
        # Load pretrained weights if specified
        if hasattr(self.cfg.train, 'pretrained_model_checkpoint') and self.cfg.train.pretrained_model_checkpoint:
            self._load_pretrained_weights(model, self.cfg.train.pretrained_model_checkpoint)

        # Contiguous check
        for param in model.parameters():
            param.data = param.data.contiguous()

        model = self._maybe_compile_model(model)
        params_require_grad = filter(lambda p: p.requires_grad, model.parameters())

        # 4. DeepSpeed Setup
        logger.info("Initializing DeepSpeed...")
        ds_config = OmegaConf.to_container(self.cfg.deepspeed_config_yaml, resolve=True)
        
        # DeepSpeed initialization
        # We use the collate_fn from the dataloader module if we let DS build the loader
        # But we want to use our custom dataloader builder for consistency.
        from ..data.dataloader import collate_fn, build_train_librispeech_loader, build_test_librispeech_loader
        
        self.model_engine, self.optimizer, _, _ = deepspeed.initialize(
            args=self.cfg,
            model=model,
            model_parameters=params_require_grad,
            config=ds_config
        )
        
        # Manually build loaders to ensure we control the sampler/batching logic exactly as we want
        # Note: If we passed training_data to deepspeed.initialize, it returns a loader. 
        # But we passed None for training_data above (implicit), so we build it here.
        self.train_dataloader = build_train_librispeech_loader(
            self.cfg, train_set, shuffle=True, drop_last=False
        )
        self.val_dataloader = build_test_librispeech_loader(
            self.cfg, val_set, shuffle=False, drop_last=False
        )
        
        if is_rank_0():
            logger.info("DeepSpeed initialized.")
            logger.info(f"Train set: {len(train_set)}, Val set: {len(val_set)}")
            logger.info(
                "Timing options: enable_cuda_sync_timing=%s, timing_rank_scope=%s",
                self.enable_cuda_sync_timing,
                self.timing_rank_scope,
            )
            logger.info(
                "ETA options: enable_eta_logging=%s, eta_distributed_mode=%s, eta_min_samples=%s",
                self.enable_eta_logging,
                self.eta_distributed_mode,
                self.eta_min_samples,
            )
            logger.info(
                "Compile options: enable_torch_compile=%s, mode=%s, dynamic=%s",
                self.enable_torch_compile,
                self.torch_compile_mode,
                self.torch_compile_dynamic,
            )
            logger.info("Data slicing: fixed_slice_seconds=%s", self.fixed_slice_seconds)

    def _maybe_compile_model(self, model):
        if not self.enable_torch_compile:
            return model
        if not hasattr(torch, "compile"):
            logger.warning("torch.compile not available in current torch version. Falling back to eager model.")
            return model

        try:
            compiled_model = torch.compile(
                model,
                mode=self.torch_compile_mode,
                dynamic=self.torch_compile_dynamic,
            )
            logger.info(
                "Enabled torch.compile with mode=%s dynamic=%s",
                self.torch_compile_mode,
                self.torch_compile_dynamic,
            )
            return compiled_model
        except Exception as exc:
            logger.warning("torch.compile failed, falling back to eager model: %s", exc)
            return model

    def _load_pretrained_weights(self, model, checkpoint_path):
        logger.info(f"Loading pretrained weights from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location='cpu')['state_dict']
        
        # CSA specific logic from original code: mapping audio_encoder to speech_encoder
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("audio_encoder."):
                k = k.replace("audio_encoder", "speech_encoder", 1)
                new_state_dict[k] = v
            # ONLY audio_encoder parts were loaded
            # text encoder initialized from HF Bert.
        
        if len(new_state_dict) > 0:
            msg = model.load_state_dict(new_state_dict, strict=False)
            if is_rank_0():
                logger.info(f"Loaded {len(new_state_dict)} layers. Message: {msg}")

    def fit(self):
        """Main training loop."""
        self.model_engine.train()
        
        # Support infinite iterator or epoch-based
        max_steps = self.cfg.train.get('max_step_iterations', 100000)
        iterator = iter(InfiniteIterator(self.train_dataloader))
        
        logger.info("Starting training...")

        i = 0
        while True:
            self.global_step = i
            
            if i > max_steps:
                logger.info("Reached max steps. Stopping.")
                break

            data_wait_t0 = time.perf_counter()
            audio_list, text_list = next(iterator)
            data_wait_ms = (time.perf_counter() - data_wait_t0) * 1000.0

            loss, step_timing = self.train_step(audio_list, text_list, return_timing=True)
            step_timing["data_wait_ms"] = data_wait_ms
            step_timing["iter_ms"] = (
                step_timing["data_wait_ms"]
                + step_timing["preprocess_ms"]
                + step_timing["fwd_ms"]
                + step_timing["bwd_ms"]
                + step_timing["step_ms"]
            )
            self._record_step_timing(step_timing)
            
            # Logging
            should_log = (i % self.cfg.train.log_every_steps == 0)
            rank_snapshots = None
            eta_snapshot = None
            if should_log:
                rank_snapshots = self._gather_rank_timing_snapshot(step=i, loss=loss, step_timing=step_timing)
                if self.enable_eta_logging:
                    eta_basis_iter_ms = float(step_timing["iter_ms"])
                    if self.eta_distributed_mode == "global_max":
                        eta_basis_iter_ms = self._eta_allreduce_max(eta_basis_iter_ms)
                    eta_snapshot = self._build_eta_snapshot(
                        step=i,
                        max_steps=max_steps,
                        iter_ms_sample=eta_basis_iter_ms,
                    )

            if should_log and is_rank_0():
                logger.info(f"Step {i}: Loss {loss:.4f}")
                logger.info(f"Timing(window={self.timing_window}): {self._timing_summary()}")
                if eta_snapshot is not None:
                    logger.info(
                        "TimingETA step=%d remain_steps=%d eta_sec=%.1f eta_hms=%s mode=%s basis_iter_ms=%.2f samples=%d",
                        eta_snapshot["step"],
                        eta_snapshot["remain_steps"],
                        eta_snapshot["eta_sec"],
                        eta_snapshot["eta_hms"],
                        eta_snapshot["mode"],
                        eta_snapshot["basis_iter_ms"],
                        eta_snapshot["samples"],
                    )
                if rank_snapshots is not None:
                    for item in rank_snapshots:
                        logger.info(
                            "TimingRank step=%d rank=%d loss=%.4f data_wait=%.2f preprocess=%.2f fwd=%.2f bwd=%.2f step=%.2f iter=%.2f",
                            i,
                            item["rank"],
                            item["loss"],
                            item["data_wait_ms"],
                            item["preprocess_ms"],
                            item["fwd_ms"],
                            item["bwd_ms"],
                            item["step_ms"],
                            item["iter_ms"],
                        )
                if self.writer:
                    self.writer.add_scalar("Train/Loss", loss, i)

            # Checkpointing
            if i % self.cfg.train.checkpoint_every_steps == 0 and i > 0:
                self.save_checkpoint(i)

            # Validation
            if i % self.cfg.train.validation_every_steps == 0 and i > 0:
                self.validate(i)

            i += 1

    def _record_step_timing(self, timings: Dict[str, float]):
        for key, value in timings.items():
            if key in self.step_timing_history:
                self.step_timing_history[key].append(float(value))

    @staticmethod
    def _percentile(values, q):
        if not values:
            return 0.0
        ordered = sorted(values)
        if len(ordered) == 1:
            return float(ordered[0])
        pos = (len(ordered) - 1) * (q / 100.0)
        left = int(pos)
        right = min(left + 1, len(ordered) - 1)
        frac = pos - left
        return float(ordered[left] * (1 - frac) + ordered[right] * frac)

    def _timing_summary(self) -> str:
        metrics = []
        for key in ["data_wait_ms", "preprocess_ms", "fwd_ms", "bwd_ms", "step_ms", "iter_ms"]:
            values = list(self.step_timing_history[key])
            p50 = self._percentile(values, 50)
            p90 = self._percentile(values, 90)
            metrics.append(f"{key}:p50={p50:.2f} p90={p90:.2f}")
        return " | ".join(metrics)

    def _synchronize_cuda_for_timing(self, device: torch.device):
        if self.enable_cuda_sync_timing and torch.cuda.is_available():
            torch.cuda.synchronize(device)

    def _eta_allreduce_max(self, value_ms: float) -> float:
        if not dist.is_available() or not dist.is_initialized():
            return float(value_ms)
        backend = dist.get_backend()
        if backend == "nccl" and torch.cuda.is_available():
            if self.model_engine is not None:
                device = self.model_engine.device
            else:
                device = torch.device("cuda", torch.cuda.current_device())
        else:
            device = torch.device("cpu")
        value_tensor = torch.tensor(float(value_ms), dtype=torch.float64, device=device)
        dist.all_reduce(value_tensor, op=dist.ReduceOp.MAX)
        return float(value_tensor.item())

    @staticmethod
    def _format_eta_hms(seconds: float) -> str:
        if seconds <= 0:
            return "00:00:00"
        total_seconds = int(math.ceil(seconds))
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def _build_eta_snapshot(self, step: int, max_steps: int, iter_ms_sample: float):
        try:
            sample_ms = float(iter_ms_sample)
        except (TypeError, ValueError):
            return None
        if sample_ms <= 0:
            return None
        iter_history = list(self.step_timing_history["iter_ms"])
        sample_count = len(iter_history)
        if sample_count < self.eta_min_samples:
            return None
        remaining_steps = max(int(max_steps) - int(step), 0)
        mean_iter_ms = float(sum(iter_history) / sample_count)
        if self.eta_distributed_mode == "global_max":
            mean_iter_ms = max(mean_iter_ms, sample_ms)
        eta_sec = (remaining_steps * mean_iter_ms) / 1000.0
        return {
            "step": int(step),
            "remain_steps": int(remaining_steps),
            "eta_sec": float(eta_sec),
            "eta_hms": self._format_eta_hms(float(eta_sec)),
            "mode": self.eta_distributed_mode,
            "basis_iter_ms": float(sample_ms),
            "samples": int(sample_count),
        }

    def _gather_rank_timing_snapshot(self, step: int, loss: float, step_timing: Dict[str, float]):
        if self.timing_rank_scope != "all":
            return None
        if not dist.is_available() or not dist.is_initialized():
            return None

        payload = {
            "step": int(step),
            "rank": int(dist.get_rank()),
            "loss": float(loss),
            "data_wait_ms": float(step_timing["data_wait_ms"]),
            "preprocess_ms": float(step_timing["preprocess_ms"]),
            "fwd_ms": float(step_timing["fwd_ms"]),
            "bwd_ms": float(step_timing["bwd_ms"]),
            "step_ms": float(step_timing["step_ms"]),
            "iter_ms": float(step_timing["iter_ms"]),
        }
        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, payload)
        if not is_rank_0():
            return None
        return sorted(gathered, key=lambda item: item["rank"])

    def train_step(self, audio_list, text_list, return_timing: bool = False):
        # Data Processing
        # Apply processor/tokenizer
        # Note: audio_preprocessor is Wav2Vec2Processor or FeatureExtractor
        device = self.model_engine.device

        preprocess_t0 = time.perf_counter()
        batch_audio = self.audio_preprocessor(
            raw_speech=audio_list, 
            padding=True, 
            max_length=None, 
            truncation=False,
            pad_to_multiple_of=None, 
            return_attention_mask=True, 
            return_tensors="pt",
            sampling_rate=self.cfg.dataset.target_sample_rate
        )
        
        batch_text = self.text_tokenizer(
            text_list,
            padding=True,
            truncation=True,
            max_length=None,
            return_tensors='pt',
            return_attention_mask=True
        )

        audio_input = batch_audio.input_values.to(device, dtype=self.dtype, non_blocking=True)
        audio_mask = batch_audio.attention_mask.to(device, non_blocking=True)
        
        text_input = batch_text.input_ids.to(device, non_blocking=True)
        text_mask = batch_text.attention_mask.to(device, non_blocking=True)
        self._synchronize_cuda_for_timing(device)
        preprocess_ms = (time.perf_counter() - preprocess_t0) * 1000.0

        # Forward
        self._synchronize_cuda_for_timing(device)
        fwd_t0 = time.perf_counter()
        loss = self.model_engine(audio_input, audio_mask, text_input, text_mask)
        self._synchronize_cuda_for_timing(device)
        fwd_ms = (time.perf_counter() - fwd_t0) * 1000.0

        # Backward & Step
        self._synchronize_cuda_for_timing(device)
        bwd_t0 = time.perf_counter()
        self.model_engine.backward(loss)
        self._synchronize_cuda_for_timing(device)
        bwd_ms = (time.perf_counter() - bwd_t0) * 1000.0

        self._synchronize_cuda_for_timing(device)
        step_t0 = time.perf_counter()
        self.model_engine.step()
        self._synchronize_cuda_for_timing(device)
        step_ms = (time.perf_counter() - step_t0) * 1000.0

        loss_value = loss.item()
        if return_timing:
            return loss_value, {
                "preprocess_ms": preprocess_ms,
                "fwd_ms": fwd_ms,
                "bwd_ms": bwd_ms,
                "step_ms": step_ms,
            }
        return loss_value

    def validate(self, step):
        logger.info("Starting validation...")
        self.model_engine.eval()
        
        losses = AverageMeter('Loss', ':.4f')
        batch_time = AverageMeter('Time', ':6.5f')
        end = time.time()
        
        with torch.no_grad():
            for idx, (audio_list, text_list) in enumerate(self.val_dataloader):
                device = self.model_engine.device
                
                # Check for empty batch
                if len(audio_list) == 0: continue

                # Process
                batch_audio = self.audio_preprocessor(
                    raw_speech=audio_list, 
                    padding=True, 
                    return_attention_mask=True, 
                    return_tensors="pt",
                    sampling_rate=self.cfg.dataset.target_sample_rate
                )
                batch_text = self.text_tokenizer(
                    text_list,
                    padding=True,
                    truncation=True,
                    return_tensors='pt',
                    return_attention_mask=True
                )
                
                audio_input = batch_audio.input_values.to(device, dtype=self.dtype, non_blocking=True)
                audio_mask = batch_audio.attention_mask.to(device, non_blocking=True)
                text_input = batch_text.input_ids.to(device, non_blocking=True)
                text_mask = batch_text.attention_mask.to(device, non_blocking=True)
                
                # Forward
                loss = self.model_engine(audio_input, audio_mask, text_input, text_mask)
                
                losses.update(loss.item(), audio_input.size(0))
                batch_time.update(time.time() - end)
                end = time.time()
                
                if idx % 10 == 0 and is_rank_0():
                    logger.info(f"Val Step {idx}/{len(self.val_dataloader)} Loss {losses.val:.4f}")

        # Reduce metrics across ranks
        meters_dict = {'Loss': losses}
        reduce_meters(meters_dict, self.local_rank, self.cfg)

        if is_rank_0():
            logger.info(f"Validation Finished for Step {step}. Avg Loss: {losses.avg_reduce:.4f}")
            if self.writer:
                self.writer.add_scalar("Val/Loss", losses.avg_reduce, step)

        self.model_engine.train()

    def save_checkpoint(self, step):
        logger.info(f"Saving checkpoint at step {step}...")
        save_dir = os.path.join(self.cfg.experiment_output_dir, f"iter_{step}")
        
        ds_config = self.cfg.deepspeed_config_yaml
        zero_stage = ds_config.get('zero_optimization', {}).get('stage', 0)
        global_rank = get_rank()
        
        if zero_stage == 3:
            save_zero_three_model(self.model_engine, global_rank, save_dir, zero_stage=3)
        else:
            self.model_engine.save_checkpoint(save_dir=save_dir, client_state={'step': step})
