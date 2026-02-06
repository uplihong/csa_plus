import os
import time
from collections import deque
import torch
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

        if hasattr(self.cfg.train, "min_max_audio_second") and self.cfg.train.min_max_audio_second:
            min_max = self.cfg.train.min_max_audio_second
            if len(min_max) != 2:
                raise ValueError("train.min_max_audio_second must be a 2-element list [min_sec, max_sec]")
            min_sec, max_sec = min_max
            if max_sec < min_sec:
                raise ValueError("train.min_max_audio_second max must be >= min")
            target_sr = self.cfg.dataset.target_sample_rate
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

        params_require_grad = filter(lambda p: p.requires_grad, model.parameters())
        
        # Contiguous check
        for param in model.parameters():
            param.data = param.data.contiguous()

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
            if i % self.cfg.train.log_every_steps == 0 and is_rank_0():
                 logger.info(f"Step {i}: Loss {loss:.4f}")
                 logger.info(f"Timing(window={self.timing_window}): {self._timing_summary()}")
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
        preprocess_ms = (time.perf_counter() - preprocess_t0) * 1000.0

        # Forward
        fwd_t0 = time.perf_counter()
        loss = self.model_engine(audio_input, audio_mask, text_input, text_mask)
        fwd_ms = (time.perf_counter() - fwd_t0) * 1000.0

        # Backward & Step
        bwd_t0 = time.perf_counter()
        self.model_engine.backward(loss)
        bwd_ms = (time.perf_counter() - bwd_t0) * 1000.0

        step_t0 = time.perf_counter()
        self.model_engine.step()
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
