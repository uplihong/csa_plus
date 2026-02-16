import os
import sys
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from csref.core.trainer import Trainer
from csref.utils.distributed import seed_everything, is_rank_0
from csref.utils.logging_utils import setup_logger, configure_project_loggers

logger = setup_logger(__name__)


def _normalize_local_rank_arg() -> None:
    """Convert DeepSpeed/torch launcher local-rank CLI args into Hydra-compatible override."""
    local_rank = None
    cleaned_argv = [sys.argv[0]]
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg.startswith("--local_rank="):
            local_rank = arg.split("=", 1)[1]
        elif arg == "--local_rank":
            if i + 1 < len(sys.argv):
                local_rank = sys.argv[i + 1]
                i += 1
        elif arg.startswith("--local-rank="):
            local_rank = arg.split("=", 1)[1]
        elif arg == "--local-rank":
            if i + 1 < len(sys.argv):
                local_rank = sys.argv[i + 1]
                i += 1
        else:
            cleaned_argv.append(arg)
        i += 1

    if local_rank is not None:
        os.environ["LOCAL_RANK"] = str(local_rank)
        if not any(arg.startswith("local_rank=") for arg in cleaned_argv[1:]):
            cleaned_argv.append(f"local_rank={local_rank}")

    sys.argv = cleaned_argv


def _configure_math_backend(cfg: DictConfig) -> None:
    requested_tf32 = bool(getattr(cfg.train, "enable_tf32", True))
    matmul_precision = str(getattr(cfg.train, "matmul_precision", "high")).lower()
    if matmul_precision not in {"high", "medium", "highest"}:
        logger.warning("Invalid train.matmul_precision=%s; fallback to high", matmul_precision)
        matmul_precision = "high"

    effective_tf32 = requested_tf32
    if requested_tf32:
        if not torch.cuda.is_available():
            logger.warning("train.enable_tf32=true but CUDA is unavailable; force disable TF32.")
            effective_tf32 = False
        else:
            cc_major, _ = torch.cuda.get_device_capability(0)
            if cc_major < 8:
                logger.warning("train.enable_tf32=true but GPU capability is < 8.0; force disable TF32.")
                effective_tf32 = False

    try:
        torch.backends.cuda.matmul.allow_tf32 = bool(effective_tf32)
        torch.backends.cudnn.allow_tf32 = bool(effective_tf32)
    except Exception as exc:
        logger.warning("Failed to set TF32 backend flags: %s", exc)

    try:
        torch.set_float32_matmul_precision(matmul_precision)
    except Exception as exc:
        logger.warning("Failed to set float32 matmul precision=%s: %s", matmul_precision, exc)

    cfg.train.enable_tf32 = bool(effective_tf32)
    cfg.train.matmul_precision = matmul_precision

@hydra.main(config_path="configs", config_name="config_csa_plus", version_base="1.3.2")
def main(cfg: DictConfig):
    if "LOCAL_RANK" in os.environ:
        cfg.local_rank = int(os.environ["LOCAL_RANK"])
    elif not hasattr(cfg, "local_rank"):
        cfg.local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    process_rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", cfg.local_rank)))
    log_path = configure_project_loggers(cfg.experiment_output_dir, rank=process_rank)
    if is_rank_0():
        logger.info(f"Logging to file: {log_path}")

    _configure_math_backend(cfg)

    # Setup global seed
    if hasattr(cfg.train, 'seed'):
        seed_everything(
            cfg.train.seed,
            deterministic=getattr(cfg.train, "deterministic", None),
            cudnn_benchmark=getattr(cfg.train, "cudnn_benchmark", None),
        )
    elif hasattr(cfg.train, "deterministic") or hasattr(cfg.train, "cudnn_benchmark"):
        deterministic = getattr(cfg.train, "deterministic", None)
        cudnn_benchmark = getattr(cfg.train, "cudnn_benchmark", None)
        if deterministic is not None:
            torch.backends.cudnn.deterministic = bool(deterministic)
        if cudnn_benchmark is not None:
            torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
        if torch.backends.cudnn.deterministic and torch.backends.cudnn.benchmark:
            torch.backends.cudnn.benchmark = False
    
    # Debug info
    if is_rank_0():
        logger.info(
            "Math backend: enable_tf32=%s, matmul_precision=%s",
            bool(getattr(cfg.train, "enable_tf32", False)),
            str(getattr(cfg.train, "matmul_precision", "high")),
        )
        logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    trainer = Trainer(cfg)
    trainer.setup()
    trainer.fit()

if __name__ == "__main__":
    _normalize_local_rank_arg()
    main()
