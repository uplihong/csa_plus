import os
import sys
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from csref.core.trainer import Trainer
from csref.utils.distributed import seed_everything, is_rank_0
from csref.utils.logging_utils import setup_logger, configure_project_loggers
from csref.utils.startup_metadata import collect_startup_metadata, write_startup_artifacts

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

    startup_metadata_enabled = bool(getattr(cfg.train, "startup_metadata_enabled", True))

    startup_config_filename = str(
        getattr(cfg.train, "startup_metadata_config_filename", "resolved_config.yaml")
    ).strip()
    if not startup_config_filename:
        logger.warning("train.startup_metadata_config_filename is empty; fallback to resolved_config.yaml")
        startup_config_filename = "resolved_config.yaml"

    startup_context_filename = str(
        getattr(cfg.train, "startup_metadata_context_filename", "run_context.json")
    ).strip()
    if not startup_context_filename:
        logger.warning("train.startup_metadata_context_filename is empty; fallback to run_context.json")
        startup_context_filename = "run_context.json"

    startup_git_max_status_lines_raw = getattr(cfg.train, "startup_metadata_git_max_status_lines", 200)
    try:
        startup_git_max_status_lines = int(startup_git_max_status_lines_raw)
    except (TypeError, ValueError):
        logger.warning(
            "Invalid train.startup_metadata_git_max_status_lines=%s; fallback to 200",
            startup_git_max_status_lines_raw,
        )
        startup_git_max_status_lines = 200
    if startup_git_max_status_lines < 0:
        logger.warning(
            "train.startup_metadata_git_max_status_lines=%s is invalid; fallback to 0",
            startup_git_max_status_lines,
        )
        startup_git_max_status_lines = 0
    
    # Debug info
    if is_rank_0():
        logger.info(
            "Math backend: enable_tf32=%s, matmul_precision=%s",
            bool(getattr(cfg.train, "enable_tf32", False)),
            str(getattr(cfg.train, "matmul_precision", "high")),
        )
        logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

        if startup_metadata_enabled:
            startup_context = collect_startup_metadata(
                process_rank=process_rank,
                local_rank=int(cfg.local_rank),
                world_size=int(os.environ.get("WORLD_SIZE", "1")),
                experiment_output_dir=cfg.experiment_output_dir,
                repo_root=os.path.dirname(os.path.abspath(__file__)),
                max_git_status_lines=startup_git_max_status_lines,
            )
            artifact_paths = write_startup_artifacts(
                output_dir=cfg.experiment_output_dir,
                cfg=cfg,
                run_context=startup_context,
                config_filename=startup_config_filename,
                context_filename=startup_context_filename,
                logger=logger,
            )
            if artifact_paths.get("resolved_config_path"):
                logger.info("Startup metadata written: %s", artifact_paths["resolved_config_path"])
            if artifact_paths.get("run_context_path"):
                logger.info("Startup metadata written: %s", artifact_paths["run_context_path"])

            git_info = startup_context.get("git", {})
            if not git_info.get("available", False):
                logger.warning(
                    "Git metadata is unavailable at launch: %s",
                    git_info.get("error", "unknown"),
                )
            elif git_info.get("error"):
                logger.warning(
                    "Git metadata collection completed with warning: %s",
                    git_info.get("error"),
                )
            if git_info.get("dirty") is True:
                dirty_files = len(git_info.get("status_porcelain") or [])
                logger.warning(
                    "Git working tree is dirty at launch (branch=%s commit=%s changed_files=%s). Training will continue.",
                    git_info.get("branch", "unknown"),
                    git_info.get("commit_short", "unknown"),
                    dirty_files,
                )
        else:
            logger.info("Startup metadata recording is disabled: train.startup_metadata_enabled=false")
    
    trainer = Trainer(cfg)
    trainer.setup()
    trainer.fit()

if __name__ == "__main__":
    _normalize_local_rank_arg()
    main()
