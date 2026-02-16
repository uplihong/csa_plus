import logging
import os
import sys
from typing import Optional, Sequence


_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
_PROJECT_LOGGER_PREFIX = "csref"
_LOG_DIR_ENV = "CSREF_LOG_OUTPUT_DIR"


def _resolve_rank(rank: Optional[int] = None) -> int:
    if rank is not None:
        return int(rank)
    # Prefer global rank when available; fallback to local rank for pre-init launcher paths.
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))


def _iter_project_logger_names() -> Sequence[str]:
    names = {"__main__", "train", _PROJECT_LOGGER_PREFIX}
    manager = logging.Logger.manager
    for name, logger_obj in manager.loggerDict.items():
        if not isinstance(logger_obj, logging.Logger):
            continue
        if name == "__main__" or name.startswith(f"{_PROJECT_LOGGER_PREFIX}."):
            names.add(name)
    return sorted(names)


def _configure_single_logger(name: str, output_dir: Optional[str], rank: int) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Clear existing handlers so repeated reconfiguration stays deterministic.
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(_LOG_FORMAT)

    # Console handler (only rank 0 to avoid duplicate logs).
    if rank == 0:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # File handler (only rank 0).
    if output_dir and rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, "train.log")
        fh = logging.FileHandler(filename)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def setup_logger(name: str = _PROJECT_LOGGER_PREFIX, output_dir: Optional[str] = None, rank: Optional[int] = None) -> logging.Logger:
    """Setup a single logger with optional file output."""
    if output_dir is None:
        output_dir = os.environ.get(_LOG_DIR_ENV)
    resolved_rank = _resolve_rank(rank)
    return _configure_single_logger(name=name, output_dir=output_dir, rank=resolved_rank)


def configure_project_loggers(output_dir: Optional[str], rank: Optional[int] = None) -> str:
    """
    Reconfigure __main__ and all `csref.*` loggers after Hydra cfg is available.
    Returns the train log path (empty string if output_dir is not set).
    """
    resolved_rank = _resolve_rank(rank)
    if output_dir:
        os.environ[_LOG_DIR_ENV] = output_dir
    else:
        os.environ.pop(_LOG_DIR_ENV, None)

    for name in _iter_project_logger_names():
        _configure_single_logger(name=name, output_dir=output_dir, rank=resolved_rank)

    return os.path.join(output_dir, "train.log") if output_dir else ""
