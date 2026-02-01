import logging
import os
import sys
from typing import Optional

from .distributed import is_rank_0

def setup_logger(name: str = "csref", output_dir: Optional[str] = None, rank: int = 0) -> logging.Logger:
    """
    Sets up a logger that prints to console and optionally to a file.
    Only rank 0 logs to console/file by default, but this can be configured.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Console Handler (only for rank 0 to avoid clutter)
    if rank == 0:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # File Handler
    if output_dir and rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, "train.log")
        fh = logging.FileHandler(filename)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
