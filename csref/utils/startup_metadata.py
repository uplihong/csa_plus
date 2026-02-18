import json
import logging
import os
import platform
import shlex
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import torch
from omegaconf import DictConfig, OmegaConf

try:
    import deepspeed
except Exception:  # pragma: no cover - fallback only
    deepspeed = None


def _run_command(args, cwd: Optional[str] = None):
    try:
        completed = subprocess.run(
            args,
            cwd=cwd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return completed.returncode, completed.stdout.strip(), completed.stderr.strip()
    except Exception as exc:
        return -1, "", str(exc)


def collect_git_metadata(repo_root: Optional[str] = None, max_status_lines: int = 200) -> Dict[str, Any]:
    try:
        max_lines = max(int(max_status_lines), 0)
    except (TypeError, ValueError):
        max_lines = 200
    metadata: Dict[str, Any] = {
        "available": False,
        "repo_root": "",
        "commit_hash": "unknown",
        "commit_short": "unknown",
        "branch": "unknown",
        "dirty": None,
        "status_porcelain": [],
        "status_truncated": False,
        "error": "",
    }

    if shutil.which("git") is None:
        metadata["error"] = "git_not_found"
        return metadata

    query_root = os.path.abspath(repo_root) if repo_root else os.getcwd()
    rc, inside_work_tree, err = _run_command(["git", "rev-parse", "--is-inside-work-tree"], cwd=query_root)
    if rc != 0 or inside_work_tree != "true":
        metadata["error"] = err or "not_inside_git_work_tree"
        return metadata

    rc, top_level, err = _run_command(["git", "rev-parse", "--show-toplevel"], cwd=query_root)
    if rc == 0 and top_level:
        metadata["repo_root"] = top_level
        query_root = top_level
    else:
        metadata["repo_root"] = query_root
        if err:
            metadata["error"] = err

    rc, commit_hash, err = _run_command(["git", "rev-parse", "HEAD"], cwd=query_root)
    if rc == 0 and commit_hash:
        metadata["commit_hash"] = commit_hash
    elif err and not metadata["error"]:
        metadata["error"] = err

    rc, commit_short, err = _run_command(["git", "rev-parse", "--short=12", "HEAD"], cwd=query_root)
    if rc == 0 and commit_short:
        metadata["commit_short"] = commit_short
    elif err and not metadata["error"]:
        metadata["error"] = err

    rc, branch, err = _run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=query_root)
    if rc == 0 and branch:
        metadata["branch"] = branch
    elif err and not metadata["error"]:
        metadata["error"] = err

    rc, porcelain, err = _run_command(["git", "status", "--porcelain"], cwd=query_root)
    if rc == 0:
        status_lines = [line for line in porcelain.splitlines() if line.strip()]
        metadata["dirty"] = len(status_lines) > 0
        if len(status_lines) > max_lines:
            metadata["status_porcelain"] = status_lines[:max_lines]
            metadata["status_truncated"] = True
        else:
            metadata["status_porcelain"] = status_lines
            metadata["status_truncated"] = False
    elif err and not metadata["error"]:
        metadata["error"] = err

    metadata["available"] = True
    return metadata


def collect_startup_metadata(
    process_rank: int,
    local_rank: int,
    world_size: int,
    experiment_output_dir: str,
    repo_root: Optional[str] = None,
    max_git_status_lines: int = 200,
) -> Dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    cuda_device_count = torch.cuda.device_count() if cuda_available else 0
    current_device = torch.cuda.current_device() if cuda_available and cuda_device_count > 0 else None
    current_device_name = (
        torch.cuda.get_device_name(current_device)
        if current_device is not None
        else None
    )

    startup_context: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "cwd": os.getcwd(),
        "argv": list(sys.argv),
        "command": " ".join(shlex.quote(arg) for arg in sys.argv),
        "paths": {
            "experiment_output_dir": os.path.abspath(experiment_output_dir),
        },
        "distributed": {
            "rank": int(process_rank),
            "local_rank": int(local_rank),
            "world_size": int(world_size),
        },
        "runtime": {
            "python_executable": sys.executable,
            "python_version": sys.version,
            "platform": platform.platform(),
            "hostname": socket.gethostname(),
            "torch_version": torch.__version__,
            "deepspeed_version": (
                getattr(deepspeed, "__version__", "unknown")
                if deepspeed is not None
                else "unavailable"
            ),
            "cuda_available": cuda_available,
            "cuda_version": torch.version.cuda,
            "cuda_device_count": cuda_device_count,
            "current_device_index": current_device,
            "current_device_name": current_device_name,
        },
    }
    startup_context["git"] = collect_git_metadata(repo_root=repo_root, max_status_lines=max_git_status_lines)
    return startup_context


def write_startup_artifacts(
    output_dir: str,
    cfg: DictConfig,
    run_context: Dict[str, Any],
    config_filename: str = "resolved_config.yaml",
    context_filename: str = "run_context.json",
    logger: Optional[logging.Logger] = None,
) -> Dict[str, str]:
    log = logger or logging.getLogger(__name__)
    artifacts: Dict[str, str] = {}

    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as exc:
        log.warning("Failed to create startup metadata output dir %s: %s", output_dir, exc)
        return artifacts

    config_path = os.path.join(output_dir, config_filename)
    context_path = os.path.join(output_dir, context_filename)

    try:
        resolved_yaml = OmegaConf.to_yaml(cfg, resolve=True)
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(resolved_yaml)
        artifacts["resolved_config_path"] = config_path
    except Exception as exc:
        log.warning("Failed to write resolved startup config to %s: %s", config_path, exc)

    try:
        with open(context_path, "w", encoding="utf-8") as f:
            json.dump(run_context, f, indent=2, sort_keys=True, ensure_ascii=False)
            f.write("\n")
        artifacts["run_context_path"] = context_path
    except Exception as exc:
        log.warning("Failed to write startup run context to %s: %s", context_path, exc)

    return artifacts
