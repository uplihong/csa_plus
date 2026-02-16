#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path


TIMING_KEYS = ["data_wait_ms", "preprocess_ms", "fwd_ms", "bwd_ms", "step_ms", "iter_ms"]


def json_safe(obj):
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    return obj


def unique_or_mixed(rows, key):
    values = {str(row.get(key, "")) for row in rows}
    values.discard("")
    if not values:
        return ""
    if len(values) == 1:
        return next(iter(values))
    return "mixed"


def env_line(key, value):
    if isinstance(value, float) and math.isnan(value):
        return f"{key}="
    if value is None:
        return f"{key}="
    return f"{key}={value}"


def parse_boolish(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def percentile(values, q):
    if not values:
        return math.nan
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * (q / 100.0)
    left = int(pos)
    right = min(left + 1, len(ordered) - 1)
    frac = pos - left
    return float(ordered[left] * (1 - frac) + ordered[right] * frac)


def parse_rank_spread_metrics(log_path: Path, tail_points: int):
    pattern = re.compile(
        r"TimingRank step=(\d+) rank=(\d+).*? step=([0-9]+(?:\.[0-9]+)?) iter=([0-9]+(?:\.[0-9]+)?)"
    )
    per_step = defaultdict(lambda: {"iter": [], "step": []})

    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pattern.search(line)
            if not m:
                continue
            step = int(m.group(1))
            step_ms = float(m.group(3))
            iter_ms = float(m.group(4))
            per_step[step]["step"].append(step_ms)
            per_step[step]["iter"].append(iter_ms)

    if not per_step:
        return None

    tail_steps = sorted(per_step.keys())[-tail_points:]
    iter_spreads = []
    step_spreads = []
    for step in tail_steps:
        iter_vals = per_step[step]["iter"]
        step_vals = per_step[step]["step"]
        if len(iter_vals) >= 2:
            iter_spreads.append(max(iter_vals) - min(iter_vals))
        if len(step_vals) >= 2:
            step_spreads.append(max(step_vals) - min(step_vals))

    return {
        "rank_iter_spread_p90_ms": percentile(iter_spreads, 90) if iter_spreads else math.nan,
        "rank_step_spread_p90_ms": percentile(step_spreads, 90) if step_spreads else math.nan,
        "rank_spread_points": len(iter_spreads),
    }


def main():
    parser = argparse.ArgumentParser(description="Summarize stage1 benchmark logs.")
    parser.add_argument("--manifest", required=True, help="Path to run_manifest.tsv")
    parser.add_argument("--output-root", required=True, help="Output directory")
    parser.add_argument("--tail-points", type=int, default=20, help="Tail timing points used per run")
    parser.add_argument("--baseline-group", default="old_like", help="Baseline group for speedup")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    output_root = Path(args.output_root)
    tail_points = int(args.tail_points)
    baseline_group = args.baseline_group
    output_root.mkdir(parents=True, exist_ok=True)

    timing_patterns = {
        k: re.compile(rf"{k}:p50=([0-9]+(?:\.[0-9]+)?) p90=([0-9]+(?:\.[0-9]+)?)")
        for k in TIMING_KEYS
    }
    step_pattern = re.compile(r"Step\s+([0-9]+):")
    eta_pattern = re.compile(
        r"TimingETA step=(\d+) remain_steps=(\d+) eta_sec=([0-9]+(?:\.[0-9]+)?) "
        r"eta_hms=([0-9]+:[0-9]{2}:[0-9]{2}) mode=([a-zA-Z0-9_]+)"
    )

    per_run_rows = []
    timing_rows = []
    eta_rows = []
    skipped_rows = []

    with manifest_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        manifest_rows = list(reader)

    for row in manifest_rows:
        mode = row.get("mode", "")
        group = row.get("group", "")
        try:
            repeat = int(row.get("repeat", "0"))
        except (TypeError, ValueError):
            repeat = 0
        status = row.get("status", "")
        train_log = Path(row.get("train_log", ""))
        launcher_log = Path(row.get("launcher_log", ""))

        if status != "success":
            skipped_rows.append(
                {
                    "mode": mode,
                    "group": group,
                    "repeat": repeat,
                    "reason": "run_failed",
                    "train_log": str(train_log),
                }
            )
            continue
        if not train_log.exists():
            skipped_rows.append(
                {
                    "mode": mode,
                    "group": group,
                    "repeat": repeat,
                    "reason": "train_log_missing",
                    "train_log": str(train_log),
                }
            )
            continue

        points = []
        eta_points = []
        last_step = None
        with train_log.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m_step = step_pattern.search(line)
                if m_step:
                    last_step = int(m_step.group(1))
                m_eta = eta_pattern.search(line)
                if m_eta:
                    eta_points.append(
                        {
                            "step": int(m_eta.group(1)),
                            "remain_steps": int(m_eta.group(2)),
                            "eta_sec": float(m_eta.group(3)),
                            "eta_hms": str(m_eta.group(4)),
                            "eta_mode": str(m_eta.group(5)),
                        }
                    )
                if "Timing(window=" not in line:
                    continue
                values = {}
                ok = True
                for key in TIMING_KEYS:
                    m = timing_patterns[key].search(line)
                    if not m:
                        ok = False
                        break
                    values[f"{key}_p50"] = float(m.group(1))
                    values[f"{key}_p90"] = float(m.group(2))
                if ok:
                    values["step"] = last_step
                    points.append(values)

        if not points:
            skipped_rows.append(
                {
                    "mode": mode,
                    "group": group,
                    "repeat": repeat,
                    "reason": "no_timing_points",
                    "train_log": str(train_log),
                }
            )
            continue

        eta_first = eta_points[0] if eta_points else None
        eta_last = eta_points[-1] if eta_points else None

        iter_values = [p["iter_ms_p50"] for p in points]
        iter_p90_values = [p["iter_ms_p90"] for p in points]
        tail_n = min(tail_points, len(iter_values))
        tail_values = iter_values[-tail_n:]
        tail_p90_values = iter_p90_values[-tail_n:]

        last_point = points[-1]
        tail_mean_iter = statistics.mean(tail_values)
        tail_mean_iter_p90 = statistics.mean(tail_p90_values)
        tail_std_iter = statistics.stdev(tail_values) if len(tail_values) > 1 else 0.0
        tail_std_iter_p90 = statistics.stdev(tail_p90_values) if len(tail_p90_values) > 1 else 0.0
        all_mean_iter = statistics.mean(iter_values)
        all_std_iter = statistics.stdev(iter_values) if len(iter_values) > 1 else 0.0
        steps_per_sec_tail = 1000.0 / tail_mean_iter

        global_batch_raw = row.get("global_batch", "")
        try:
            global_batch = int(global_batch_raw) if global_batch_raw else None
        except ValueError:
            global_batch = None

        if global_batch is not None and global_batch > 0:
            samples_per_sec_tail = global_batch * 1000.0 / tail_mean_iter
        else:
            samples_per_sec_tail = math.nan

        iter_ratio = parse_float(row.get("iter_p90_over_p50", ""))
        data_ratio = parse_float(row.get("data_p90_over_p50", ""))
        step_ratio = parse_float(row.get("step_p90_over_p50", ""))
        if math.isnan(iter_ratio) and last_point["iter_ms_p50"] > 0:
            iter_ratio = last_point["iter_ms_p90"] / last_point["iter_ms_p50"]
        if math.isnan(data_ratio) and last_point["data_wait_ms_p50"] > 0:
            data_ratio = last_point["data_wait_ms_p90"] / last_point["data_wait_ms_p50"]
        if math.isnan(step_ratio) and last_point["step_ms_p50"] > 0:
            step_ratio = last_point["step_ms_p90"] / last_point["step_ms_p50"]
        unstable_flag = parse_boolish(row.get("unstable_run_flag", ""))
        if not unstable_flag:
            unstable_flag = (
                (not math.isnan(iter_ratio) and iter_ratio > 2.0)
                or (not math.isnan(step_ratio) and step_ratio > 3.0)
            )

        rank_iter_spread_p90_ms = math.nan
        rank_step_spread_p90_ms = math.nan
        rank_spread_points = 0
        rank_spread_log_source = ""
        if str(row.get("timing_rank_scope", "")).strip().lower() == "all":
            rank_metrics = None
            rank_source = None
            for candidate in [train_log, launcher_log]:
                if not candidate.exists():
                    continue
                parsed = parse_rank_spread_metrics(candidate, tail_n)
                if parsed is None:
                    continue
                if rank_metrics is None or parsed["rank_spread_points"] > rank_metrics["rank_spread_points"]:
                    rank_metrics = parsed
                    rank_source = candidate
            if rank_metrics is not None:
                rank_iter_spread_p90_ms = rank_metrics["rank_iter_spread_p90_ms"]
                rank_step_spread_p90_ms = rank_metrics["rank_step_spread_p90_ms"]
                rank_spread_points = rank_metrics["rank_spread_points"]
                rank_spread_log_source = str(rank_source)

        per_run = {
            "mode": mode,
            "group": group,
            "repeat": repeat,
            "run_dir": row.get("run_dir", ""),
            "train_log": str(train_log),
            "launcher_log": str(launcher_log),
            "zero_stage": row.get("zero_stage", ""),
            "micro_batch": row.get("micro_batch", ""),
            "world_size": row.get("world_size", ""),
            "global_batch": global_batch if global_batch is not None else "",
            "deterministic": row.get("deterministic", ""),
            "cudnn_benchmark": row.get("cudnn_benchmark", ""),
            "wall_clock_breakdown": row.get("wall_clock_breakdown", ""),
            "num_workers": row.get("num_workers", ""),
            "prefetch_factor": row.get("prefetch_factor", ""),
            "dataset_manifest_path": row.get("dataset_manifest_path", ""),
            "dataset_use_trim": row.get("dataset_use_trim", ""),
            "dataset_offline_trimmed": row.get("dataset_offline_trimmed", ""),
            "enable_cuda_sync_timing": row.get("enable_cuda_sync_timing", ""),
            "timing_rank_scope": row.get("timing_rank_scope", ""),
            "precision_mode_req": row.get("precision_mode_req", ""),
            "precision_mode_effective": row.get("precision_mode_effective", ""),
            "model_load_dtype_effective": row.get("model_load_dtype_effective", ""),
            "attn_impl_effective": row.get("attn_impl_effective", ""),
            "speech_attn_impl_effective": row.get("speech_attn_impl_effective", row.get("attn_impl_effective", "")),
            "text_attn_impl_effective": row.get("text_attn_impl_effective", row.get("attn_impl_effective", "")),
            "torch_compile_enabled": row.get("torch_compile_enabled", ""),
            "tf32_enabled": row.get("tf32_enabled", ""),
            "gpu_name": row.get("gpu_name", ""),
            "gpu_cc": row.get("gpu_cc", ""),
            "gpu_uuid_list": row.get("gpu_uuid_list", ""),
            "gpu_power_limit_w": row.get("gpu_power_limit_w", ""),
            "pcie_gen": row.get("pcie_gen", ""),
            "driver_version": row.get("driver_version", ""),
            "git_commit_hash": row.get("git_commit_hash", ""),
            "git_commit_short": row.get("git_commit_short", ""),
            "git_branch": row.get("git_branch", ""),
            "git_dirty": row.get("git_dirty", ""),
            "gpu_telemetry_rows": row.get("gpu_telemetry_rows", ""),
            "gpu_telemetry_empty_flag": row.get("gpu_telemetry_empty_flag", ""),
            "host_telemetry_rows": row.get("host_telemetry_rows", ""),
            "host_telemetry_empty_flag": row.get("host_telemetry_empty_flag", ""),
            "log_every_steps": row.get("log_every_steps", ""),
            "validation_every_steps": row.get("validation_every_steps", ""),
            "checkpoint_every_steps": row.get("checkpoint_every_steps", ""),
            "last_eta_step": row.get("last_eta_step", ""),
            "last_eta_remaining_steps": row.get("last_eta_remaining_steps", ""),
            "last_eta_sec": row.get("last_eta_sec", ""),
            "last_eta_hms": row.get("last_eta_hms", ""),
            "last_eta_mode": row.get("last_eta_mode", ""),
            "timing_points": len(points),
            "tail_points_used": tail_n,
            "eta_points": len(eta_points),
            "eta_first_step": eta_first["step"] if eta_first else math.nan,
            "eta_first_sec": eta_first["eta_sec"] if eta_first else math.nan,
            "eta_first_hms": eta_first["eta_hms"] if eta_first else "",
            "eta_last_step": eta_last["step"] if eta_last else math.nan,
            "eta_last_sec": eta_last["eta_sec"] if eta_last else math.nan,
            "eta_last_hms": eta_last["eta_hms"] if eta_last else "",
            "eta_mode_effective": eta_last["eta_mode"] if eta_last else "",
            "last_step": last_point["step"],
            "last_iter_p50_ms": last_point["iter_ms_p50"],
            "last_iter_p90_ms": last_point["iter_ms_p90"],
            "last_fwd_p50_ms": last_point["fwd_ms_p50"],
            "last_fwd_p90_ms": last_point["fwd_ms_p90"],
            "last_bwd_p50_ms": last_point["bwd_ms_p50"],
            "last_bwd_p90_ms": last_point["bwd_ms_p90"],
            "last_data_wait_p50_ms": last_point["data_wait_ms_p50"],
            "last_data_wait_p90_ms": last_point["data_wait_ms_p90"],
            "last_step_p50_ms": last_point["step_ms_p50"],
            "last_step_p90_ms": last_point["step_ms_p90"],
            "tail_mean_iter_p50_ms": tail_mean_iter,
            "tail_mean_iter_p90_ms": tail_mean_iter_p90,
            "tail_std_iter_p50_ms": tail_std_iter,
            "tail_std_iter_p90_ms": tail_std_iter_p90,
            "all_mean_iter_p50_ms": all_mean_iter,
            "all_std_iter_p50_ms": all_std_iter,
            "iter_p90_over_p50": iter_ratio,
            "data_p90_over_p50": data_ratio,
            "step_p90_over_p50": step_ratio,
            "rank_iter_spread_p90_ms": rank_iter_spread_p90_ms,
            "rank_step_spread_p90_ms": rank_step_spread_p90_ms,
            "rank_spread_points": rank_spread_points,
            "rank_spread_log_source": rank_spread_log_source,
            "unstable_run_flag": unstable_flag,
            "tail_steps_per_sec": steps_per_sec_tail,
            "tail_samples_per_sec": samples_per_sec_tail,
        }
        per_run_rows.append(per_run)

        for idx, point in enumerate(points, start=1):
            timing_rows.append(
                {
                    "mode": mode,
                    "group": group,
                    "repeat": repeat,
                    "index": idx,
                    "step": point["step"],
                    "data_wait_ms_p50": point["data_wait_ms_p50"],
                    "data_wait_ms_p90": point["data_wait_ms_p90"],
                    "preprocess_ms_p50": point["preprocess_ms_p50"],
                    "preprocess_ms_p90": point["preprocess_ms_p90"],
                    "fwd_ms_p50": point["fwd_ms_p50"],
                    "fwd_ms_p90": point["fwd_ms_p90"],
                    "bwd_ms_p50": point["bwd_ms_p50"],
                    "bwd_ms_p90": point["bwd_ms_p90"],
                    "step_ms_p50": point["step_ms_p50"],
                    "step_ms_p90": point["step_ms_p90"],
                    "iter_ms_p50": point["iter_ms_p50"],
                    "iter_ms_p90": point["iter_ms_p90"],
                    "train_log": str(train_log),
                }
            )
        for idx, point in enumerate(eta_points, start=1):
            eta_rows.append(
                {
                    "mode": mode,
                    "group": group,
                    "repeat": repeat,
                    "index": idx,
                    "step": point["step"],
                    "remain_steps": point["remain_steps"],
                    "eta_sec": point["eta_sec"],
                    "eta_hms": point["eta_hms"],
                    "eta_mode": point["eta_mode"],
                    "train_log": str(train_log),
                }
            )

    per_run_rows.sort(key=lambda x: (x["mode"], x["group"], x["repeat"]))
    timing_rows.sort(key=lambda x: (x["mode"], x["group"], x["repeat"], x["index"]))
    eta_rows.sort(key=lambda x: (x["mode"], x["group"], x["repeat"], x["index"]))

    per_run_csv = output_root / "per_run_metrics.csv"
    timing_csv = output_root / "timing_points.csv"
    eta_csv = output_root / "eta_points.csv"
    group_csv = output_root / "group_summary.csv"
    ranked_csv = output_root / "ranked_groups.csv"
    summary_json = output_root / "summary.json"
    summary_md = output_root / "summary.md"
    skipped_tsv = output_root / "skipped_runs.tsv"
    best_json = output_root / "best_config.json"
    best_env = output_root / "best_config.env"
    best_overrides = output_root / "best_overrides.txt"

    if per_run_rows:
        with per_run_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(per_run_rows[0].keys()))
            writer.writeheader()
            writer.writerows(per_run_rows)
    if timing_rows:
        with timing_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(timing_rows[0].keys()))
            writer.writeheader()
            writer.writerows(timing_rows)
    if eta_rows:
        with eta_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(eta_rows[0].keys()))
            writer.writeheader()
            writer.writerows(eta_rows)

    grouped = defaultdict(list)
    for row in per_run_rows:
        grouped[row["group"]].append(row)

    baseline_mean = math.nan
    if baseline_group in grouped and grouped[baseline_group]:
        baseline_mean = statistics.mean([r["tail_mean_iter_p50_ms"] for r in grouped[baseline_group]])

    group_rows = []
    for group, rows in sorted(grouped.items()):
        vals = [r["tail_mean_iter_p50_ms"] for r in rows]
        vals_p90 = [r["tail_mean_iter_p90_ms"] for r in rows]
        steps_vals = [r["tail_steps_per_sec"] for r in rows]
        samples_vals = [r["tail_samples_per_sec"] for r in rows if not math.isnan(r["tail_samples_per_sec"])]
        iter_ratio_vals = [r["iter_p90_over_p50"] for r in rows if not math.isnan(r["iter_p90_over_p50"])]
        rank_iter_spread_vals = [r["rank_iter_spread_p90_ms"] for r in rows if not math.isnan(r["rank_iter_spread_p90_ms"])]
        rank_step_spread_vals = [r["rank_step_spread_p90_ms"] for r in rows if not math.isnan(r["rank_step_spread_p90_ms"])]
        rank_spread_point_vals = [parse_float(r.get("rank_spread_points", "")) for r in rows]
        rank_spread_point_vals = [v for v in rank_spread_point_vals if not math.isnan(v)]
        unstable_runs = [r for r in rows if parse_boolish(r.get("unstable_run_flag", False))]
        telemetry_rows_vals = [parse_float(r.get("gpu_telemetry_rows", "")) for r in rows]
        telemetry_rows_vals = [v for v in telemetry_rows_vals if not math.isnan(v)]
        telemetry_empty_runs = [r for r in rows if parse_boolish(r.get("gpu_telemetry_empty_flag", False))]
        host_telemetry_rows_vals = [parse_float(r.get("host_telemetry_rows", "")) for r in rows]
        host_telemetry_rows_vals = [v for v in host_telemetry_rows_vals if not math.isnan(v)]
        host_telemetry_empty_runs = [r for r in rows if parse_boolish(r.get("host_telemetry_empty_flag", False))]
        mean_iter = statistics.mean(vals)
        mean_iter_p90 = statistics.mean(vals_p90)
        std_iter = statistics.stdev(vals) if len(vals) > 1 else 0.0
        std_iter_p90 = statistics.stdev(vals_p90) if len(vals_p90) > 1 else 0.0
        unstable_ratio = len(unstable_runs) / len(rows) if rows else 0.0
        stability_ratio = (mean_iter_p90 / mean_iter) if mean_iter > 0 else math.nan
        stability_score = stability_ratio * (1.0 + unstable_ratio) if not math.isnan(stability_ratio) else math.nan

        group_rows.append(
            {
                "group": group,
                "mode": unique_or_mixed(rows, "mode"),
                "runs": len(rows),
                "zero_stage": unique_or_mixed(rows, "zero_stage"),
                "micro_batch": unique_or_mixed(rows, "micro_batch"),
                "num_workers": unique_or_mixed(rows, "num_workers"),
                "prefetch_factor": unique_or_mixed(rows, "prefetch_factor"),
                "enable_cuda_sync_timing": unique_or_mixed(rows, "enable_cuda_sync_timing"),
                "timing_rank_scope": unique_or_mixed(rows, "timing_rank_scope"),
                "precision_mode_effective": unique_or_mixed(rows, "precision_mode_effective"),
                "model_load_dtype_effective": unique_or_mixed(rows, "model_load_dtype_effective"),
                "attn_impl_effective": unique_or_mixed(rows, "attn_impl_effective"),
                "speech_attn_impl_effective": unique_or_mixed(rows, "speech_attn_impl_effective"),
                "text_attn_impl_effective": unique_or_mixed(rows, "text_attn_impl_effective"),
                "torch_compile_enabled": unique_or_mixed(rows, "torch_compile_enabled"),
                "tf32_enabled": unique_or_mixed(rows, "tf32_enabled"),
                "gpu_name": unique_or_mixed(rows, "gpu_name"),
                "gpu_cc": unique_or_mixed(rows, "gpu_cc"),
                "gpu_uuid_list": unique_or_mixed(rows, "gpu_uuid_list"),
                "gpu_power_limit_w": unique_or_mixed(rows, "gpu_power_limit_w"),
                "pcie_gen": unique_or_mixed(rows, "pcie_gen"),
                "driver_version": unique_or_mixed(rows, "driver_version"),
                "git_commit_short": unique_or_mixed(rows, "git_commit_short"),
                "git_branch": unique_or_mixed(rows, "git_branch"),
                "git_dirty": unique_or_mixed(rows, "git_dirty"),
                "tail_mean_iter_p50_ms_mean": mean_iter,
                "tail_mean_iter_p90_ms_mean": mean_iter_p90,
                "tail_mean_iter_p50_ms_std": std_iter,
                "tail_mean_iter_p90_ms_std": std_iter_p90,
                "tail_mean_iter_p50_ms_min": min(vals),
                "tail_mean_iter_p50_ms_max": max(vals),
                "tail_mean_iter_p90_ms_min": min(vals_p90),
                "tail_mean_iter_p90_ms_max": max(vals_p90),
                "iter_p90_over_p50_mean": statistics.mean(iter_ratio_vals) if iter_ratio_vals else math.nan,
                "iter_p90_over_p50_std": statistics.stdev(iter_ratio_vals) if len(iter_ratio_vals) > 1 else 0.0,
                "rank_iter_spread_p90_ms_mean": statistics.mean(rank_iter_spread_vals) if rank_iter_spread_vals else math.nan,
                "rank_iter_spread_p90_ms_std": statistics.stdev(rank_iter_spread_vals) if len(rank_iter_spread_vals) > 1 else 0.0,
                "rank_step_spread_p90_ms_mean": statistics.mean(rank_step_spread_vals) if rank_step_spread_vals else math.nan,
                "rank_step_spread_p90_ms_std": statistics.stdev(rank_step_spread_vals) if len(rank_step_spread_vals) > 1 else 0.0,
                "rank_spread_points_mean": statistics.mean(rank_spread_point_vals) if rank_spread_point_vals else math.nan,
                "tail_steps_per_sec_mean": statistics.mean(steps_vals),
                "tail_steps_per_sec_std": statistics.stdev(steps_vals) if len(steps_vals) > 1 else 0.0,
                "tail_samples_per_sec_mean": statistics.mean(samples_vals) if samples_vals else math.nan,
                "tail_samples_per_sec_std": statistics.stdev(samples_vals) if len(samples_vals) > 1 else 0.0,
                "unstable_run_count": len(unstable_runs),
                "unstable_run_ratio": unstable_ratio,
                "gpu_telemetry_rows_mean": statistics.mean(telemetry_rows_vals) if telemetry_rows_vals else math.nan,
                "gpu_telemetry_empty_count": len(telemetry_empty_runs),
                "gpu_telemetry_empty_ratio": (len(telemetry_empty_runs) / len(rows)) if rows else 0.0,
                "host_telemetry_rows_mean": statistics.mean(host_telemetry_rows_vals) if host_telemetry_rows_vals else math.nan,
                "host_telemetry_empty_count": len(host_telemetry_empty_runs),
                "host_telemetry_empty_ratio": (len(host_telemetry_empty_runs) / len(rows)) if rows else 0.0,
                "stability_score": stability_score,
                "speedup_vs_baseline": baseline_mean / mean_iter if not math.isnan(baseline_mean) else math.nan,
            }
        )

    if group_rows:
        by_iter = sorted(group_rows, key=lambda x: x["tail_mean_iter_p50_ms_mean"])
        for idx, row in enumerate(by_iter, start=1):
            row["rank_by_iter_p50"] = idx
        by_stability = sorted(
            group_rows,
            key=lambda x: (
                x["stability_score"] if not math.isnan(x["stability_score"]) else float("inf"),
                x["tail_mean_iter_p50_ms_mean"],
            ),
        )
        for idx, row in enumerate(by_stability, start=1):
            row["rank_by_stability"] = idx
        by_balanced = sorted(
            group_rows,
            key=lambda x: (x["rank_by_iter_p50"] + x["rank_by_stability"], x["rank_by_iter_p50"]),
        )
        for idx, row in enumerate(by_balanced, start=1):
            row["rank_balanced"] = idx

    if group_rows:
        with group_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(group_rows[0].keys()))
            writer.writeheader()
            writer.writerows(group_rows)

    ranked_rows = sorted(group_rows, key=lambda x: x["tail_mean_iter_p50_ms_mean"])
    ranked_by_stability = sorted(
        group_rows,
        key=lambda x: (
            x["stability_score"] if not math.isnan(x["stability_score"]) else float("inf"),
            x["tail_mean_iter_p50_ms_mean"],
        ),
    )
    if ranked_rows:
        with ranked_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(ranked_rows[0].keys()))
            writer.writeheader()
            writer.writerows(ranked_rows)

    if skipped_rows:
        with skipped_tsv.open("w", encoding="utf-8", newline="") as f:
            f.write("mode\tgroup\trepeat\treason\ttrain_log\n")
            for row in skipped_rows:
                f.write(f"{row['mode']}\t{row['group']}\t{row['repeat']}\t{row['reason']}\t{row['train_log']}\n")

    best_group = ranked_rows[0] if ranked_rows else None
    best_payload = None
    if best_group is not None:
        best_payload = {
            "group": best_group["group"],
            "mode": best_group["mode"],
            "zero_stage": best_group["zero_stage"],
            "micro_batch": best_group["micro_batch"],
            "num_workers": best_group["num_workers"],
            "prefetch_factor": best_group["prefetch_factor"],
            "precision_mode_effective": best_group["precision_mode_effective"],
            "model_load_dtype_effective": best_group["model_load_dtype_effective"],
            "attn_impl_effective": best_group["attn_impl_effective"],
            "speech_attn_impl_effective": best_group["speech_attn_impl_effective"],
            "text_attn_impl_effective": best_group["text_attn_impl_effective"],
            "torch_compile_enabled": best_group["torch_compile_enabled"],
            "tf32_enabled": best_group["tf32_enabled"],
            "gpu_name": best_group["gpu_name"],
            "gpu_cc": best_group["gpu_cc"],
            "gpu_uuid_list": best_group["gpu_uuid_list"],
            "gpu_power_limit_w": best_group["gpu_power_limit_w"],
            "pcie_gen": best_group["pcie_gen"],
            "driver_version": best_group["driver_version"],
            "git_commit_short": best_group["git_commit_short"],
            "git_branch": best_group["git_branch"],
            "git_dirty": best_group["git_dirty"],
            "tail_mean_iter_p50_ms_mean": best_group["tail_mean_iter_p50_ms_mean"],
            "tail_mean_iter_p90_ms_mean": best_group["tail_mean_iter_p90_ms_mean"],
            "rank_iter_spread_p90_ms_mean": best_group.get("rank_iter_spread_p90_ms_mean", math.nan),
            "rank_step_spread_p90_ms_mean": best_group.get("rank_step_spread_p90_ms_mean", math.nan),
            "stability_score": best_group["stability_score"],
            "rank_by_stability": best_group["rank_by_stability"],
            "rank_balanced": best_group["rank_balanced"],
            "tail_steps_per_sec_mean": best_group["tail_steps_per_sec_mean"],
            "tail_samples_per_sec_mean": best_group["tail_samples_per_sec_mean"],
            "gpu_telemetry_rows_mean": best_group["gpu_telemetry_rows_mean"],
            "gpu_telemetry_empty_ratio": best_group["gpu_telemetry_empty_ratio"],
            "host_telemetry_rows_mean": best_group.get("host_telemetry_rows_mean", math.nan),
            "host_telemetry_empty_ratio": best_group.get("host_telemetry_empty_ratio", math.nan),
            "speedup_vs_baseline": best_group["speedup_vs_baseline"],
        }

    summary_payload = {
        "generated_at": datetime.now().isoformat(),
        "output_root": str(output_root),
        "tail_points": tail_points,
        "baseline_group": baseline_group,
        "groups": group_rows,
        "ranked_groups": ranked_rows,
        "ranked_groups_by_stability": ranked_by_stability,
        "best": best_payload,
        "runs": per_run_rows,
        "skipped_runs": skipped_rows,
    }
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(json_safe(summary_payload), f, indent=2, ensure_ascii=False, allow_nan=False)

    if best_payload is not None:
        with best_json.open("w", encoding="utf-8") as f:
            json.dump(json_safe(best_payload), f, indent=2, ensure_ascii=False, allow_nan=False)

        env_lines = [
            env_line("BEST_GROUP", best_payload.get("group", "")),
            env_line("BEST_MODE", best_payload.get("mode", "")),
            env_line("BEST_ZERO_STAGE", best_payload.get("zero_stage", "")),
            env_line("BEST_MICRO_BATCH", best_payload.get("micro_batch", "")),
            env_line("BEST_NUM_WORKERS", best_payload.get("num_workers", "")),
            env_line("BEST_PREFETCH_FACTOR", best_payload.get("prefetch_factor", "")),
            env_line("BEST_PRECISION_MODE_EFFECTIVE", best_payload.get("precision_mode_effective", "")),
            env_line("BEST_MODEL_LOAD_DTYPE_EFFECTIVE", best_payload.get("model_load_dtype_effective", "")),
            env_line("BEST_ATTN_IMPL_EFFECTIVE", best_payload.get("attn_impl_effective", "")),
            env_line("BEST_SPEECH_ATTN_IMPL_EFFECTIVE", best_payload.get("speech_attn_impl_effective", "")),
            env_line("BEST_TEXT_ATTN_IMPL_EFFECTIVE", best_payload.get("text_attn_impl_effective", "")),
            env_line("BEST_TORCH_COMPILE_ENABLED", best_payload.get("torch_compile_enabled", "")),
            env_line("BEST_TF32_ENABLED", best_payload.get("tf32_enabled", "")),
            env_line("BEST_GPU_NAME", best_payload.get("gpu_name", "")),
            env_line("BEST_GPU_CC", best_payload.get("gpu_cc", "")),
            env_line("BEST_GPU_UUID_LIST", best_payload.get("gpu_uuid_list", "")),
            env_line("BEST_GPU_POWER_LIMIT_W", best_payload.get("gpu_power_limit_w", "")),
            env_line("BEST_PCIE_GEN", best_payload.get("pcie_gen", "")),
            env_line("BEST_DRIVER_VERSION", best_payload.get("driver_version", "")),
            env_line("BEST_GIT_COMMIT_SHORT", best_payload.get("git_commit_short", "")),
            env_line("BEST_GIT_BRANCH", best_payload.get("git_branch", "")),
            env_line("BEST_GIT_DIRTY", best_payload.get("git_dirty", "")),
            env_line("BEST_TAIL_MEAN_ITER_P50_MS", best_payload.get("tail_mean_iter_p50_ms_mean", "")),
            env_line("BEST_TAIL_MEAN_ITER_P90_MS", best_payload.get("tail_mean_iter_p90_ms_mean", "")),
            env_line("BEST_RANK_ITER_SPREAD_P90_MS", best_payload.get("rank_iter_spread_p90_ms_mean", "")),
            env_line("BEST_RANK_STEP_SPREAD_P90_MS", best_payload.get("rank_step_spread_p90_ms_mean", "")),
            env_line("BEST_STABILITY_SCORE", best_payload.get("stability_score", "")),
            env_line("BEST_RANK_BY_STABILITY", best_payload.get("rank_by_stability", "")),
            env_line("BEST_RANK_BALANCED", best_payload.get("rank_balanced", "")),
            env_line("BEST_TAIL_STEPS_PER_SEC", best_payload.get("tail_steps_per_sec_mean", "")),
            env_line("BEST_TAIL_SAMPLES_PER_SEC", best_payload.get("tail_samples_per_sec_mean", "")),
            env_line("BEST_GPU_TELEMETRY_ROWS_MEAN", best_payload.get("gpu_telemetry_rows_mean", "")),
            env_line("BEST_GPU_TELEMETRY_EMPTY_RATIO", best_payload.get("gpu_telemetry_empty_ratio", "")),
            env_line("BEST_HOST_TELEMETRY_ROWS_MEAN", best_payload.get("host_telemetry_rows_mean", "")),
            env_line("BEST_HOST_TELEMETRY_EMPTY_RATIO", best_payload.get("host_telemetry_empty_ratio", "")),
            env_line("BEST_SPEEDUP_VS_BASELINE", best_payload.get("speedup_vs_baseline", "")),
        ]
        best_env.write_text("\n".join(env_lines) + "\n", encoding="utf-8")

        override_lines = [
            "# Recommended Hydra/DeepSpeed overrides from best group",
            f"deepspeed_config_yaml.zero_optimization.stage={best_payload.get('zero_stage', '')}",
            f"deepspeed_config_yaml.train_micro_batch_size_per_gpu={best_payload.get('micro_batch', '')}",
            f"++train.data.num_workers={best_payload.get('num_workers', '')}",
            f"++train.data.prefetch_factor={best_payload.get('prefetch_factor', '')}",
            f"++model.speech_encoder.attn_implementation={best_payload.get('speech_attn_impl_effective', '')}",
            f"++model.text_encoder.attn_implementation={best_payload.get('text_attn_impl_effective', '')}",
            f"++model.speech_encoder.torch_dtype={best_payload.get('model_load_dtype_effective', '')}",
            f"++model.text_encoder.torch_dtype={best_payload.get('model_load_dtype_effective', '')}",
            f"++train.enable_tf32={best_payload.get('tf32_enabled', '')}",
        ]
        precision_mode = best_payload.get("precision_mode_effective", "")
        if precision_mode == "bf16":
            override_lines.append("deepspeed_config_yaml.bf16.enabled=true")
            override_lines.append("deepspeed_config_yaml.fp16.enabled=false")
        elif precision_mode == "fp16":
            override_lines.append("deepspeed_config_yaml.bf16.enabled=false")
            override_lines.append("deepspeed_config_yaml.fp16.enabled=true")
        best_overrides.write_text("\n".join(override_lines) + "\n", encoding="utf-8")

    lines = []
    lines.append("# Stage1 Benchmark Report")
    lines.append("")
    lines.append(f"- Generated at: `{summary_payload['generated_at']}`")
    lines.append(f"- Output root: `{output_root}`")
    lines.append(f"- Tail timing points used per run: `{tail_points}`")
    lines.append(f"- Baseline group for speedup: `{baseline_group}`")
    if eta_rows:
        lines.append(f"- ETA points CSV: `{eta_csv}`")
    lines.append("")
    if best_payload is not None:
        lines.append("## Best Config")
        lines.append("")
        lines.append(f"- Group: `{best_payload['group']}`")
        lines.append(f"- zero_stage: `{best_payload['zero_stage']}`")
        lines.append(f"- micro_batch: `{best_payload['micro_batch']}`")
        lines.append(f"- precision_mode_effective: `{best_payload['precision_mode_effective']}`")
        lines.append(f"- model_load_dtype_effective: `{best_payload['model_load_dtype_effective']}`")
        lines.append(f"- attn_impl_effective: `{best_payload['attn_impl_effective']}`")
        lines.append(f"- speech_attn_impl_effective: `{best_payload['speech_attn_impl_effective']}`")
        lines.append(f"- text_attn_impl_effective: `{best_payload['text_attn_impl_effective']}`")
        lines.append(f"- torch_compile_enabled: `{best_payload['torch_compile_enabled']}`")
        lines.append(f"- tf32_enabled: `{best_payload['tf32_enabled']}`")
        lines.append(f"- gpu_name: `{best_payload['gpu_name']}`")
        lines.append(f"- gpu_cc: `{best_payload['gpu_cc']}`")
        lines.append(f"- gpu_uuid_list: `{best_payload['gpu_uuid_list']}`")
        lines.append(f"- gpu_power_limit_w: `{best_payload['gpu_power_limit_w']}`")
        lines.append(f"- pcie_gen: `{best_payload['pcie_gen']}`")
        lines.append(f"- driver_version: `{best_payload['driver_version']}`")
        lines.append(f"- git_commit_short: `{best_payload['git_commit_short']}`")
        lines.append(f"- git_branch: `{best_payload['git_branch']}`")
        lines.append(f"- git_dirty: `{best_payload['git_dirty']}`")
        lines.append(f"- tail_mean_iter_p50_ms: `{best_payload['tail_mean_iter_p50_ms_mean']:.4f}`")
        lines.append(f"- tail_mean_iter_p90_ms: `{best_payload['tail_mean_iter_p90_ms_mean']:.4f}`")
        if not math.isnan(best_payload.get("rank_iter_spread_p90_ms_mean", math.nan)):
            lines.append(f"- rank_iter_spread_p90_ms: `{best_payload['rank_iter_spread_p90_ms_mean']:.4f}`")
        if not math.isnan(best_payload.get("rank_step_spread_p90_ms_mean", math.nan)):
            lines.append(f"- rank_step_spread_p90_ms: `{best_payload['rank_step_spread_p90_ms_mean']:.4f}`")
        lines.append(f"- stability_score: `{best_payload['stability_score']:.4f}`")
        lines.append(f"- rank_by_stability: `{best_payload['rank_by_stability']}`")
        lines.append(f"- rank_balanced: `{best_payload['rank_balanced']}`")
        lines.append(f"- tail_steps_per_sec: `{best_payload['tail_steps_per_sec_mean']:.4f}`")
        if not math.isnan(best_payload.get("gpu_telemetry_rows_mean", math.nan)):
            lines.append(f"- gpu_telemetry_rows_mean: `{best_payload['gpu_telemetry_rows_mean']:.2f}`")
        if not math.isnan(best_payload.get("gpu_telemetry_empty_ratio", math.nan)):
            lines.append(f"- gpu_telemetry_empty_ratio: `{best_payload['gpu_telemetry_empty_ratio']:.4f}`")
        if not math.isnan(best_payload.get("host_telemetry_rows_mean", math.nan)):
            lines.append(f"- host_telemetry_rows_mean: `{best_payload['host_telemetry_rows_mean']:.2f}`")
        if not math.isnan(best_payload.get("host_telemetry_empty_ratio", math.nan)):
            lines.append(f"- host_telemetry_empty_ratio: `{best_payload['host_telemetry_empty_ratio']:.4f}`")
        if not math.isnan(best_payload.get("tail_samples_per_sec_mean", math.nan)):
            lines.append(f"- tail_samples_per_sec: `{best_payload['tail_samples_per_sec_mean']:.4f}`")
        lines.append(f"- best_config JSON: `{best_json}`")
        lines.append(f"- best_config ENV: `{best_env}`")
        lines.append(f"- best overrides: `{best_overrides}`")
        lines.append("")

    lines.append("## ETA Overview")
    lines.append("")
    if per_run_rows:
        eta_available_runs = [row for row in per_run_rows if int(row.get("eta_points", 0)) > 0]
        lines.append(f"- Runs with ETA points: `{len(eta_available_runs)}/{len(per_run_rows)}`")
        lines.append(f"- Total ETA points: `{len(eta_rows)}`")
        lines.append("")
        if eta_available_runs:
            lines.append("| mode | group | repeat | eta_points | eta_mode_effective | eta_last_step | eta_last_sec | eta_last_hms |")
            lines.append("|---|---|---:|---:|---|---:|---:|---|")
            for row in eta_available_runs:
                eta_last_sec = row.get("eta_last_sec", math.nan)
                eta_last_sec_str = "nan" if math.isnan(eta_last_sec) else f"{eta_last_sec:.1f}"
                eta_last_step = row.get("eta_last_step", math.nan)
                eta_last_step_str = "nan" if math.isnan(eta_last_step) else f"{int(eta_last_step)}"
                lines.append(
                    f"| {row['mode']} | {row['group']} | {row['repeat']} | {row['eta_points']} | {row.get('eta_mode_effective', '')} | {eta_last_step_str} | {eta_last_sec_str} | {row.get('eta_last_hms', '')} |"
                )
    else:
        lines.append("No successful runs found.")
    lines.append("")

    lines.append("## Group Summary")
    lines.append("")
    if ranked_rows:
        lines.append("| rank_iter | rank_stability | rank_balanced | group | runs | zero_stage | micro_batch | precision | model_dtype | speech_attn | text_attn | compile | tf32 | num_workers | prefetch | tail_mean_iter_p50_ms_mean | tail_mean_iter_p90_ms_mean | stability_score | unstable_run_ratio | tail_steps_per_sec_mean | tail_samples_per_sec_mean | speedup_vs_baseline |")
        lines.append("|---:|---:|---:|---|---:|---|---|---|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
        for row in ranked_rows:
            sample_speed = "nan" if math.isnan(row["tail_samples_per_sec_mean"]) else f"{row['tail_samples_per_sec_mean']:.4f}"
            speedup = "nan" if math.isnan(row["speedup_vs_baseline"]) else f"{row['speedup_vs_baseline']:.4f}"
            stability = "nan" if math.isnan(row["stability_score"]) else f"{row['stability_score']:.4f}"
            lines.append(
                f"| {row['rank_by_iter_p50']} | {row['rank_by_stability']} | {row['rank_balanced']} | {row['group']} | {row['runs']} | {row['zero_stage']} | {row['micro_batch']} | {row['precision_mode_effective']} | {row['model_load_dtype_effective']} | {row['speech_attn_impl_effective']} | {row['text_attn_impl_effective']} | {row['torch_compile_enabled']} | {row['tf32_enabled']} | {row['num_workers']} | {row['prefetch_factor']} | "
                f"{row['tail_mean_iter_p50_ms_mean']:.4f} | {row['tail_mean_iter_p90_ms_mean']:.4f} | {stability} | {row['unstable_run_ratio']:.4f} | {row['tail_steps_per_sec_mean']:.4f} | {sample_speed} | {speedup} |"
            )
    else:
        lines.append("No successful runs found.")

    rank_spread_rows = [
        row
        for row in group_rows
        if not math.isnan(row.get("rank_iter_spread_p90_ms_mean", math.nan))
    ]
    if rank_spread_rows:
        ranked_by_rank_spread = sorted(rank_spread_rows, key=lambda x: x["rank_iter_spread_p90_ms_mean"])
        lines.append("")
        lines.append("## Rank Imbalance (timing_rank_scope=all)")
        lines.append("")
        lines.append("- `tail_mean_iter_*` reflects throughput tail (single-rank view).")
        lines.append("- `rank_*_spread_p90_ms` reflects per-step cross-rank imbalance tail.")
        lines.append("")
        lines.append("| group | rank_iter_spread_p90_ms_mean | rank_step_spread_p90_ms_mean | rank_spread_points_mean |")
        lines.append("|---|---:|---:|---:|")
        for row in ranked_by_rank_spread[:10]:
            iter_spread = row.get("rank_iter_spread_p90_ms_mean", math.nan)
            step_spread = row.get("rank_step_spread_p90_ms_mean", math.nan)
            points = row.get("rank_spread_points_mean", math.nan)
            iter_str = "nan" if math.isnan(iter_spread) else f"{iter_spread:.4f}"
            step_str = "nan" if math.isnan(step_spread) else f"{step_spread:.4f}"
            points_str = "nan" if math.isnan(points) else f"{points:.2f}"
            lines.append(f"| {row['group']} | {iter_str} | {step_str} | {points_str} |")

    if ranked_by_stability:
        lines.append("")
        lines.append("## Top Stable Groups")
        lines.append("")
        lines.append("| rank_stability | group | rank_iter | rank_balanced | stability_score | tail_mean_iter_p50_ms_mean | tail_mean_iter_p90_ms_mean | unstable_run_ratio |")
        lines.append("|---:|---|---:|---:|---:|---:|---:|---:|")
        for row in ranked_by_stability[:10]:
            stability = "nan" if math.isnan(row["stability_score"]) else f"{row['stability_score']:.4f}"
            lines.append(
                f"| {row['rank_by_stability']} | {row['group']} | {row['rank_by_iter_p50']} | {row['rank_balanced']} | {stability} | {row['tail_mean_iter_p50_ms_mean']:.4f} | {row['tail_mean_iter_p90_ms_mean']:.4f} | {row['unstable_run_ratio']:.4f} |"
            )

    if skipped_rows:
        lines.append("")
        lines.append("## Skipped Runs")
        lines.append("")
        lines.append(f"See: `{skipped_tsv}`")

    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[INFO] Wrote: {per_run_csv}")
    print(f"[INFO] Wrote: {timing_csv}")
    if eta_rows:
        print(f"[INFO] Wrote: {eta_csv}")
    print(f"[INFO] Wrote: {group_csv}")
    print(f"[INFO] Wrote: {ranked_csv}")
    print(f"[INFO] Wrote: {summary_json}")
    print(f"[INFO] Wrote: {summary_md}")
    if best_payload is not None:
        print(f"[INFO] Wrote: {best_json}")
        print(f"[INFO] Wrote: {best_env}")
        print(f"[INFO] Wrote: {best_overrides}")
    if skipped_rows:
        print(f"[WARN] Some runs were skipped. Details: {skipped_tsv}")


if __name__ == "__main__":
    main()
