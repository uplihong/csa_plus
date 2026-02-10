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

    timing_patterns = {k: re.compile(rf"{k}:p50=([0-9]+(?:\.[0-9]+)?)") for k in TIMING_KEYS}
    step_pattern = re.compile(r"Step\s+([0-9]+):")

    per_run_rows = []
    timing_rows = []
    skipped_rows = []

    with manifest_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        manifest_rows = list(reader)

    for row in manifest_rows:
        mode = row.get("mode", "")
        group = row.get("group", "")
        repeat = int(row.get("repeat", "0"))
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
        last_step = None
        with train_log.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m_step = step_pattern.search(line)
                if m_step:
                    last_step = int(m_step.group(1))
                if "Timing(window=" not in line:
                    continue
                values = {}
                ok = True
                for key in TIMING_KEYS:
                    m = timing_patterns[key].search(line)
                    if not m:
                        ok = False
                        break
                    values[key] = float(m.group(1))
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

        iter_values = [p["iter_ms"] for p in points]
        tail_n = min(tail_points, len(iter_values))
        tail_values = iter_values[-tail_n:]

        last_point = points[-1]
        tail_mean_iter = statistics.mean(tail_values)
        tail_std_iter = statistics.stdev(tail_values) if len(tail_values) > 1 else 0.0
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
            "attn_impl_effective": row.get("attn_impl_effective", ""),
            "torch_compile_enabled": row.get("torch_compile_enabled", ""),
            "gpu_name": row.get("gpu_name", ""),
            "gpu_cc": row.get("gpu_cc", ""),
            "log_every_steps": row.get("log_every_steps", ""),
            "validation_every_steps": row.get("validation_every_steps", ""),
            "checkpoint_every_steps": row.get("checkpoint_every_steps", ""),
            "timing_points": len(points),
            "tail_points_used": tail_n,
            "last_step": last_point["step"],
            "last_iter_p50_ms": last_point["iter_ms"],
            "last_fwd_p50_ms": last_point["fwd_ms"],
            "last_bwd_p50_ms": last_point["bwd_ms"],
            "tail_mean_iter_p50_ms": tail_mean_iter,
            "tail_std_iter_p50_ms": tail_std_iter,
            "all_mean_iter_p50_ms": all_mean_iter,
            "all_std_iter_p50_ms": all_std_iter,
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
                    "data_wait_ms_p50": point["data_wait_ms"],
                    "preprocess_ms_p50": point["preprocess_ms"],
                    "fwd_ms_p50": point["fwd_ms"],
                    "bwd_ms_p50": point["bwd_ms"],
                    "step_ms_p50": point["step_ms"],
                    "iter_ms_p50": point["iter_ms"],
                    "train_log": str(train_log),
                }
            )

    per_run_rows.sort(key=lambda x: (x["mode"], x["group"], x["repeat"]))
    timing_rows.sort(key=lambda x: (x["mode"], x["group"], x["repeat"], x["index"]))

    per_run_csv = output_root / "per_run_metrics.csv"
    timing_csv = output_root / "timing_points.csv"
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

    grouped = defaultdict(list)
    for row in per_run_rows:
        grouped[row["group"]].append(row)

    baseline_mean = math.nan
    if baseline_group in grouped and grouped[baseline_group]:
        baseline_mean = statistics.mean([r["tail_mean_iter_p50_ms"] for r in grouped[baseline_group]])

    group_rows = []
    for group, rows in sorted(grouped.items()):
        vals = [r["tail_mean_iter_p50_ms"] for r in rows]
        steps_vals = [r["tail_steps_per_sec"] for r in rows]
        samples_vals = [r["tail_samples_per_sec"] for r in rows if not math.isnan(r["tail_samples_per_sec"])]
        mean_iter = statistics.mean(vals)
        std_iter = statistics.stdev(vals) if len(vals) > 1 else 0.0

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
                "attn_impl_effective": unique_or_mixed(rows, "attn_impl_effective"),
                "torch_compile_enabled": unique_or_mixed(rows, "torch_compile_enabled"),
                "gpu_name": unique_or_mixed(rows, "gpu_name"),
                "gpu_cc": unique_or_mixed(rows, "gpu_cc"),
                "tail_mean_iter_p50_ms_mean": mean_iter,
                "tail_mean_iter_p50_ms_std": std_iter,
                "tail_mean_iter_p50_ms_min": min(vals),
                "tail_mean_iter_p50_ms_max": max(vals),
                "tail_steps_per_sec_mean": statistics.mean(steps_vals),
                "tail_steps_per_sec_std": statistics.stdev(steps_vals) if len(steps_vals) > 1 else 0.0,
                "tail_samples_per_sec_mean": statistics.mean(samples_vals) if samples_vals else math.nan,
                "tail_samples_per_sec_std": statistics.stdev(samples_vals) if len(samples_vals) > 1 else 0.0,
                "speedup_vs_baseline": baseline_mean / mean_iter if not math.isnan(baseline_mean) else math.nan,
            }
        )

    if group_rows:
        with group_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(group_rows[0].keys()))
            writer.writeheader()
            writer.writerows(group_rows)

    ranked_rows = sorted(group_rows, key=lambda x: x["tail_mean_iter_p50_ms_mean"])
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
            "attn_impl_effective": best_group["attn_impl_effective"],
            "torch_compile_enabled": best_group["torch_compile_enabled"],
            "gpu_name": best_group["gpu_name"],
            "gpu_cc": best_group["gpu_cc"],
            "tail_mean_iter_p50_ms_mean": best_group["tail_mean_iter_p50_ms_mean"],
            "tail_steps_per_sec_mean": best_group["tail_steps_per_sec_mean"],
            "tail_samples_per_sec_mean": best_group["tail_samples_per_sec_mean"],
            "speedup_vs_baseline": best_group["speedup_vs_baseline"],
        }

    summary_payload = {
        "generated_at": datetime.now().isoformat(),
        "output_root": str(output_root),
        "tail_points": tail_points,
        "baseline_group": baseline_group,
        "groups": group_rows,
        "ranked_groups": ranked_rows,
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
            env_line("BEST_ATTN_IMPL_EFFECTIVE", best_payload.get("attn_impl_effective", "")),
            env_line("BEST_TORCH_COMPILE_ENABLED", best_payload.get("torch_compile_enabled", "")),
            env_line("BEST_GPU_NAME", best_payload.get("gpu_name", "")),
            env_line("BEST_GPU_CC", best_payload.get("gpu_cc", "")),
            env_line("BEST_TAIL_MEAN_ITER_P50_MS", best_payload.get("tail_mean_iter_p50_ms_mean", "")),
            env_line("BEST_TAIL_STEPS_PER_SEC", best_payload.get("tail_steps_per_sec_mean", "")),
            env_line("BEST_TAIL_SAMPLES_PER_SEC", best_payload.get("tail_samples_per_sec_mean", "")),
            env_line("BEST_SPEEDUP_VS_BASELINE", best_payload.get("speedup_vs_baseline", "")),
        ]
        best_env.write_text("\n".join(env_lines) + "\n", encoding="utf-8")

        override_lines = [
            "# Recommended Hydra/DeepSpeed overrides from best group",
            f"deepspeed_config_yaml.zero_optimization.stage={best_payload.get('zero_stage', '')}",
            f"deepspeed_config_yaml.train_micro_batch_size_per_gpu={best_payload.get('micro_batch', '')}",
            f"++train.data.num_workers={best_payload.get('num_workers', '')}",
            f"++train.data.prefetch_factor={best_payload.get('prefetch_factor', '')}",
            f"++model.speech_encoder.attn_implementation={best_payload.get('attn_impl_effective', '')}",
            f"++model.text_encoder.attn_implementation={best_payload.get('attn_impl_effective', '')}",
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
    lines.append("")
    if best_payload is not None:
        lines.append("## Best Config")
        lines.append("")
        lines.append(f"- Group: `{best_payload['group']}`")
        lines.append(f"- zero_stage: `{best_payload['zero_stage']}`")
        lines.append(f"- micro_batch: `{best_payload['micro_batch']}`")
        lines.append(f"- precision_mode_effective: `{best_payload['precision_mode_effective']}`")
        lines.append(f"- attn_impl_effective: `{best_payload['attn_impl_effective']}`")
        lines.append(f"- torch_compile_enabled: `{best_payload['torch_compile_enabled']}`")
        lines.append(f"- gpu_name: `{best_payload['gpu_name']}`")
        lines.append(f"- gpu_cc: `{best_payload['gpu_cc']}`")
        lines.append(f"- tail_mean_iter_p50_ms: `{best_payload['tail_mean_iter_p50_ms_mean']:.4f}`")
        lines.append(f"- tail_steps_per_sec: `{best_payload['tail_steps_per_sec_mean']:.4f}`")
        if not math.isnan(best_payload.get("tail_samples_per_sec_mean", math.nan)):
            lines.append(f"- tail_samples_per_sec: `{best_payload['tail_samples_per_sec_mean']:.4f}`")
        lines.append(f"- best_config JSON: `{best_json}`")
        lines.append(f"- best_config ENV: `{best_env}`")
        lines.append(f"- best overrides: `{best_overrides}`")
        lines.append("")

    lines.append("## Group Summary")
    lines.append("")
    if ranked_rows:
        lines.append("| rank | group | runs | zero_stage | micro_batch | precision | attn_impl | compile | num_workers | prefetch | tail_mean_iter_p50_ms_mean | tail_steps_per_sec_mean | tail_samples_per_sec_mean | speedup_vs_baseline |")
        lines.append("|---:|---|---:|---|---|---|---|---|---|---|---:|---:|---:|---:|")
        for idx, row in enumerate(ranked_rows, start=1):
            sample_speed = "nan" if math.isnan(row["tail_samples_per_sec_mean"]) else f"{row['tail_samples_per_sec_mean']:.4f}"
            speedup = "nan" if math.isnan(row["speedup_vs_baseline"]) else f"{row['speedup_vs_baseline']:.4f}"
            lines.append(
                f"| {idx} | {row['group']} | {row['runs']} | {row['zero_stage']} | {row['micro_batch']} | {row['precision_mode_effective']} | {row['attn_impl_effective']} | {row['torch_compile_enabled']} | {row['num_workers']} | {row['prefetch_factor']} | "
                f"{row['tail_mean_iter_p50_ms_mean']:.4f} | {row['tail_steps_per_sec_mean']:.4f} | {sample_speed} | {speedup} |"
            )
    else:
        lines.append("No successful runs found.")

    if skipped_rows:
        lines.append("")
        lines.append("## Skipped Runs")
        lines.append("")
        lines.append(f"See: `{skipped_tsv}`")

    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[INFO] Wrote: {per_run_csv}")
    print(f"[INFO] Wrote: {timing_csv}")
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
