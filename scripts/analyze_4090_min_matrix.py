#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
import statistics
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


CASE_ORDER = ["z0_sdpa", "z0_flash_attention_2", "z1_sdpa"]


def to_float(value: object) -> float:
    try:
        text = str(value).strip()
    except Exception:
        return math.nan
    if text == "" or text.lower() in {"nan", "none", "null"}:
        return math.nan
    try:
        return float(text)
    except Exception:
        return math.nan


def safe_mean(values: Iterable[float]) -> float:
    vals = [v for v in values if not math.isnan(v)]
    return statistics.mean(vals) if vals else math.nan


def safe_median(values: Iterable[float]) -> float:
    vals = [v for v in values if not math.isnan(v)]
    return statistics.median(vals) if vals else math.nan


def safe_max(values: Iterable[float]) -> float:
    vals = [v for v in values if not math.isnan(v)]
    return max(vals) if vals else math.nan


def safe_cv(values: Iterable[float]) -> float:
    vals = [v for v in values if not math.isnan(v)]
    if not vals:
        return math.nan
    if len(vals) == 1:
        return 0.0
    mean_v = statistics.mean(vals)
    if mean_v <= 0:
        return math.nan
    return statistics.stdev(vals) / mean_v


def slope_per_min(xs: List[float], ys: List[float]) -> float:
    points = [(x, y) for x, y in zip(xs, ys) if not math.isnan(x) and not math.isnan(y)]
    if len(points) < 2:
        return math.nan
    x_vals = [x for x, _ in points]
    y_vals = [y for _, y in points]
    x_mean = sum(x_vals) / len(x_vals)
    y_mean = sum(y_vals) / len(y_vals)
    den = sum((x - x_mean) ** 2 for x in x_vals)
    if den == 0:
        return math.nan
    num = sum((x - x_mean) * (y - y_mean) for x, y in points)
    return (num / den) * 60.0


def read_tsv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def first_row_or_empty(path: Path, delimiter: str = ",") -> Dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            return row
    return {}


def parse_step_spans(train_log: Path, target_step: int) -> Dict[str, float]:
    if not train_log.exists():
        return {
            "step_span_0_target_sec": math.nan,
            "step_span_0_last_sec": math.nan,
            "last_step_seen": math.nan,
        }

    # Example: 2026-02-12 00:24:36,570 - ... - INFO - Step 0: Loss ...
    pat = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}).*Step\s+([0-9]+):")
    fmt = "%Y-%m-%d %H:%M:%S,%f"

    step0_ts: Optional[datetime] = None
    target_ts: Optional[datetime] = None
    last_ts: Optional[datetime] = None
    last_step = -1

    with train_log.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pat.search(line)
            if not m:
                continue
            ts = datetime.strptime(m.group(1), fmt)
            step = int(m.group(2))
            if step0_ts is None and step == 0:
                step0_ts = ts
            if target_ts is None and step >= target_step:
                target_ts = ts
            if step > last_step:
                last_step = step
                last_ts = ts

    span_0_target = math.nan
    span_0_last = math.nan
    if step0_ts is not None and target_ts is not None:
        span_0_target = (target_ts - step0_ts).total_seconds()
    if step0_ts is not None and last_ts is not None:
        span_0_last = (last_ts - step0_ts).total_seconds()

    return {
        "step_span_0_target_sec": span_0_target,
        "step_span_0_last_sec": span_0_last,
        "last_step_seen": float(last_step if last_step >= 0 else math.nan),
    }


def scan_log_flags(train_log: Path, launcher_log: Path) -> Dict[str, bool]:
    flags = {
        "dtype_warning": False,
        "oom_error": False,
        "traceback_error": False,
    }

    patterns = {
        "dtype_warning": re.compile(r"Flash Attention 2 only supports|current dype|current dtype.*float32", re.IGNORECASE),
        "oom_error": re.compile(r"out of memory|cuda oom|oom-kill|cudnn_status_alloc_failed", re.IGNORECASE),
        "traceback_error": re.compile(r"Traceback \(most recent call last\)", re.IGNORECASE),
    }

    for path in (train_log, launcher_log):
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                for key, pat in patterns.items():
                    if not flags[key] and pat.search(line):
                        flags[key] = True
    return flags


def summarize_gpu(run_dir: Path) -> Dict[str, float]:
    csv_path = run_dir / "gpu_telemetry.csv"
    if not csv_path.exists():
        return {
            "gpu_total_power_mean_w": math.nan,
            "gpu_total_power_max_w": math.nan,
            "gpu_util_gap_mean_pct": math.nan,
            "gpu_util_gap_max_pct": math.nan,
        }

    rows = read_csv(csv_path)
    by_ts: Dict[str, Dict[int, Tuple[float, float]]] = {}
    for row in rows:
        ts = row.get("epoch_sec", "")
        idx = int(to_float(row.get("gpu_index")))
        util = to_float(row.get("util_gpu_pct"))
        power = to_float(row.get("power_w"))
        if ts == "" or idx < 0:
            continue
        by_ts.setdefault(ts, {})[idx] = (util, power)

    total_powers = []
    util_gaps = []
    for _, item in by_ts.items():
        if len(item) < 2:
            continue
        sorted_ids = sorted(item.keys())
        id0, id1 = sorted_ids[0], sorted_ids[1]
        u0, p0 = item[id0]
        u1, p1 = item[id1]
        if not math.isnan(p0) and not math.isnan(p1):
            total_powers.append(p0 + p1)
        if not math.isnan(u0) and not math.isnan(u1):
            util_gaps.append(abs(u0 - u1))

    return {
        "gpu_total_power_mean_w": safe_mean(total_powers),
        "gpu_total_power_max_w": safe_max(total_powers),
        "gpu_util_gap_mean_pct": safe_mean(util_gaps),
        "gpu_util_gap_max_pct": safe_max(util_gaps),
    }


def summarize_host(case_root: Path) -> Dict[str, float]:
    csv_path = case_root / "host_telemetry.csv"
    if not csv_path.exists():
        return {
            "host_rss_tail_slope_mib_min": math.nan,
            "host_rss_tail_drift_mib": math.nan,
            "host_load1_mean": math.nan,
        }

    rows = read_csv(csv_path)
    ts = [to_float(r.get("epoch_sec")) for r in rows]
    rss_mib = [to_float(r.get("train_rss_kib")) / 1024.0 for r in rows]
    load1 = [to_float(r.get("load1")) for r in rows]
    proc_count = [to_float(r.get("train_proc_count")) for r in rows]

    # Restrict evaluation to the steady training interval and drop warmup tail.
    eval_idx = [i for i, c in enumerate(proc_count) if not math.isnan(c) and c >= 10]
    if eval_idx:
        lo, hi = min(eval_idx), max(eval_idx)
        ts_eval = ts[lo : hi + 1]
        rss_eval = rss_mib[lo : hi + 1]
    else:
        ts_eval = ts
        rss_eval = rss_mib

    pairs = [(x, y) for x, y in zip(ts_eval, rss_eval) if not math.isnan(x) and not math.isnan(y)]
    if len(pairs) >= 2:
        cut = min(len(pairs) - 1, max(1, int(len(pairs) * 0.2)))
        pairs = pairs[cut:]
        ts_eval = [x for x, _ in pairs]
        rss_eval = [y for _, y in pairs]

    slope = slope_per_min(ts_eval, rss_eval)
    if rss_eval:
        n = max(1, int(len(rss_eval) * 0.1))
        head = safe_mean(rss_eval[:n])
        tail = safe_mean(rss_eval[-n:])
        drift = tail - head if not (math.isnan(head) or math.isnan(tail)) else math.nan
    else:
        drift = math.nan

    return {
        "host_rss_tail_slope_mib_min": slope,
        "host_rss_tail_drift_mib": drift,
        "host_load1_mean": safe_mean(load1),
    }


def discover_case_roots(suite_root: Path) -> Dict[str, List[Tuple[int, Path]]]:
    case_roots: Dict[str, List[Tuple[int, Path]]] = {}
    pat = re.compile(r"^(.*)_r([0-9]+)$")
    for p in sorted(suite_root.iterdir()):
        if not p.is_dir():
            continue
        m = pat.match(p.name)
        if m:
            case_name = m.group(1)
            round_idx = int(m.group(2))
            if case_name not in CASE_ORDER:
                continue
            if not (p / "run_manifest.tsv").exists():
                continue
            case_roots.setdefault(case_name, []).append((round_idx, p))
            continue

        # Backward compatibility: previous suites used plain case directory names.
        if p.name in CASE_ORDER and (p / "run_manifest.tsv").exists():
            case_roots.setdefault(p.name, []).append((1, p))
    for k in list(case_roots.keys()):
        case_roots[k] = sorted(case_roots[k], key=lambda x: x[0])
    return case_roots


def load_run_record(case_name: str, round_idx: int, case_root: Path, target_step: int) -> Dict[str, object]:
    manifest_rows = read_tsv(case_root / "run_manifest.tsv")
    manifest = manifest_rows[0] if manifest_rows else {}
    per_run = first_row_or_empty(case_root / "per_run_metrics.csv")

    run_dir = Path(manifest.get("run_dir", "")) if manifest else Path("")
    train_log = Path(manifest.get("train_log", "")) if manifest else Path("")
    launcher_log = Path(manifest.get("launcher_log", "")) if manifest else Path("")

    gpu_stats = summarize_gpu(run_dir) if run_dir.exists() else summarize_gpu(case_root)
    host_stats = summarize_host(case_root)
    span_stats = parse_step_spans(train_log, target_step)
    log_flags = scan_log_flags(train_log, launcher_log)

    record: Dict[str, object] = {
        "case": case_name,
        "round": round_idx,
        "case_root": str(case_root),
        "status": manifest.get("status", ""),
        "exit_code": manifest.get("exit_code", ""),
        "zero_stage": manifest.get("zero_stage", ""),
        "attn_impl_effective": manifest.get("attn_impl_effective", ""),
        "speech_attn_impl_effective": manifest.get("speech_attn_impl_effective", manifest.get("attn_impl_effective", "")),
        "text_attn_impl_effective": manifest.get("text_attn_impl_effective", manifest.get("attn_impl_effective", "")),
        "model_load_dtype_effective": manifest.get("model_load_dtype_effective", ""),
        "tf32_enabled": manifest.get("tf32_enabled", ""),
        "tail_samples_per_sec": to_float(per_run.get("tail_samples_per_sec")),
        "tail_mean_iter_p50_ms": to_float(per_run.get("tail_mean_iter_p50_ms")),
        "last_iter_p50_ms": to_float(per_run.get("last_iter_p50_ms")),
        "iter_p90_over_p50": to_float(per_run.get("iter_p90_over_p50", manifest.get("iter_p90_over_p50"))),
        "step_p90_over_p50": to_float(per_run.get("step_p90_over_p50", manifest.get("step_p90_over_p50"))),
        "gpu_total_power_mean_w": gpu_stats["gpu_total_power_mean_w"],
        "gpu_total_power_max_w": gpu_stats["gpu_total_power_max_w"],
        "gpu_util_gap_mean_pct": gpu_stats["gpu_util_gap_mean_pct"],
        "gpu_util_gap_max_pct": gpu_stats["gpu_util_gap_max_pct"],
        "host_rss_tail_slope_mib_min": host_stats["host_rss_tail_slope_mib_min"],
        "host_rss_tail_drift_mib": host_stats["host_rss_tail_drift_mib"],
        "host_load1_mean": host_stats["host_load1_mean"],
        "step_span_0_target_sec": span_stats["step_span_0_target_sec"],
        "step_span_0_last_sec": span_stats["step_span_0_last_sec"],
        "last_step_seen": span_stats["last_step_seen"],
        "dtype_warning": log_flags["dtype_warning"],
        "oom_error": log_flags["oom_error"],
        "traceback_error": log_flags["traceback_error"],
    }
    return record


def aggregate_case(case_name: str, runs: List[Dict[str, object]]) -> Dict[str, object]:
    success_runs = [r for r in runs if str(r.get("status", "")).lower() == "success"]

    summary: Dict[str, object] = {
        "case": case_name,
        "runs_total": len(runs),
        "runs_success": len(success_runs),
        "runs_failed": len(runs) - len(success_runs),
        "dtype_warning_count": sum(1 for r in runs if bool(r.get("dtype_warning"))),
        "oom_error_count": sum(1 for r in runs if bool(r.get("oom_error"))),
        "traceback_error_count": sum(1 for r in runs if bool(r.get("traceback_error"))),
    }

    def vals(key: str) -> List[float]:
        return [to_float(r.get(key)) for r in success_runs]

    summary.update(
        {
            "tail_samples_per_sec_median": safe_median(vals("tail_samples_per_sec")),
            "tail_samples_per_sec_mean": safe_mean(vals("tail_samples_per_sec")),
            "tail_mean_iter_p50_ms_median": safe_median(vals("tail_mean_iter_p50_ms")),
            "tail_mean_iter_p50_ms_mean": safe_mean(vals("tail_mean_iter_p50_ms")),
            "iter_p90_over_p50_median": safe_median(vals("iter_p90_over_p50")),
            "step_p90_over_p50_median": safe_median(vals("step_p90_over_p50")),
            "iter_p50_cv": safe_cv(vals("tail_mean_iter_p50_ms")),
            "gpu_total_power_mean_w_mean": safe_mean(vals("gpu_total_power_mean_w")),
            "gpu_util_gap_mean_pct_mean": safe_mean(vals("gpu_util_gap_mean_pct")),
            "host_rss_tail_slope_mib_min_median": safe_median(vals("host_rss_tail_slope_mib_min")),
            "step_span_0_target_sec_median": safe_median(vals("step_span_0_target_sec")),
            "step_span_0_last_sec_median": safe_median(vals("step_span_0_last_sec")),
        }
    )
    return summary


def format_num(v: object, digits: int = 4) -> str:
    x = to_float(v)
    if math.isnan(x):
        return "nan"
    return f"{x:.{digits}f}"


def compute_gate(case_map: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    sdpa = case_map.get("z0_sdpa", {})
    flash = case_map.get("z0_flash_attention_2", {})
    z1 = case_map.get("z1_sdpa", {})

    sdpa_sps = to_float(sdpa.get("tail_samples_per_sec_median"))
    flash_sps = to_float(flash.get("tail_samples_per_sec_median"))
    z1_sps = to_float(z1.get("tail_samples_per_sec_median"))

    sdpa_iter_ratio = to_float(sdpa.get("iter_p90_over_p50_median"))
    flash_iter_ratio = to_float(flash.get("iter_p90_over_p50_median"))
    sdpa_step_ratio = to_float(sdpa.get("step_p90_over_p50_median"))
    flash_step_ratio = to_float(flash.get("step_p90_over_p50_median"))

    throughput_gain = math.nan
    if not math.isnan(sdpa_sps) and sdpa_sps > 0 and not math.isnan(flash_sps):
        throughput_gain = flash_sps / sdpa_sps - 1.0

    iter_ratio_ok = (
        not math.isnan(flash_iter_ratio)
        and not math.isnan(sdpa_iter_ratio)
        and flash_iter_ratio <= (sdpa_iter_ratio + 0.05)
    )

    step_ratio_improve = math.nan
    if not math.isnan(sdpa_step_ratio) and sdpa_step_ratio > 0 and not math.isnan(flash_step_ratio):
        step_ratio_improve = (sdpa_step_ratio - flash_step_ratio) / sdpa_step_ratio

    no_dtype_warning = int(flash.get("dtype_warning_count", 0)) == 0
    no_errors = (
        int(flash.get("runs_failed", 0)) == 0
        and int(flash.get("oom_error_count", 0)) == 0
        and int(flash.get("traceback_error_count", 0)) == 0
    )

    go_throughput = (
        no_dtype_warning
        and no_errors
        and iter_ratio_ok
        and (not math.isnan(throughput_gain) and throughput_gain >= 0.03)
    )
    go_stability = (
        no_dtype_warning
        and no_errors
        and iter_ratio_ok
        and (not math.isnan(step_ratio_improve) and step_ratio_improve >= 0.25)
    )

    if go_throughput:
        decision = "go_flash_default_throughput"
    elif go_stability:
        decision = "go_flash_default_stability"
    else:
        decision = "no_go_keep_sdpa_default"

    z0_ref_sps = math.nan
    if not math.isnan(sdpa_sps) and not math.isnan(flash_sps):
        z0_ref_sps = max(sdpa_sps, flash_sps)
    elif not math.isnan(sdpa_sps):
        z0_ref_sps = sdpa_sps
    elif not math.isnan(flash_sps):
        z0_ref_sps = flash_sps

    z0_vs_z1_gain = math.nan
    z0_vs_z1_pass = False
    if not math.isnan(z0_ref_sps) and z0_ref_sps > 0 and not math.isnan(z1_sps) and z1_sps > 0:
        z0_vs_z1_gain = z0_ref_sps / z1_sps - 1.0
        z0_vs_z1_pass = z0_vs_z1_gain >= 0.30

    return {
        "throughput_gain_flash_vs_sdpa": throughput_gain,
        "iter_ratio_ok": iter_ratio_ok,
        "step_ratio_improve_flash_vs_sdpa": step_ratio_improve,
        "no_dtype_warning": no_dtype_warning,
        "no_errors": no_errors,
        "decision": decision,
        "z0_ref_samples_per_sec": z0_ref_sps,
        "z0_vs_z1_gain": z0_vs_z1_gain,
        "z0_vs_z1_pass_30pct": z0_vs_z1_pass,
    }


def write_case_csv(path: Path, case_summaries: List[Dict[str, object]]) -> None:
    if not case_summaries:
        return
    fieldnames = list(case_summaries[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(case_summaries)


def write_markdown(
    path: Path,
    suite_root: Path,
    target_step: int,
    runs: List[Dict[str, object]],
    case_summaries: List[Dict[str, object]],
    gate: Dict[str, object],
) -> None:
    lines: List[str] = []
    lines.append("# 4090 Revalidation Report")
    lines.append("")
    lines.append(f"- Generated at: `{datetime.now().isoformat()}`")
    lines.append(f"- Suite root: `{suite_root}`")
    lines.append(f"- Target step-span metric: `Step 0 -> Step {target_step}`")
    lines.append(f"- Total run records: `{len(runs)}`")
    lines.append("")

    lines.append("## Case Summary")
    lines.append("")
    lines.append("| case | runs_success/runs_total | tail_samples_median | tail_iter_p50_median_ms | iter_p90/p50_median | step_p90/p50_median | iter_p50_cv | gpu_total_power_mean_w | gpu_util_gap_mean_pct | host_rss_tail_slope_mib_min | step_span_0_target_median_sec | dtype_warn | oom | traceback |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    by_case = {x["case"]: x for x in case_summaries}
    for case in CASE_ORDER:
        row = by_case.get(case)
        if row is None:
            continue
        lines.append(
            f"| {case} | {row['runs_success']}/{row['runs_total']} | "
            f"{format_num(row['tail_samples_per_sec_median'], 3)} | "
            f"{format_num(row['tail_mean_iter_p50_ms_median'], 3)} | "
            f"{format_num(row['iter_p90_over_p50_median'], 3)} | "
            f"{format_num(row['step_p90_over_p50_median'], 3)} | "
            f"{format_num(row['iter_p50_cv'], 4)} | "
            f"{format_num(row['gpu_total_power_mean_w_mean'], 2)} | "
            f"{format_num(row['gpu_util_gap_mean_pct_mean'], 2)} | "
            f"{format_num(row['host_rss_tail_slope_mib_min_median'], 3)} | "
            f"{format_num(row['step_span_0_target_sec_median'], 2)} | "
            f"{row['dtype_warning_count']} | {row['oom_error_count']} | {row['traceback_error_count']} |"
        )

    lines.append("")
    lines.append("## Go/No-Go")
    lines.append("")
    lines.append(f"- throughput_gain_flash_vs_sdpa: `{format_num(gate.get('throughput_gain_flash_vs_sdpa'), 4)}` (target >= `0.0300`)")
    lines.append(f"- iter_ratio_ok (flash <= sdpa + 0.05): `{gate.get('iter_ratio_ok')}`")
    lines.append(f"- step_ratio_improve_flash_vs_sdpa: `{format_num(gate.get('step_ratio_improve_flash_vs_sdpa'), 4)}` (stability fallback target >= `0.2500`)")
    lines.append(f"- no_dtype_warning: `{gate.get('no_dtype_warning')}`")
    lines.append(f"- no_errors: `{gate.get('no_errors')}`")
    lines.append(f"- decision: `{gate.get('decision')}`")
    lines.append("")
    lines.append(f"- z0_vs_z1_gain: `{format_num(gate.get('z0_vs_z1_gain'), 4)}` (target >= `0.3000`)")
    lines.append(f"- z0_vs_z1_pass_30pct: `{gate.get('z0_vs_z1_pass_30pct')}`")

    lines.append("")
    lines.append("## Next Actions")
    lines.append("")
    decision = str(gate.get("decision", ""))
    if decision.startswith("go_flash_default"):
        lines.append("1. Keep `z0` as mainline and proceed to phase-2 micro-batch grid: `mb=160,192` on the winning z0 attention path.")
        lines.append("2. Run optional A/B for `ENABLE_TORCH_COMPILE=true` on the winning z0 path only.")
        lines.append("3. Evaluate optional throughput-specialized `fixed_slice_seconds=2.0` separately from baseline.")
    else:
        lines.append("1. Keep `z0+sdpa` as default and increase repeats (>=5) for flash2 vs sdpa to reduce uncertainty.")
        lines.append("2. Inspect `run_order.tsv`, `quick_leak_diagnose.csv`, and per-run host telemetry for contention episodes.")
        lines.append("3. Only after stability convergence, re-attempt phase-2 (`mb=160,192`, compile/fixed-slice A/B).")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze 4090 min-matrix revalidation suite.")
    parser.add_argument("--suite-root", required=True)
    parser.add_argument("--step-span-target", type=int, default=1000)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-csv", default="")
    parser.add_argument("--output-md", default="")
    args = parser.parse_args()

    suite_root = Path(args.suite_root)
    if not suite_root.exists():
        raise SystemExit(f"[ERROR] suite-root not found: {suite_root}")

    case_roots = discover_case_roots(suite_root)
    if not case_roots:
        raise SystemExit(f"[ERROR] no case directories found under: {suite_root}")

    run_records: List[Dict[str, object]] = []
    for case_name in CASE_ORDER:
        for round_idx, case_root in case_roots.get(case_name, []):
            run_records.append(load_run_record(case_name, round_idx, case_root, args.step_span_target))

    case_summaries = [aggregate_case(case_name, [r for r in run_records if r["case"] == case_name]) for case_name in CASE_ORDER if any(r["case"] == case_name for r in run_records)]
    case_map = {x["case"]: x for x in case_summaries}
    gate = compute_gate(case_map)

    payload = {
        "generated_at": datetime.now().isoformat(),
        "suite_root": str(suite_root),
        "step_span_target": args.step_span_target,
        "runs": run_records,
        "case_summaries": case_summaries,
        "gate": gate,
    }

    output_json = Path(args.output_json) if args.output_json else suite_root / "suite_report.json"
    output_csv = Path(args.output_csv) if args.output_csv else suite_root / "suite_case_summary.csv"
    output_md = Path(args.output_md) if args.output_md else suite_root / "suite_report.md"

    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_case_csv(output_csv, case_summaries)
    write_markdown(output_md, suite_root, args.step_span_target, run_records, case_summaries, gate)

    print(f"[INFO] Wrote: {output_json}")
    print(f"[INFO] Wrote: {output_csv}")
    print(f"[INFO] Wrote: {output_md}")
    print(f"[INFO] Decision: {gate.get('decision')}")


if __name__ == "__main__":
    main()
