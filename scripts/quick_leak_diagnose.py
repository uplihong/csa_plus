#!/usr/bin/env python3
import argparse
import csv
import math
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


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


def clean(values: Iterable[float]) -> List[float]:
    return [v for v in values if not math.isnan(v)]


def mean(values: Iterable[float]) -> float:
    vals = clean(values)
    return statistics.mean(vals) if vals else math.nan


def percentile(values: Iterable[float], q: float) -> float:
    vals = sorted(clean(values))
    if not vals:
        return math.nan
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q
    left = int(math.floor(pos))
    right = int(math.ceil(pos))
    if left == right:
        return vals[left]
    frac = pos - left
    return vals[left] * (1.0 - frac) + vals[right] * frac


def slope_per_unit(xs: Iterable[float], ys: Iterable[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if not math.isnan(x) and not math.isnan(y)]
    if len(pairs) < 2:
        return math.nan
    x_vals = [x for x, _ in pairs]
    y_vals = [y for _, y in pairs]
    x_mean = sum(x_vals) / len(x_vals)
    y_mean = sum(y_vals) / len(y_vals)
    den = sum((x - x_mean) ** 2 for x in x_vals)
    if den == 0:
        return math.nan
    num = sum((x - x_mean) * (y - y_mean) for x, y in pairs)
    return num / den


def head_tail_mean(values: List[float], ratio: float = 0.1) -> Tuple[float, float]:
    vals = clean(values)
    if not vals:
        return math.nan, math.nan
    n = max(1, int(len(vals) * ratio))
    return mean(vals[:n]), mean(vals[-n:])


def read_tsv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def summarize_gpu(run_dir: Path) -> Dict[str, float]:
    csv_path = run_dir / "gpu_telemetry.csv"
    if not csv_path.exists():
        return {}
    rows = read_csv(csv_path)
    by_gpu: Dict[int, Dict[str, List[float]]] = {}
    for row in rows:
        gpu_idx = int(to_float(row.get("gpu_index")))
        by_gpu.setdefault(gpu_idx, {"ts": [], "mem": [], "util": [], "power": []})
        by_gpu[gpu_idx]["ts"].append(to_float(row.get("epoch_sec")))
        by_gpu[gpu_idx]["mem"].append(to_float(row.get("mem_used_mib")))
        by_gpu[gpu_idx]["util"].append(to_float(row.get("util_gpu_pct")))
        by_gpu[gpu_idx]["power"].append(to_float(row.get("power_w")))

    gpu_tail_slopes = []
    gpu_tail_drifts = []
    gpu_util_means = []
    gpu_power_means = []
    for idx in sorted(by_gpu.keys()):
        ts = by_gpu[idx]["ts"]
        mem = by_gpu[idx]["mem"]
        util = by_gpu[idx]["util"]
        power = by_gpu[idx]["power"]
        if len(mem) < 2:
            continue
        cut = max(1, int(len(mem) * 0.2))
        slope_tail = slope_per_unit(ts[cut:], mem[cut:]) * 60.0
        head, tail = head_tail_mean(mem, ratio=0.1)
        gpu_tail_slopes.append(slope_tail)
        gpu_tail_drifts.append(tail - head)
        gpu_util_means.append(mean(util))
        gpu_power_means.append(mean(power))

    return {
        "gpu_count": float(len(by_gpu)),
        "gpu_mem_tail_slope_max_mib_min": max(clean(gpu_tail_slopes), default=math.nan),
        "gpu_mem_tail_slope_mean_mib_min": mean(gpu_tail_slopes),
        "gpu_mem_tail_drift_max_mib": max(clean(gpu_tail_drifts), default=math.nan),
        "gpu_mem_tail_drift_mean_mib": mean(gpu_tail_drifts),
        "gpu_util_mean_pct": mean(gpu_util_means),
        "gpu_power_mean_w": mean(gpu_power_means),
    }


def summarize_host(root: Path) -> Dict[str, float]:
    csv_path = root / "host_telemetry.csv"
    if not csv_path.exists():
        return {}
    rows = read_csv(csv_path)
    ts = [to_float(r.get("epoch_sec")) for r in rows]
    rss_mib = [to_float(r.get("train_rss_kib")) / 1024.0 for r in rows]
    proc_count = [to_float(r.get("train_proc_count")) for r in rows]

    stable_idx = [i for i, p in enumerate(proc_count) if not math.isnan(p) and p >= 10]
    if stable_idx:
        lo = min(stable_idx)
        hi = max(stable_idx)
        ts_eval = ts[lo : hi + 1]
        rss_eval = rss_mib[lo : hi + 1]
    else:
        ts_eval = ts
        rss_eval = rss_mib

    slope = slope_per_unit(ts_eval, rss_eval) * 60.0
    head, tail = head_tail_mean(rss_eval, ratio=0.1)
    return {
        "host_rss_mean_mib": mean(rss_mib),
        "host_rss_p90_mib": percentile(rss_mib, 0.9),
        "host_rss_tail_slope_mib_min": slope,
        "host_rss_tail_drift_mib": tail - head,
        "host_proc_count_p50": percentile(proc_count, 0.5),
        "host_proc_count_p90": percentile(proc_count, 0.9),
    }


def summarize_timing(root: Path) -> Dict[str, float]:
    csv_path = root / "timing_points.csv"
    if not csv_path.exists():
        return {}
    rows = read_csv(csv_path)
    iter_ms = [to_float(r.get("iter_ms_p50")) for r in rows]
    if not iter_ms:
        return {}
    n = max(1, int(len(iter_ms) * 0.1))
    head = mean(iter_ms[:n])
    tail = mean(iter_ms[-n:])
    drift_pct = math.nan
    if not math.isnan(head) and head > 0:
        drift_pct = (tail / head - 1.0) * 100.0
    return {
        "iter_head_mean_ms": head,
        "iter_tail_mean_ms": tail,
        "iter_tail_vs_head_pct": drift_pct,
    }


def verdict(metrics: Dict[str, float], gpu_slope_thr: float, gpu_drift_thr: float, host_slope_thr: float, host_drift_thr: float, iter_drift_thr_pct: float) -> str:
    flags = []
    gpu_slope = metrics.get("gpu_mem_tail_slope_max_mib_min", math.nan)
    gpu_drift = metrics.get("gpu_mem_tail_drift_max_mib", math.nan)
    host_slope = metrics.get("host_rss_tail_slope_mib_min", math.nan)
    host_drift = metrics.get("host_rss_tail_drift_mib", math.nan)
    iter_drift = metrics.get("iter_tail_vs_head_pct", math.nan)

    if not math.isnan(gpu_slope) and gpu_slope > gpu_slope_thr:
        flags.append(f"gpu_tail_slope>{gpu_slope_thr}")
    if not math.isnan(gpu_drift) and gpu_drift > gpu_drift_thr:
        flags.append(f"gpu_tail_drift>{gpu_drift_thr}")
    if not math.isnan(host_slope) and host_slope > host_slope_thr:
        flags.append(f"host_tail_slope>{host_slope_thr}")
    if not math.isnan(host_drift) and host_drift > host_drift_thr:
        flags.append(f"host_tail_drift>{host_drift_thr}")
    if not math.isnan(iter_drift) and iter_drift > iter_drift_thr_pct:
        flags.append(f"iter_drift>{iter_drift_thr_pct}%")

    if not flags:
        return "no_obvious_leak"
    return "suspected_leak_or_contention(" + ",".join(flags) + ")"


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick leak diagnosis for stage1 bench outputs.")
    parser.add_argument("output_roots", nargs="+", help="One or more benchmark output roots")
    parser.add_argument("--gpu-tail-slope-thr", type=float, default=20.0, help="GPU tail memory slope threshold (MiB/min)")
    parser.add_argument("--gpu-tail-drift-thr", type=float, default=4096.0, help="GPU tail memory drift threshold (MiB)")
    parser.add_argument("--host-tail-slope-thr", type=float, default=100.0, help="Host RSS tail slope threshold (MiB/min)")
    parser.add_argument("--host-tail-drift-thr", type=float, default=8192.0, help="Host RSS tail drift threshold (MiB)")
    parser.add_argument("--iter-drift-thr-pct", type=float, default=20.0, help="Iter tail-vs-head drift threshold (%)")
    args = parser.parse_args()

    header = [
        "output_root",
        "group",
        "status",
        "duration_sec",
        "attn_impl",
        "speech_attn_impl",
        "text_attn_impl",
        "model_load_dtype",
        "tf32_enabled",
        "zero_stage",
        "tail_iter_ms",
        "gpu_mem_tail_slope_max_mib_min",
        "gpu_mem_tail_drift_max_mib",
        "host_rss_tail_slope_mib_min",
        "host_rss_tail_drift_mib",
        "iter_tail_vs_head_pct",
        "verdict",
    ]
    print(",".join(header))

    for root_str in args.output_roots:
        root = Path(root_str)
        manifest_path = root / "run_manifest.tsv"
        if not manifest_path.exists():
            print(f"{root},<missing>,<missing>,,,,,,,,,'manifest_missing'")
            continue

        rows = read_tsv(manifest_path)
        for row in rows:
            run_dir = Path(row.get("run_dir", ""))
            metrics: Dict[str, float] = {}
            metrics.update(summarize_gpu(run_dir))
            metrics.update(summarize_host(root))
            metrics.update(summarize_timing(root))

            verdict_text = verdict(
                metrics,
                gpu_slope_thr=args.gpu_tail_slope_thr,
                gpu_drift_thr=args.gpu_tail_drift_thr,
                host_slope_thr=args.host_tail_slope_thr,
                host_drift_thr=args.host_tail_drift_thr,
                iter_drift_thr_pct=args.iter_drift_thr_pct,
            )

            output = [
                str(root),
                row.get("group", ""),
                row.get("status", ""),
                row.get("duration_sec", ""),
                row.get("attn_impl_effective", ""),
                row.get("speech_attn_impl_effective", row.get("attn_impl_effective", "")),
                row.get("text_attn_impl_effective", row.get("attn_impl_effective", "")),
                row.get("model_load_dtype_effective", ""),
                row.get("tf32_enabled", ""),
                row.get("zero_stage", ""),
                f"{to_float(row.get('last_iter_ms_p50')):.2f}" if not math.isnan(to_float(row.get("last_iter_ms_p50"))) else "",
                f"{metrics.get('gpu_mem_tail_slope_max_mib_min', math.nan):.3f}" if not math.isnan(metrics.get("gpu_mem_tail_slope_max_mib_min", math.nan)) else "",
                f"{metrics.get('gpu_mem_tail_drift_max_mib', math.nan):.1f}" if not math.isnan(metrics.get("gpu_mem_tail_drift_max_mib", math.nan)) else "",
                f"{metrics.get('host_rss_tail_slope_mib_min', math.nan):.3f}" if not math.isnan(metrics.get("host_rss_tail_slope_mib_min", math.nan)) else "",
                f"{metrics.get('host_rss_tail_drift_mib', math.nan):.1f}" if not math.isnan(metrics.get("host_rss_tail_drift_mib", math.nan)) else "",
                f"{metrics.get('iter_tail_vs_head_pct', math.nan):.2f}" if not math.isnan(metrics.get("iter_tail_vs_head_pct", math.nan)) else "",
                verdict_text,
            ]
            print(",".join(output))


if __name__ == "__main__":
    main()
