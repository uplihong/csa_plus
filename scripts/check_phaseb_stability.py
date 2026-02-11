#!/usr/bin/env python3
import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def parse_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def parse_optional_bool(value):
    if value is None:
        return None
    text = str(value).strip().lower()
    if text == "":
        return None
    return text in {"1", "true", "yes", "y"}


def split_csv_items(raw):
    items = []
    for token in str(raw).split(","):
        token = token.strip()
        if token:
            items.append(token)
    return items


def cv(values):
    if not values:
        return math.nan
    mean_v = statistics.mean(values)
    if mean_v <= 0:
        return math.nan
    if len(values) <= 1:
        return 0.0
    return statistics.stdev(values) / mean_v


def main():
    parser = argparse.ArgumentParser(description="Check Phase-B stability acceptance from run_manifest.tsv.")
    parser.add_argument(
        "--output-root",
        required=True,
        help="Benchmark output directory containing run_manifest.tsv",
    )
    parser.add_argument(
        "--manifest",
        default="",
        help="Optional explicit manifest path. Defaults to <output-root>/run_manifest.tsv",
    )
    parser.add_argument(
        "--groups",
        default="sweep_z0_mb128_nw6_pf4,sweep_z1_mb160_nw6_pf2",
        help="Comma-separated required sentinel groups",
    )
    parser.add_argument("--min-repeats", type=int, default=3)
    parser.add_argument("--max-iter-ratio-median", type=float, default=1.5)
    parser.add_argument("--max-iter-ratio-max", type=float, default=2.5)
    parser.add_argument("--max-iter-p50-cv", type=float, default=0.05)
    parser.add_argument(
        "--max-telemetry-empty-ratio",
        type=float,
        default=1.0,
        help="Set <1.0 to enforce telemetry quality; default 1.0 means do not fail on empty telemetry.",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root)
    manifest_path = Path(args.manifest) if args.manifest else output_root / "run_manifest.tsv"
    required_groups = split_csv_items(args.groups)

    if not manifest_path.exists():
        print(f"[ERROR] manifest not found: {manifest_path}", file=sys.stderr)
        return 2

    with manifest_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))

    success_rows = [r for r in rows if r.get("status") == "success"]
    by_group = defaultdict(list)
    for r in success_rows:
        by_group[r.get("group", "")].append(r)

    failed_reasons = []
    summary_lines = []

    header = (
        "group | repeats | iter_ratio_median | iter_ratio_max | iter_p50_cv | "
        "telemetry_empty_ratio | pass"
    )
    summary_lines.append(header)
    summary_lines.append("-" * len(header))

    for group in required_groups:
        group_rows = by_group.get(group, [])
        iter_ratios = [parse_float(r.get("iter_p90_over_p50", "")) for r in group_rows]
        iter_ratios = [v for v in iter_ratios if not math.isnan(v)]
        iter_p50 = [parse_float(r.get("last_iter_ms_p50", "")) for r in group_rows]
        iter_p50 = [v for v in iter_p50 if not math.isnan(v)]
        telemetry_empty_flags = [parse_optional_bool(r.get("gpu_telemetry_empty_flag", "")) for r in group_rows]
        telemetry_empty_flags = [x for x in telemetry_empty_flags if x is not None]
        telemetry_empty_ratio = (
            sum(1 for x in telemetry_empty_flags if x) / len(telemetry_empty_flags)
            if telemetry_empty_flags
            else math.nan
        )

        group_pass = True
        reasons = []

        if len(group_rows) < args.min_repeats:
            group_pass = False
            reasons.append(f"repeats {len(group_rows)} < {args.min_repeats}")
        if not iter_ratios:
            group_pass = False
            reasons.append("missing iter_p90_over_p50")
            iter_ratio_median = math.nan
            iter_ratio_max = math.nan
        else:
            iter_ratio_median = statistics.median(iter_ratios)
            iter_ratio_max = max(iter_ratios)
            if iter_ratio_median > args.max_iter_ratio_median:
                group_pass = False
                reasons.append(
                    f"iter_ratio_median {iter_ratio_median:.3f} > {args.max_iter_ratio_median:.3f}"
                )
            if iter_ratio_max > args.max_iter_ratio_max:
                group_pass = False
                reasons.append(f"iter_ratio_max {iter_ratio_max:.3f} > {args.max_iter_ratio_max:.3f}")

        iter_p50_cv = cv(iter_p50)
        if not math.isnan(iter_p50_cv) and iter_p50_cv > args.max_iter_p50_cv:
            group_pass = False
            reasons.append(f"iter_p50_cv {iter_p50_cv:.4f} > {args.max_iter_p50_cv:.4f}")

        if not math.isnan(telemetry_empty_ratio) and telemetry_empty_ratio > args.max_telemetry_empty_ratio:
            group_pass = False
            reasons.append(
                f"telemetry_empty_ratio {telemetry_empty_ratio:.3f} > {args.max_telemetry_empty_ratio:.3f}"
            )

        pass_flag = "PASS" if group_pass else "FAIL"
        summary_lines.append(
            f"{group} | {len(group_rows)} | "
            f"{'nan' if math.isnan(iter_ratio_median) else f'{iter_ratio_median:.3f}'} | "
            f"{'nan' if math.isnan(iter_ratio_max) else f'{iter_ratio_max:.3f}'} | "
            f"{'nan' if math.isnan(iter_p50_cv) else f'{iter_p50_cv:.4f}'} | "
            f"{'nan' if math.isnan(telemetry_empty_ratio) else f'{telemetry_empty_ratio:.3f}'} | {pass_flag}"
        )
        if reasons:
            failed_reasons.append(f"{group}: " + "; ".join(reasons))

    for line in summary_lines:
        print(line)

    if failed_reasons:
        print("")
        print("[FAIL] Phase-B stability acceptance failed:")
        for reason in failed_reasons:
            print(f"- {reason}")
        return 2

    print("")
    print("[PASS] Phase-B stability acceptance passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
