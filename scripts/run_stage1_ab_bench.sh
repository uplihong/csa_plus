#!/usr/bin/env bash
set -euo pipefail

# End-to-end A/B benchmark runner for Stage1 BF16 on multi-GPU.
# It runs old-like vs tuned configurations for multiple repeats, then writes:
# - per-run metrics
# - per-timing-point raw records
# - group-level statistics
# - markdown summary
# All artifacts are saved under outputs/.

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/bench_stage1_ab_${TIMESTAMP}}"

EXPERIMENT="${EXPERIMENT:-limit_longest_1-3_stage1_bf16}"
INCLUDE="${INCLUDE:-localhost:0,1,2,3}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29650}"
REPEATS="${REPEATS:-3}"
MAX_STEPS="${MAX_STEPS:-5000}"
MICRO_BATCH="${MICRO_BATCH:-128}"
TAIL_TIMING_POINTS="${TAIL_TIMING_POINTS:-20}"
STOP_ON_ERROR="${STOP_ON_ERROR:-1}"
CONDA_ENV="${CONDA_ENV:-}"

DATASET_ROOT="${DATASET_ROOT:-/code/data/LibriSpeech/LibriSpeech}"
PRETRAINED_CKPT="${PRETRAINED_CKPT:-/code/data/weights/csa/ckpt_epoch_8.pth}"
SPEECH_MODEL_PATH="${SPEECH_MODEL_PATH:-/code/data/weights/wav2vec2-base}"
TEXT_MODEL_PATH="${TEXT_MODEL_PATH:-/code/data/weights/bert-base-uncased}"

mkdir -p "${OUTPUT_ROOT}"
MANIFEST_PATH="${OUTPUT_ROOT}/run_manifest.tsv"

echo -e "group\trepeat\tstatus\tport\trun_dir\tlauncher_log\ttrain_log" > "${MANIFEST_PATH}"

if ! [[ "${REPEATS}" =~ ^[0-9]+$ ]] || [[ "${REPEATS}" -lt 1 ]]; then
  echo "REPEATS must be a positive integer, got: ${REPEATS}" >&2
  exit 1
fi
if ! [[ "${MASTER_PORT_BASE}" =~ ^[0-9]+$ ]]; then
  echo "MASTER_PORT_BASE must be an integer, got: ${MASTER_PORT_BASE}" >&2
  exit 1
fi
if ! [[ "${MAX_STEPS}" =~ ^[0-9]+$ ]] || [[ "${MAX_STEPS}" -lt 1 ]]; then
  echo "MAX_STEPS must be a positive integer, got: ${MAX_STEPS}" >&2
  exit 1
fi
if ! [[ "${MICRO_BATCH}" =~ ^[0-9]+$ ]] || [[ "${MICRO_BATCH}" -lt 1 ]]; then
  echo "MICRO_BATCH must be a positive integer, got: ${MICRO_BATCH}" >&2
  exit 1
fi
if ! [[ "${TAIL_TIMING_POINTS}" =~ ^[0-9]+$ ]] || [[ "${TAIL_TIMING_POINTS}" -lt 1 ]]; then
  echo "TAIL_TIMING_POINTS must be a positive integer, got: ${TAIL_TIMING_POINTS}" >&2
  exit 1
fi

WORLD_SIZE="${WORLD_SIZE:-0}"
if ! [[ "${WORLD_SIZE}" =~ ^[0-9]+$ ]]; then
  echo "WORLD_SIZE must be an integer, got: ${WORLD_SIZE}" >&2
  exit 1
fi
if [[ "${WORLD_SIZE}" -eq 0 ]]; then
  INCLUDE_AFTER_COLON="${INCLUDE#*:}"
  if [[ "${INCLUDE_AFTER_COLON}" != "${INCLUDE}" ]]; then
    IFS=',' read -r -a GPU_IDS <<< "${INCLUDE_AFTER_COLON}"
    WORLD_SIZE="${#GPU_IDS[@]}"
  fi
fi
GLOBAL_BATCH=0
if [[ "${WORLD_SIZE}" -gt 0 ]]; then
  GLOBAL_BATCH=$((MICRO_BATCH * WORLD_SIZE))
fi

PORT_COUNTER=0

run_case() {
  local group="$1"
  local repeat="$2"
  local log_every="$3"
  local val_every="$4"
  local ckpt_every="$5"
  local deterministic="$6"
  local cudnn_benchmark="$7"
  local wall_clock_breakdown="$8"

  local run_dir="${OUTPUT_ROOT}/${group}_r${repeat}"
  local launcher_log="${run_dir}/launcher.log"
  local train_log="${run_dir}/train.log"
  mkdir -p "${run_dir}"

  local port=$((MASTER_PORT_BASE + PORT_COUNTER))
  PORT_COUNTER=$((PORT_COUNTER + 1))

  local cmd=(
    deepspeed
    --include "${INCLUDE}"
    --master_port "${port}"
    train.py
    "+experiment=${EXPERIMENT}"
    "experiment_output_dir=${run_dir}"
    "hydra.run.dir=${run_dir}"
    "++train.max_step_iterations=${MAX_STEPS}"
    "++train.log_every_steps=${log_every}"
    "++train.validation_every_steps=${val_every}"
    "++train.checkpoint_every_steps=${ckpt_every}"
    "++train.deterministic=${deterministic}"
    "++train.cudnn_benchmark=${cudnn_benchmark}"
    "++train.pretrained_model_checkpoint=${PRETRAINED_CKPT}"
    "++dataset.root_dir=${DATASET_ROOT}"
    "++model.speech_encoder.pretrained_path=${SPEECH_MODEL_PATH}"
    "++model.text_encoder.pretrained_path=${TEXT_MODEL_PATH}"
    "deepspeed_config_yaml.train_micro_batch_size_per_gpu=${MICRO_BATCH}"
    "deepspeed_config_yaml.wall_clock_breakdown=${wall_clock_breakdown}"
    "deepspeed_config_yaml.bf16.enabled=true"
    "deepspeed_config_yaml.fp16.enabled=false"
  )

  local status="success"
  {
    echo "=== META ==="
    echo "timestamp=$(date -Is)"
    echo "group=${group}"
    echo "repeat=${repeat}"
    echo "port=${port}"
    echo "run_dir=${run_dir}"
    echo "=== CMD ==="
    printf '%q ' "${cmd[@]}"
    echo
    echo "============"
    if [[ -n "${CONDA_ENV}" ]]; then
      conda run -n "${CONDA_ENV}" "${cmd[@]}"
    else
      "${cmd[@]}"
    fi
  } > "${launcher_log}" 2>&1 || status="failed"

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${group}" "${repeat}" "${status}" "${port}" "${run_dir}" "${launcher_log}" "${train_log}" >> "${MANIFEST_PATH}"

  if [[ "${status}" == "failed" ]]; then
    echo "[WARN] ${group} repeat ${repeat} failed, see ${launcher_log}" >&2
    if [[ "${STOP_ON_ERROR}" == "1" ]]; then
      exit 1
    fi
  fi
}

echo "[INFO] Output root: ${OUTPUT_ROOT}"
echo "[INFO] Running A/B benchmark with REPEATS=${REPEATS}, MAX_STEPS=${MAX_STEPS}, MICRO_BATCH=${MICRO_BATCH}"
echo "[INFO] INCLUDE=${INCLUDE}, WORLD_SIZE=${WORLD_SIZE}, GLOBAL_BATCH=${GLOBAL_BATCH}"

for repeat in $(seq 1 "${REPEATS}"); do
  # A: old-like setup from the original slow config.
  run_case "old_like" "${repeat}" 1 500 500 true false true
  # B: tuned setup for throughput.
  run_case "tuned" "${repeat}" 20 5000 5000 false false false
done

PYTHON_CMD=(python)
if [[ -n "${CONDA_ENV}" ]]; then
  PYTHON_CMD=(conda run -n "${CONDA_ENV}" python)
fi

"${PYTHON_CMD[@]}" - "${MANIFEST_PATH}" "${OUTPUT_ROOT}" "${TAIL_TIMING_POINTS}" "${GLOBAL_BATCH}" <<'PY'
import csv
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

manifest_path = Path(sys.argv[1])
output_root = Path(sys.argv[2])
tail_points = int(sys.argv[3])
global_batch = int(sys.argv[4])

timing_keys = ["data_wait_ms", "preprocess_ms", "fwd_ms", "bwd_ms", "step_ms", "iter_ms"]
timing_patterns = {k: re.compile(rf"{k}:p50=([0-9]+(?:\.[0-9]+)?)") for k in timing_keys}
step_pattern = re.compile(r"Step\s+([0-9]+):")

per_run_rows = []
timing_rows = []
skipped_rows = []

with manifest_path.open("r", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    manifest_rows = list(reader)

for row in manifest_rows:
    group = row["group"]
    repeat = int(row["repeat"])
    status = row["status"]
    train_log = Path(row["train_log"])
    launcher_log = Path(row["launcher_log"])
    run_dir = row["run_dir"]

    if status != "success":
        skipped_rows.append(
            {"group": group, "repeat": repeat, "reason": "run_failed", "train_log": str(train_log)}
        )
        continue
    if not train_log.exists():
        skipped_rows.append(
            {"group": group, "repeat": repeat, "reason": "train_log_missing", "train_log": str(train_log)}
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
            for k in timing_keys:
                m = timing_patterns[k].search(line)
                if not m:
                    ok = False
                    break
                values[k] = float(m.group(1))
            if ok:
                values["step"] = last_step
                points.append(values)

    if not points:
        skipped_rows.append(
            {"group": group, "repeat": repeat, "reason": "no_timing_points", "train_log": str(train_log)}
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
    samples_per_sec_tail = (global_batch * 1000.0 / tail_mean_iter) if global_batch > 0 else math.nan

    per_run_rows.append(
        {
            "group": group,
            "repeat": repeat,
            "run_dir": run_dir,
            "train_log": str(train_log),
            "launcher_log": str(launcher_log),
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
    )

    for idx, p in enumerate(points, start=1):
        timing_rows.append(
            {
                "group": group,
                "repeat": repeat,
                "index": idx,
                "step": p["step"],
                "data_wait_ms_p50": p["data_wait_ms"],
                "preprocess_ms_p50": p["preprocess_ms"],
                "fwd_ms_p50": p["fwd_ms"],
                "bwd_ms_p50": p["bwd_ms"],
                "step_ms_p50": p["step_ms"],
                "iter_ms_p50": p["iter_ms"],
                "train_log": str(train_log),
            }
        )

per_run_rows.sort(key=lambda x: (x["group"], x["repeat"]))
timing_rows.sort(key=lambda x: (x["group"], x["repeat"], x["index"]))

per_run_csv = output_root / "per_run_metrics.csv"
timing_csv = output_root / "timing_points.csv"
group_csv = output_root / "group_summary.csv"
summary_json = output_root / "summary.json"
summary_md = output_root / "summary.md"
skipped_tsv = output_root / "skipped_runs.tsv"

with per_run_csv.open("w", encoding="utf-8", newline="") as f:
    if per_run_rows:
        writer = csv.DictWriter(f, fieldnames=list(per_run_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_run_rows)
    else:
        f.write("")

with timing_csv.open("w", encoding="utf-8", newline="") as f:
    if timing_rows:
        writer = csv.DictWriter(f, fieldnames=list(timing_rows[0].keys()))
        writer.writeheader()
        writer.writerows(timing_rows)
    else:
        f.write("")

grouped = defaultdict(list)
for row in per_run_rows:
    grouped[row["group"]].append(row)

baseline_name = "old_like"
baseline_mean = math.nan
if grouped.get(baseline_name):
    baseline_mean = statistics.mean([r["tail_mean_iter_p50_ms"] for r in grouped[baseline_name]])

group_rows = []
for group, rows in sorted(grouped.items()):
    vals = [r["tail_mean_iter_p50_ms"] for r in rows]
    steps_vals = [r["tail_steps_per_sec"] for r in rows]
    samples_vals = [r["tail_samples_per_sec"] for r in rows if not math.isnan(r["tail_samples_per_sec"])]
    mean_iter = statistics.mean(vals)
    std_iter = statistics.stdev(vals) if len(vals) > 1 else 0.0
    row = {
        "group": group,
        "runs": len(rows),
        "tail_mean_iter_p50_ms_mean": mean_iter,
        "tail_mean_iter_p50_ms_std": std_iter,
        "tail_mean_iter_p50_ms_min": min(vals),
        "tail_mean_iter_p50_ms_max": max(vals),
        "tail_steps_per_sec_mean": statistics.mean(steps_vals),
        "tail_steps_per_sec_std": statistics.stdev(steps_vals) if len(steps_vals) > 1 else 0.0,
        "tail_samples_per_sec_mean": statistics.mean(samples_vals) if samples_vals else math.nan,
        "tail_samples_per_sec_std": statistics.stdev(samples_vals) if len(samples_vals) > 1 else 0.0,
        "speedup_vs_old_like": (baseline_mean / mean_iter) if not math.isnan(baseline_mean) else math.nan,
    }
    group_rows.append(row)

with group_csv.open("w", encoding="utf-8", newline="") as f:
    if group_rows:
        writer = csv.DictWriter(f, fieldnames=list(group_rows[0].keys()))
        writer.writeheader()
        writer.writerows(group_rows)
    else:
        f.write("")

with skipped_tsv.open("w", encoding="utf-8", newline="") as f:
    if skipped_rows:
        f.write("group\trepeat\treason\ttrain_log\n")
        for row in skipped_rows:
            f.write(f"{row['group']}\t{row['repeat']}\t{row['reason']}\t{row['train_log']}\n")

summary_payload = {
    "generated_at": datetime.now().isoformat(),
    "output_root": str(output_root),
    "tail_points": tail_points,
    "global_batch": global_batch,
    "groups": group_rows,
    "runs": per_run_rows,
    "skipped_runs": skipped_rows,
    "files": {
        "manifest": str(manifest_path),
        "per_run_metrics_csv": str(per_run_csv),
        "timing_points_csv": str(timing_csv),
        "group_summary_csv": str(group_csv),
        "skipped_runs_tsv": str(skipped_tsv),
    },
}

def json_safe(obj):
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    return obj

with summary_json.open("w", encoding="utf-8") as f:
    json.dump(json_safe(summary_payload), f, indent=2, ensure_ascii=False, allow_nan=False)

lines = []
lines.append("# Stage1 BF16 A/B Benchmark Report")
lines.append("")
lines.append(f"- Generated at: `{summary_payload['generated_at']}`")
lines.append(f"- Output root: `{output_root}`")
lines.append(f"- Tail timing points used per run: `{tail_points}`")
lines.append(f"- Global batch (for samples/s): `{global_batch}`")
lines.append("")
lines.append("## Group Summary")
lines.append("")
if group_rows:
    lines.append("| group | runs | tail_mean_iter_p50_ms_mean | tail_mean_iter_p50_ms_std | tail_steps_per_sec_mean | tail_samples_per_sec_mean | speedup_vs_old_like |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in group_rows:
        sps = "nan" if math.isnan(r["tail_samples_per_sec_mean"]) else f"{r['tail_samples_per_sec_mean']:.4f}"
        spd = "nan" if math.isnan(r["speedup_vs_old_like"]) else f"{r['speedup_vs_old_like']:.4f}"
        lines.append(
            f"| {r['group']} | {r['runs']} | {r['tail_mean_iter_p50_ms_mean']:.4f} | {r['tail_mean_iter_p50_ms_std']:.4f} | "
            f"{r['tail_steps_per_sec_mean']:.4f} | {sps} | {spd} |"
        )
else:
    lines.append("No successful runs found.")

lines.append("")
lines.append("## Per-Run Files")
lines.append("")
lines.append(f"- Manifest: `{manifest_path}`")
lines.append(f"- Per-run metrics: `{per_run_csv}`")
lines.append(f"- Raw timing points: `{timing_csv}`")
lines.append(f"- Group summary (CSV): `{group_csv}`")
lines.append(f"- Group summary (JSON): `{summary_json}`")
if skipped_rows:
    lines.append(f"- Skipped runs: `{skipped_tsv}`")
lines.append("")
lines.append("## Per-Run Metrics")
lines.append("")
if per_run_rows:
    lines.append("| group | repeat | tail_mean_iter_p50_ms | tail_std_iter_p50_ms | tail_steps_per_sec | tail_samples_per_sec | train_log |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for r in per_run_rows:
        sps = "nan" if math.isnan(r["tail_samples_per_sec"]) else f"{r['tail_samples_per_sec']:.4f}"
        lines.append(
            f"| {r['group']} | {r['repeat']} | {r['tail_mean_iter_p50_ms']:.4f} | {r['tail_std_iter_p50_ms']:.4f} | "
            f"{r['tail_steps_per_sec']:.4f} | {sps} | `{r['train_log']}` |"
        )
else:
    lines.append("No successful runs found.")

summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

print(f"[INFO] Wrote: {per_run_csv}")
print(f"[INFO] Wrote: {timing_csv}")
print(f"[INFO] Wrote: {group_csv}")
print(f"[INFO] Wrote: {summary_json}")
print(f"[INFO] Wrote: {summary_md}")
if skipped_rows:
    print(f"[WARN] Some runs were skipped. Details: {skipped_tsv}")
PY

echo "[INFO] Benchmark finished."
echo "[INFO] Manifest: ${MANIFEST_PATH}"
echo "[INFO] Report: ${OUTPUT_ROOT}/summary.md"
