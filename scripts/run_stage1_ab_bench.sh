#!/usr/bin/env bash
set -euo pipefail

# Unified benchmark runner for Stage1 BF16 on multi-GPU.
# Modes:
#   1) ab    : old_like vs tuned A/B comparison (default)
#   2) sweep : grid search over zero stage x micro batch
#   3) both  : run ab first, then sweep
#
# Artifacts are saved under outputs/:
# - run_manifest.tsv
# - per_run_metrics.csv
# - timing_points.csv
# - group_summary.csv
# - ranked_groups.csv
# - best_config.json
# - best_config.env
# - best_overrides.txt
# - summary.json
# - summary.md

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/bench_stage1_ab_${TIMESTAMP}}"

MODE="${MODE:-ab}"                # ab | sweep | both
EXPERIMENT="${EXPERIMENT:-limit_longest_1-3_stage1_bf16}"
INCLUDE="${INCLUDE:-localhost:0,1,2,3}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29650}"
REPEATS="${REPEATS:-3}"
MAX_STEPS="${MAX_STEPS:-5000}"
TAIL_TIMING_POINTS="${TAIL_TIMING_POINTS:-20}"
STOP_ON_ERROR="${STOP_ON_ERROR:-1}"
CONDA_ENV="${CONDA_ENV:-}"

DATASET_ROOT="${DATASET_ROOT:-/code/data/LibriSpeech/LibriSpeech}"
PRETRAINED_CKPT="${PRETRAINED_CKPT:-/code/data/weights/csa/ckpt_epoch_8.pth}"
SPEECH_MODEL_PATH="${SPEECH_MODEL_PATH:-/code/data/weights/wav2vec2-base}"
TEXT_MODEL_PATH="${TEXT_MODEL_PATH:-/code/data/weights/bert-base-uncased}"

# Baseline group for speedup metric in reports.
BASELINE_GROUP="${BASELINE_GROUP:-old_like}"

# AB mode params.
AB_MICRO_BATCH="${AB_MICRO_BATCH:-128}"
AB_OLD_LOG_EVERY="${AB_OLD_LOG_EVERY:-1}"
AB_OLD_VALIDATION_EVERY="${AB_OLD_VALIDATION_EVERY:-500}"
AB_OLD_CHECKPOINT_EVERY="${AB_OLD_CHECKPOINT_EVERY:-500}"
AB_OLD_DETERMINISTIC="${AB_OLD_DETERMINISTIC:-true}"
AB_OLD_CUDNN_BENCHMARK="${AB_OLD_CUDNN_BENCHMARK:-false}"
AB_OLD_WALL_CLOCK_BREAKDOWN="${AB_OLD_WALL_CLOCK_BREAKDOWN:-true}"
AB_OLD_ZERO_STAGE="${AB_OLD_ZERO_STAGE:-1}"
AB_OLD_NUM_WORKERS="${AB_OLD_NUM_WORKERS:-2}"
AB_OLD_PREFETCH_FACTOR="${AB_OLD_PREFETCH_FACTOR:-2}"

AB_TUNED_LOG_EVERY="${AB_TUNED_LOG_EVERY:-20}"
AB_TUNED_VALIDATION_EVERY="${AB_TUNED_VALIDATION_EVERY:-5000}"
AB_TUNED_CHECKPOINT_EVERY="${AB_TUNED_CHECKPOINT_EVERY:-5000}"
AB_TUNED_DETERMINISTIC="${AB_TUNED_DETERMINISTIC:-false}"
AB_TUNED_CUDNN_BENCHMARK="${AB_TUNED_CUDNN_BENCHMARK:-false}"
AB_TUNED_WALL_CLOCK_BREAKDOWN="${AB_TUNED_WALL_CLOCK_BREAKDOWN:-false}"
AB_TUNED_ZERO_STAGE="${AB_TUNED_ZERO_STAGE:-1}"
AB_TUNED_NUM_WORKERS="${AB_TUNED_NUM_WORKERS:-2}"
AB_TUNED_PREFETCH_FACTOR="${AB_TUNED_PREFETCH_FACTOR:-2}"

# Sweep mode params.
SWEEP_ZERO_STAGES="${SWEEP_ZERO_STAGES:-0,1}"
SWEEP_MICRO_BATCHES="${SWEEP_MICRO_BATCHES:-96,128,160,192}"
SWEEP_LOG_EVERY="${SWEEP_LOG_EVERY:-20}"
SWEEP_VALIDATION_EVERY="${SWEEP_VALIDATION_EVERY:-5000}"
SWEEP_CHECKPOINT_EVERY="${SWEEP_CHECKPOINT_EVERY:-5000}"
SWEEP_DETERMINISTIC="${SWEEP_DETERMINISTIC:-false}"
SWEEP_CUDNN_BENCHMARK="${SWEEP_CUDNN_BENCHMARK:-false}"
SWEEP_WALL_CLOCK_BREAKDOWN="${SWEEP_WALL_CLOCK_BREAKDOWN:-false}"
SWEEP_NUM_WORKERS="${SWEEP_NUM_WORKERS:-2}"
SWEEP_PREFETCH_FACTOR="${SWEEP_PREFETCH_FACTOR:-2}"

mkdir -p "${OUTPUT_ROOT}"
MANIFEST_PATH="${OUTPUT_ROOT}/run_manifest.tsv"

echo -e "mode\tgroup\trepeat\tstatus\tport\trun_dir\tlauncher_log\ttrain_log\tzero_stage\tmicro_batch\tworld_size\tglobal_batch\tdeterministic\tcudnn_benchmark\twall_clock_breakdown\tlog_every_steps\tvalidation_every_steps\tcheckpoint_every_steps\tnum_workers\tprefetch_factor" > "${MANIFEST_PATH}"

is_pos_int() {
  [[ "$1" =~ ^[0-9]+$ ]] && [[ "$1" -gt 0 ]]
}

is_nonneg_int() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

is_bool() {
  [[ "$1" == "true" || "$1" == "false" ]]
}

require_pos_int() {
  local name="$1"
  local value="$2"
  if ! is_pos_int "${value}"; then
    echo "${name} must be a positive integer, got: ${value}" >&2
    exit 1
  fi
}

require_nonneg_int() {
  local name="$1"
  local value="$2"
  if ! is_nonneg_int "${value}"; then
    echo "${name} must be a non-negative integer, got: ${value}" >&2
    exit 1
  fi
}

require_bool() {
  local name="$1"
  local value="$2"
  if ! is_bool "${value}"; then
    echo "${name} must be true/false, got: ${value}" >&2
    exit 1
  fi
}

require_pos_int "REPEATS" "${REPEATS}"
require_pos_int "MASTER_PORT_BASE" "${MASTER_PORT_BASE}"
require_pos_int "MAX_STEPS" "${MAX_STEPS}"
require_pos_int "TAIL_TIMING_POINTS" "${TAIL_TIMING_POINTS}"
require_pos_int "AB_MICRO_BATCH" "${AB_MICRO_BATCH}"

require_bool "AB_OLD_DETERMINISTIC" "${AB_OLD_DETERMINISTIC}"
require_bool "AB_OLD_CUDNN_BENCHMARK" "${AB_OLD_CUDNN_BENCHMARK}"
require_bool "AB_OLD_WALL_CLOCK_BREAKDOWN" "${AB_OLD_WALL_CLOCK_BREAKDOWN}"
require_bool "AB_TUNED_DETERMINISTIC" "${AB_TUNED_DETERMINISTIC}"
require_bool "AB_TUNED_CUDNN_BENCHMARK" "${AB_TUNED_CUDNN_BENCHMARK}"
require_bool "AB_TUNED_WALL_CLOCK_BREAKDOWN" "${AB_TUNED_WALL_CLOCK_BREAKDOWN}"
require_bool "SWEEP_DETERMINISTIC" "${SWEEP_DETERMINISTIC}"
require_bool "SWEEP_CUDNN_BENCHMARK" "${SWEEP_CUDNN_BENCHMARK}"
require_bool "SWEEP_WALL_CLOCK_BREAKDOWN" "${SWEEP_WALL_CLOCK_BREAKDOWN}"

require_pos_int "AB_OLD_LOG_EVERY" "${AB_OLD_LOG_EVERY}"
require_pos_int "AB_OLD_VALIDATION_EVERY" "${AB_OLD_VALIDATION_EVERY}"
require_pos_int "AB_OLD_CHECKPOINT_EVERY" "${AB_OLD_CHECKPOINT_EVERY}"
require_pos_int "AB_TUNED_LOG_EVERY" "${AB_TUNED_LOG_EVERY}"
require_pos_int "AB_TUNED_VALIDATION_EVERY" "${AB_TUNED_VALIDATION_EVERY}"
require_pos_int "AB_TUNED_CHECKPOINT_EVERY" "${AB_TUNED_CHECKPOINT_EVERY}"

require_nonneg_int "AB_OLD_ZERO_STAGE" "${AB_OLD_ZERO_STAGE}"
require_nonneg_int "AB_TUNED_ZERO_STAGE" "${AB_TUNED_ZERO_STAGE}"
require_pos_int "AB_OLD_NUM_WORKERS" "${AB_OLD_NUM_WORKERS}"
require_pos_int "AB_OLD_PREFETCH_FACTOR" "${AB_OLD_PREFETCH_FACTOR}"
require_pos_int "AB_TUNED_NUM_WORKERS" "${AB_TUNED_NUM_WORKERS}"
require_pos_int "AB_TUNED_PREFETCH_FACTOR" "${AB_TUNED_PREFETCH_FACTOR}"

require_pos_int "SWEEP_LOG_EVERY" "${SWEEP_LOG_EVERY}"
require_pos_int "SWEEP_VALIDATION_EVERY" "${SWEEP_VALIDATION_EVERY}"
require_pos_int "SWEEP_CHECKPOINT_EVERY" "${SWEEP_CHECKPOINT_EVERY}"
require_pos_int "SWEEP_NUM_WORKERS" "${SWEEP_NUM_WORKERS}"
require_pos_int "SWEEP_PREFETCH_FACTOR" "${SWEEP_PREFETCH_FACTOR}"

if [[ "${MODE}" != "ab" && "${MODE}" != "sweep" && "${MODE}" != "both" ]]; then
  echo "MODE must be one of: ab, sweep, both. got: ${MODE}" >&2
  exit 1
fi

WORLD_SIZE="${WORLD_SIZE:-0}"
require_nonneg_int "WORLD_SIZE" "${WORLD_SIZE}"
if [[ "${WORLD_SIZE}" -eq 0 ]]; then
  INCLUDE_AFTER_COLON="${INCLUDE#*:}"
  if [[ "${INCLUDE_AFTER_COLON}" != "${INCLUDE}" ]]; then
    IFS=',' read -r -a GPU_IDS <<< "${INCLUDE_AFTER_COLON}"
    WORLD_SIZE="${#GPU_IDS[@]}"
  fi
fi
if [[ "${WORLD_SIZE}" -eq 0 ]]; then
  echo "[WARN] WORLD_SIZE unresolved from INCLUDE=${INCLUDE}. samples/s will be NaN." >&2
fi

PORT_COUNTER=0

run_case() {
  local mode_name="$1"
  local group="$2"
  local repeat="$3"
  local log_every="$4"
  local val_every="$5"
  local ckpt_every="$6"
  local deterministic="$7"
  local cudnn_benchmark="$8"
  local wall_clock_breakdown="$9"
  local micro_batch="${10}"
  local zero_stage="${11}"
  local num_workers="${12}"
  local prefetch_factor="${13}"

  local run_dir="${OUTPUT_ROOT}/${group}_r${repeat}"
  local launcher_log="${run_dir}/launcher.log"
  local train_log="${run_dir}/train.log"
  mkdir -p "${run_dir}"

  local port=$((MASTER_PORT_BASE + PORT_COUNTER))
  PORT_COUNTER=$((PORT_COUNTER + 1))

  local global_batch=""
  if [[ "${WORLD_SIZE}" -gt 0 ]]; then
    global_batch=$((micro_batch * WORLD_SIZE))
  fi

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
    "++train.data.num_workers=${num_workers}"
    "++train.data.prefetch_factor=${prefetch_factor}"
    "++train.pretrained_model_checkpoint=${PRETRAINED_CKPT}"
    "++dataset.root_dir=${DATASET_ROOT}"
    "++model.speech_encoder.pretrained_path=${SPEECH_MODEL_PATH}"
    "++model.text_encoder.pretrained_path=${TEXT_MODEL_PATH}"
    "deepspeed_config_yaml.train_micro_batch_size_per_gpu=${micro_batch}"
    "deepspeed_config_yaml.zero_optimization.stage=${zero_stage}"
    "deepspeed_config_yaml.wall_clock_breakdown=${wall_clock_breakdown}"
    "deepspeed_config_yaml.bf16.enabled=true"
    "deepspeed_config_yaml.fp16.enabled=false"
  )

  local status="success"
  {
    echo "=== META ==="
    echo "timestamp=$(date -Is)"
    echo "mode=${mode_name}"
    echo "group=${group}"
    echo "repeat=${repeat}"
    echo "port=${port}"
    echo "zero_stage=${zero_stage}"
    echo "micro_batch=${micro_batch}"
    echo "world_size=${WORLD_SIZE}"
    echo "global_batch=${global_batch}"
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

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${mode_name}" "${group}" "${repeat}" "${status}" "${port}" "${run_dir}" "${launcher_log}" "${train_log}" \
    "${zero_stage}" "${micro_batch}" "${WORLD_SIZE}" "${global_batch}" \
    "${deterministic}" "${cudnn_benchmark}" "${wall_clock_breakdown}" \
    "${log_every}" "${val_every}" "${ckpt_every}" "${num_workers}" "${prefetch_factor}" >> "${MANIFEST_PATH}"

  if [[ "${status}" == "failed" ]]; then
    echo "[WARN] ${group} repeat ${repeat} failed, see ${launcher_log}" >&2
    if [[ "${STOP_ON_ERROR}" == "1" ]]; then
      exit 1
    fi
  fi
}

split_csv_to_array() {
  local csv="$1"
  local -n out_arr="$2"
  IFS=',' read -r -a out_arr <<< "${csv}"
  if [[ "${#out_arr[@]}" -eq 0 ]]; then
    echo "CSV list is empty: ${csv}" >&2
    exit 1
  fi
}

echo "[INFO] Output root: ${OUTPUT_ROOT}"
echo "[INFO] Mode=${MODE}, REPEATS=${REPEATS}, MAX_STEPS=${MAX_STEPS}, INCLUDE=${INCLUDE}, WORLD_SIZE=${WORLD_SIZE}"

if [[ "${MODE}" == "ab" || "${MODE}" == "both" ]]; then
  echo "[INFO] Running AB mode..."
  for repeat in $(seq 1 "${REPEATS}"); do
    run_case "ab" "old_like" "${repeat}" \
      "${AB_OLD_LOG_EVERY}" "${AB_OLD_VALIDATION_EVERY}" "${AB_OLD_CHECKPOINT_EVERY}" \
      "${AB_OLD_DETERMINISTIC}" "${AB_OLD_CUDNN_BENCHMARK}" "${AB_OLD_WALL_CLOCK_BREAKDOWN}" \
      "${AB_MICRO_BATCH}" "${AB_OLD_ZERO_STAGE}" "${AB_OLD_NUM_WORKERS}" "${AB_OLD_PREFETCH_FACTOR}"

    run_case "ab" "tuned" "${repeat}" \
      "${AB_TUNED_LOG_EVERY}" "${AB_TUNED_VALIDATION_EVERY}" "${AB_TUNED_CHECKPOINT_EVERY}" \
      "${AB_TUNED_DETERMINISTIC}" "${AB_TUNED_CUDNN_BENCHMARK}" "${AB_TUNED_WALL_CLOCK_BREAKDOWN}" \
      "${AB_MICRO_BATCH}" "${AB_TUNED_ZERO_STAGE}" "${AB_TUNED_NUM_WORKERS}" "${AB_TUNED_PREFETCH_FACTOR}"
  done
fi

if [[ "${MODE}" == "sweep" || "${MODE}" == "both" ]]; then
  echo "[INFO] Running sweep mode..."
  split_csv_to_array "${SWEEP_ZERO_STAGES}" SWEEP_ZS
  split_csv_to_array "${SWEEP_MICRO_BATCHES}" SWEEP_MBS

  for z in "${SWEEP_ZS[@]}"; do
    require_nonneg_int "SWEEP_ZERO_STAGE item" "${z}"
  done
  for mb in "${SWEEP_MBS[@]}"; do
    require_pos_int "SWEEP_MICRO_BATCH item" "${mb}"
  done

  for z in "${SWEEP_ZS[@]}"; do
    for mb in "${SWEEP_MBS[@]}"; do
      local_group="sweep_z${z}_mb${mb}"
      for repeat in $(seq 1 "${REPEATS}"); do
        run_case "sweep" "${local_group}" "${repeat}" \
          "${SWEEP_LOG_EVERY}" "${SWEEP_VALIDATION_EVERY}" "${SWEEP_CHECKPOINT_EVERY}" \
          "${SWEEP_DETERMINISTIC}" "${SWEEP_CUDNN_BENCHMARK}" "${SWEEP_WALL_CLOCK_BREAKDOWN}" \
          "${mb}" "${z}" "${SWEEP_NUM_WORKERS}" "${SWEEP_PREFETCH_FACTOR}"
      done
    done
  done
fi

PYTHON_CMD=(python)
if [[ -n "${CONDA_ENV}" ]]; then
  PYTHON_CMD=(conda run -n "${CONDA_ENV}" python)
fi

"${PYTHON_CMD[@]}" - "${MANIFEST_PATH}" "${OUTPUT_ROOT}" "${TAIL_TIMING_POINTS}" "${BASELINE_GROUP}" <<'PY'
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
baseline_group = sys.argv[4]

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
    mode = row["mode"]
    group = row["group"]
    repeat = int(row["repeat"])
    status = row["status"]
    train_log = Path(row["train_log"])
    launcher_log = Path(row["launcher_log"])

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
        global_batch = int(global_batch_raw) if global_batch_raw != "" else None
    except ValueError:
        global_batch = None

    if global_batch is not None and global_batch > 0:
        samples_per_sec_tail = (global_batch * 1000.0 / tail_mean_iter)
    else:
        samples_per_sec_tail = math.nan

    per_run = {
        "mode": mode,
        "group": group,
        "repeat": repeat,
        "run_dir": row["run_dir"],
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

    for idx, p in enumerate(points, start=1):
        timing_rows.append(
            {
                "mode": mode,
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

with per_run_csv.open("w", encoding="utf-8", newline="") as f:
    if per_run_rows:
        writer = csv.DictWriter(f, fieldnames=list(per_run_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_run_rows)

with timing_csv.open("w", encoding="utf-8", newline="") as f:
    if timing_rows:
        writer = csv.DictWriter(f, fieldnames=list(timing_rows[0].keys()))
        writer.writeheader()
        writer.writerows(timing_rows)

grouped = defaultdict(list)
for row in per_run_rows:
    grouped[row["group"]].append(row)

baseline_mean = math.nan
if baseline_group in grouped and grouped[baseline_group]:
    baseline_mean = statistics.mean([r["tail_mean_iter_p50_ms"] for r in grouped[baseline_group]])

def unique_or_mixed(rows, key):
    vals = {str(r.get(key, "")) for r in rows}
    vals.discard("")
    if not vals:
        return ""
    if len(vals) == 1:
        return next(iter(vals))
    return "mixed"

group_rows = []
for group, rows in sorted(grouped.items()):
    vals = [r["tail_mean_iter_p50_ms"] for r in rows]
    steps_vals = [r["tail_steps_per_sec"] for r in rows]
    samples_vals = [r["tail_samples_per_sec"] for r in rows if not math.isnan(r["tail_samples_per_sec"])]
    mean_iter = statistics.mean(vals)
    std_iter = statistics.stdev(vals) if len(vals) > 1 else 0.0

    row = {
        "group": group,
        "mode": unique_or_mixed(rows, "mode"),
        "runs": len(rows),
        "zero_stage": unique_or_mixed(rows, "zero_stage"),
        "micro_batch": unique_or_mixed(rows, "micro_batch"),
        "num_workers": unique_or_mixed(rows, "num_workers"),
        "prefetch_factor": unique_or_mixed(rows, "prefetch_factor"),
        "deterministic": unique_or_mixed(rows, "deterministic"),
        "cudnn_benchmark": unique_or_mixed(rows, "cudnn_benchmark"),
        "wall_clock_breakdown": unique_or_mixed(rows, "wall_clock_breakdown"),
        "tail_mean_iter_p50_ms_mean": mean_iter,
        "tail_mean_iter_p50_ms_std": std_iter,
        "tail_mean_iter_p50_ms_min": min(vals),
        "tail_mean_iter_p50_ms_max": max(vals),
        "tail_steps_per_sec_mean": statistics.mean(steps_vals),
        "tail_steps_per_sec_std": statistics.stdev(steps_vals) if len(steps_vals) > 1 else 0.0,
        "tail_samples_per_sec_mean": statistics.mean(samples_vals) if samples_vals else math.nan,
        "tail_samples_per_sec_std": statistics.stdev(samples_vals) if len(samples_vals) > 1 else 0.0,
        "speedup_vs_baseline": (baseline_mean / mean_iter) if not math.isnan(baseline_mean) else math.nan,
    }
    group_rows.append(row)

with group_csv.open("w", encoding="utf-8", newline="") as f:
    if group_rows:
        writer = csv.DictWriter(f, fieldnames=list(group_rows[0].keys()))
        writer.writeheader()
        writer.writerows(group_rows)

ranked_rows = sorted(group_rows, key=lambda x: x["tail_mean_iter_p50_ms_mean"])
with ranked_csv.open("w", encoding="utf-8", newline="") as f:
    if ranked_rows:
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
        "deterministic": best_group["deterministic"],
        "cudnn_benchmark": best_group["cudnn_benchmark"],
        "wall_clock_breakdown": best_group["wall_clock_breakdown"],
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
    "files": {
        "manifest": str(manifest_path),
        "per_run_metrics_csv": str(per_run_csv),
        "timing_points_csv": str(timing_csv),
        "group_summary_csv": str(group_csv),
        "ranked_groups_csv": str(ranked_csv),
        "best_config_json": str(best_json),
        "best_config_env": str(best_env),
        "best_overrides": str(best_overrides),
        "summary_json": str(summary_json),
        "summary_md": str(summary_md),
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

if best_payload is not None:
    with best_json.open("w", encoding="utf-8") as f:
        json.dump(json_safe(best_payload), f, indent=2, ensure_ascii=False, allow_nan=False)

    def env_line(key, val):
        if isinstance(val, float) and math.isnan(val):
            return f"{key}="
        if val is None:
            return f"{key}="
        return f"{key}={val}"

    env_lines = [
        env_line("BEST_GROUP", best_payload.get("group", "")),
        env_line("BEST_MODE", best_payload.get("mode", "")),
        env_line("BEST_ZERO_STAGE", best_payload.get("zero_stage", "")),
        env_line("BEST_MICRO_BATCH", best_payload.get("micro_batch", "")),
        env_line("BEST_NUM_WORKERS", best_payload.get("num_workers", "")),
        env_line("BEST_PREFETCH_FACTOR", best_payload.get("prefetch_factor", "")),
        env_line("BEST_DETERMINISTIC", best_payload.get("deterministic", "")),
        env_line("BEST_CUDNN_BENCHMARK", best_payload.get("cudnn_benchmark", "")),
        env_line("BEST_WALL_CLOCK_BREAKDOWN", best_payload.get("wall_clock_breakdown", "")),
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
        f"++train.deterministic={best_payload.get('deterministic', '')}",
        f"++train.cudnn_benchmark={best_payload.get('cudnn_benchmark', '')}",
        f"++train.data.num_workers={best_payload.get('num_workers', '')}",
        f"++train.data.prefetch_factor={best_payload.get('prefetch_factor', '')}",
        f"deepspeed_config_yaml.wall_clock_breakdown={best_payload.get('wall_clock_breakdown', '')}",
    ]
    best_overrides.write_text("\n".join(override_lines) + "\n", encoding="utf-8")

lines = []
lines.append("# Stage1 BF16 Benchmark Report")
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
    lines.append(f"- tail_mean_iter_p50_ms: `{best_payload['tail_mean_iter_p50_ms_mean']:.4f}`")
    lines.append(f"- tail_steps_per_sec: `{best_payload['tail_steps_per_sec_mean']:.4f}`")
    sps = best_payload.get("tail_samples_per_sec_mean")
    if isinstance(sps, float) and not math.isnan(sps):
        lines.append(f"- tail_samples_per_sec: `{sps:.4f}`")
    lines.append(f"- best_config JSON: `{best_json}`")
    lines.append(f"- best_config ENV: `{best_env}`")
    lines.append(f"- best overrides: `{best_overrides}`")
    lines.append("")

lines.append("## Group Summary")
lines.append("")
if group_rows:
    lines.append("| rank | group | mode | runs | zero_stage | micro_batch | tail_mean_iter_p50_ms_mean | tail_mean_iter_p50_ms_std | tail_steps_per_sec_mean | tail_samples_per_sec_mean | speedup_vs_baseline |")
    lines.append("|---:|---|---|---:|---|---|---:|---:|---:|---:|---:|")
    for idx, r in enumerate(ranked_rows, start=1):
        sps = "nan" if math.isnan(r["tail_samples_per_sec_mean"]) else f"{r['tail_samples_per_sec_mean']:.4f}"
        spd = "nan" if math.isnan(r["speedup_vs_baseline"]) else f"{r['speedup_vs_baseline']:.4f}"
        lines.append(
            f"| {idx} | {r['group']} | {r['mode']} | {r['runs']} | {r['zero_stage']} | {r['micro_batch']} | "
            f"{r['tail_mean_iter_p50_ms_mean']:.4f} | {r['tail_mean_iter_p50_ms_std']:.4f} | "
            f"{r['tail_steps_per_sec_mean']:.4f} | {sps} | {spd} |"
        )
else:
    lines.append("No successful runs found.")

lines.append("")
lines.append("## Output Files")
lines.append("")
for key, path in summary_payload["files"].items():
    lines.append(f"- {key}: `{path}`")

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
PY

echo "[INFO] Benchmark finished."
echo "[INFO] Manifest: ${MANIFEST_PATH}"
echo "[INFO] Report: ${OUTPUT_ROOT}/summary.md"
