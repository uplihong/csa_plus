#!/usr/bin/env bash
set -euo pipefail

# Minimal Phase-B stability matrix for 2x4090:
# - sentinel A: z0_mb128_nw6_pf4, repeats=3
# - sentinel B: z1_mb160_nw6_pf2, repeats=3
# It appends both runs into the same OUTPUT_ROOT and validates acceptance.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/bench_4090_phaseB_diag}"
DRIVER_LOG_PATH="${DRIVER_LOG_PATH:-${OUTPUT_ROOT}/driver.log}"
INCLUDE="${INCLUDE:-localhost:0,1}"
REPEATS="${REPEATS:-3}"
MAX_STEPS="${MAX_STEPS:-1000}"

run_sentinel() {
  local zero_stage="$1"
  local micro_batch="$2"
  local num_workers="$3"
  local prefetch="$4"

  MODE=sweep \
  INCLUDE="${INCLUDE}" \
  REPEATS="${REPEATS}" \
  STOP_ON_ERROR=0 \
  MAX_STEPS="${MAX_STEPS}" \
  SWEEP_ZERO_STAGES="${zero_stage}" \
  SWEEP_MICRO_BATCHES="${micro_batch}" \
  SWEEP_NUM_WORKERS_LIST="${num_workers}" \
  SWEEP_PREFETCH_LIST="${prefetch}" \
  SWEEP_LOG_EVERY=50 \
  TAIL_TIMING_POINTS=10 \
  HEARTBEAT_EVERY_SEC=30 \
  RUN_TIMEOUT_SEC=2400 \
  FAILURE_DUMP_TAIL=true \
  FAIL_TAIL_LINES=120 \
  RESUME_RUNS=true \
  SWEEP_VALIDATION_EVERY=1000000 \
  SWEEP_CHECKPOINT_EVERY=1000000 \
  DATASET_ROOT="${DATASET_ROOT:-data/LibriSpeech/LibriSpeech_16k_trim}" \
  DATASET_MANIFEST_PATH="${DATASET_MANIFEST_PATH:-data/LibriSpeech/LibriSpeech_16k_trim/manifest_16k_trim.tsv}" \
  DATASET_USE_TRIM=false \
  DATASET_OFFLINE_TRIMMED=true \
  ENABLE_CUDA_SYNC_TIMING=false \
  TIMING_RANK_SCOPE=all \
  PRECISION_MODE=auto \
  ATTN_IMPL=auto \
  ENABLE_TORCH_COMPILE=false \
  TORCH_COMPILE_MODE=max-autotune \
  TORCH_COMPILE_DYNAMIC=true \
  ENABLE_LENGTH_FIXED_SLICE=false \
  ENABLE_GPU_TELEMETRY=true \
  GPU_TELEMETRY_INTERVAL_SEC=2 \
  STALL_ALERT_RATIO=2.0 \
  OUTPUT_ROOT="${OUTPUT_ROOT}" \
  DRIVER_LOG_PATH="${DRIVER_LOG_PATH}" \
  "${ROOT_DIR}/scripts/run_stage1_ab_bench.sh"
}

mkdir -p "${OUTPUT_ROOT}"

echo "[INFO] Running Phase-B sentinel A: z0_mb128_nw6_pf4"
run_sentinel 0 128 6 4

echo "[INFO] Running Phase-B sentinel B: z1_mb160_nw6_pf2"
run_sentinel 1 160 6 2

echo "[INFO] Running stability acceptance check"
"${ROOT_DIR}/scripts/check_phaseb_stability.py" \
  --output-root "${OUTPUT_ROOT}" \
  --groups "sweep_z0_mb128_nw6_pf4,sweep_z1_mb160_nw6_pf2" \
  --min-repeats 3 \
  --max-iter-ratio-median 1.5 \
  --max-iter-ratio-max 2.5 \
  --max-iter-p50-cv 0.05

echo "[INFO] Done. See ${OUTPUT_ROOT}/summary.md and ${OUTPUT_ROOT}/run_manifest.tsv"
