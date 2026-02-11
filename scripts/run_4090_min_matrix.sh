#!/usr/bin/env bash
set -euo pipefail

# One-click minimal 4090 matrix for throughput/stability:
#   A) z0 + sdpa
#   B) z0 + flash_attention_2
#   C) z1 + sdpa
#
# Each case uses the same batch/data params by default so comparisons are fair.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

SUITE_ROOT="${SUITE_ROOT:-outputs/bench_4090_min_matrix_${TIMESTAMP}}"
INCLUDE="${INCLUDE:-localhost:0,1}"
REPEATS="${REPEATS:-1}"
MAX_STEPS="${MAX_STEPS:-1000}"
RUN_TIMEOUT_SEC="${RUN_TIMEOUT_SEC:-0}"

MICRO_BATCH="${MICRO_BATCH:-128}"
NUM_WORKERS="${NUM_WORKERS:-6}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"

DATASET_ROOT="${DATASET_ROOT:-data/LibriSpeech/LibriSpeech_16k_trim}"
DATASET_MANIFEST_PATH="${DATASET_MANIFEST_PATH:-data/LibriSpeech/LibriSpeech_16k_trim/manifest_16k_trim.tsv}"

PRECISION_MODE="${PRECISION_MODE:-auto}"
MODEL_LOAD_DTYPE="${MODEL_LOAD_DTYPE:-auto}"
ENABLE_TF32="${ENABLE_TF32:-true}"
MATMUL_PRECISION="${MATMUL_PRECISION:-high}"
ENABLE_TORCH_COMPILE="${ENABLE_TORCH_COMPILE:-false}"
TORCH_COMPILE_MODE="${TORCH_COMPILE_MODE:-max-autotune}"
TORCH_COMPILE_DYNAMIC="${TORCH_COMPILE_DYNAMIC:-true}"
ENABLE_LENGTH_FIXED_SLICE="${ENABLE_LENGTH_FIXED_SLICE:-false}"
FIXED_SLICE_SECONDS="${FIXED_SLICE_SECONDS:-}"

ENABLE_GPU_TELEMETRY="${ENABLE_GPU_TELEMETRY:-true}"
GPU_TELEMETRY_INTERVAL_SEC="${GPU_TELEMETRY_INTERVAL_SEC:-2}"
STALL_ALERT_RATIO="${STALL_ALERT_RATIO:-2.0}"

mkdir -p "${SUITE_ROOT}"

declare -a CASE_DIRS=()

run_case() {
  local case_name="$1"
  local zero_stage="$2"
  local speech_attn="$3"
  local text_attn="$4"
  local out_dir="${SUITE_ROOT}/${case_name}"

  mkdir -p "${out_dir}"
  CASE_DIRS+=("${out_dir}")

  echo "[INFO] ===== ${case_name} ====="
  echo "[INFO] out_dir=${out_dir}"

  MODE=sweep \
  INCLUDE="${INCLUDE}" \
  REPEATS="${REPEATS}" \
  STOP_ON_ERROR=1 \
  MAX_STEPS="${MAX_STEPS}" \
  SWEEP_ZERO_STAGES="${zero_stage}" \
  SWEEP_MICRO_BATCHES="${MICRO_BATCH}" \
  SWEEP_NUM_WORKERS_LIST="${NUM_WORKERS}" \
  SWEEP_PREFETCH_LIST="${PREFETCH_FACTOR}" \
  SWEEP_LOG_EVERY=50 \
  TAIL_TIMING_POINTS=10 \
  HEARTBEAT_EVERY_SEC=30 \
  RUN_TIMEOUT_SEC="${RUN_TIMEOUT_SEC}" \
  FAILURE_DUMP_TAIL=true \
  FAIL_TAIL_LINES=120 \
  RESUME_RUNS=false \
  SWEEP_VALIDATION_EVERY=1000000 \
  SWEEP_CHECKPOINT_EVERY=1000000 \
  DATASET_ROOT="${DATASET_ROOT}" \
  DATASET_MANIFEST_PATH="${DATASET_MANIFEST_PATH}" \
  DATASET_USE_TRIM=false \
  DATASET_OFFLINE_TRIMMED=true \
  ENABLE_CUDA_SYNC_TIMING=false \
  TIMING_RANK_SCOPE=all \
  PRECISION_MODE="${PRECISION_MODE}" \
  MODEL_LOAD_DTYPE="${MODEL_LOAD_DTYPE}" \
  ATTN_IMPL=auto \
  SPEECH_ATTN_IMPL="${speech_attn}" \
  TEXT_ATTN_IMPL="${text_attn}" \
  ENABLE_TF32="${ENABLE_TF32}" \
  MATMUL_PRECISION="${MATMUL_PRECISION}" \
  ENABLE_TORCH_COMPILE="${ENABLE_TORCH_COMPILE}" \
  TORCH_COMPILE_MODE="${TORCH_COMPILE_MODE}" \
  TORCH_COMPILE_DYNAMIC="${TORCH_COMPILE_DYNAMIC}" \
  ENABLE_LENGTH_FIXED_SLICE="${ENABLE_LENGTH_FIXED_SLICE}" \
  FIXED_SLICE_SECONDS="${FIXED_SLICE_SECONDS}" \
  ENABLE_GPU_TELEMETRY="${ENABLE_GPU_TELEMETRY}" \
  GPU_TELEMETRY_INTERVAL_SEC="${GPU_TELEMETRY_INTERVAL_SEC}" \
  STALL_ALERT_RATIO="${STALL_ALERT_RATIO}" \
  OUTPUT_ROOT="${out_dir}" \
  DRIVER_LOG_PATH="${out_dir}/driver.log" \
  "${ROOT_DIR}/scripts/run_stage1_ab_bench.sh"
}

check_flash2_dtype_warning() {
  local target_dir="$1"
  if rg -n "Flash Attention 2 only supports|current dype|current dtype.*float32" "${target_dir}" -g "launcher.log" >/dev/null 2>&1; then
    echo "[WARN] flash2 dtype warning detected in ${target_dir}"
    rg -n "Flash Attention 2 only supports|current dype|current dtype.*float32" "${target_dir}" -g "launcher.log" || true
  else
    echo "[INFO] no flash2 dtype warning in ${target_dir}"
  fi
}

echo "[INFO] Suite root: ${SUITE_ROOT}"
echo "[INFO] include=${INCLUDE}, repeats=${REPEATS}, max_steps=${MAX_STEPS}, mb=${MICRO_BATCH}, nw=${NUM_WORKERS}, pf=${PREFETCH_FACTOR}"

run_case "z0_sdpa" 0 "sdpa" "sdpa"
run_case "z0_flash_attention_2" 0 "flash_attention_2" "flash_attention_2"
run_case "z1_sdpa" 1 "sdpa" "sdpa"

check_flash2_dtype_warning "${SUITE_ROOT}/z0_flash_attention_2"

echo "[INFO] Running quick leak diagnose..."
python "${ROOT_DIR}/scripts/quick_leak_diagnose.py" "${CASE_DIRS[@]}" > "${SUITE_ROOT}/quick_leak_diagnose.csv"

echo "[INFO] Quick leak report: ${SUITE_ROOT}/quick_leak_diagnose.csv"
echo "[INFO] Done. Case outputs:"
for d in "${CASE_DIRS[@]}"; do
  echo "  - ${d}"
done
