#!/usr/bin/env bash
set -euo pipefail

# 4090 revalidation suite:
# - 3 cases: z0+sdpa, z0+flash_attention_2, z1+sdpa
# - run multiple rounds (default 3), with randomized case order per round
# - collect GPU telemetry (from run_stage1_ab_bench.sh) + host telemetry (this script)
# - generate consolidated go/no-go report

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

SUITE_ROOT="${SUITE_ROOT:-outputs/bench_4090_min_matrix_${TIMESTAMP}}"
INCLUDE="${INCLUDE:-localhost:0,1}"
ROUNDS="${ROUNDS:-${REPEATS:-3}}"
MAX_STEPS="${MAX_STEPS:-2000}"
RUN_TIMEOUT_SEC="${RUN_TIMEOUT_SEC:-0}"
STOP_ON_ERROR="${STOP_ON_ERROR:-1}"
STEP_SPAN_TARGET="${STEP_SPAN_TARGET:-1000}"

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

HOST_TELEMETRY_INTERVAL_SEC="${HOST_TELEMETRY_INTERVAL_SEC:-5}"

CASE_NAMES=("z0_sdpa" "z0_flash_attention_2" "z1_sdpa")

ACTIVE_HOST_MONITOR_PID=""
declare -a RUN_DIRS=()

is_pos_int() {
  [[ "$1" =~ ^[0-9]+$ ]] && [[ "$1" -gt 0 ]]
}

require_pos_int() {
  local name="$1"
  local value="$2"
  if ! is_pos_int "${value}"; then
    echo "[ERROR] ${name} must be positive integer, got: ${value}" >&2
    exit 1
  fi
}

case_zero_stage() {
  local case_name="$1"
  case "${case_name}" in
    z0_sdpa|z0_flash_attention_2)
      echo "0"
      ;;
    z1_sdpa)
      echo "1"
      ;;
    *)
      echo "[ERROR] Unknown case_name: ${case_name}" >&2
      exit 1
      ;;
  esac
}

case_speech_attn() {
  local case_name="$1"
  case "${case_name}" in
    z0_sdpa|z1_sdpa)
      echo "sdpa"
      ;;
    z0_flash_attention_2)
      echo "flash_attention_2"
      ;;
    *)
      echo "[ERROR] Unknown case_name: ${case_name}" >&2
      exit 1
      ;;
  esac
}

case_text_attn() {
  case_speech_attn "$1"
}

shuffle_cases() {
  if command -v shuf >/dev/null 2>&1; then
    printf '%s\n' "${CASE_NAMES[@]}" | shuf
    return 0
  fi

  python - <<'PY'
import random
cases = ["z0_sdpa", "z0_flash_attention_2", "z1_sdpa"]
random.shuffle(cases)
for c in cases:
    print(c)
PY
}

start_host_telemetry() {
  local out_csv="$1"
  local interval_sec="$2"

  {
    echo "timestamp,epoch_sec,mem_total_kib,mem_available_kib,swap_total_kib,swap_free_kib,load1,load5,load15,train_proc_count,train_rss_kib,train_vsz_kib"
    while true; do
      ts="$(date -Is)"
      epoch="$(date +%s)"

      mt="$(awk '/^MemTotal:/ {print $2}' /proc/meminfo | head -n1)"
      ma="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo | head -n1)"
      st="$(awk '/^SwapTotal:/ {print $2}' /proc/meminfo | head -n1)"
      sf="$(awk '/^SwapFree:/ {print $2}' /proc/meminfo | head -n1)"
      read -r l1 l5 l15 _ < /proc/loadavg
      read -r pc pr pv <<< "$(ps -eo pid=,rss=,vsz=,args= | awk '
        BEGIN{c=0;r=0;v=0}
        {
          pid=$1; rss=$2; vsz=$3;
          $1=$2=$3="";
          sub(/^ +/,"",$0);
          if($0 ~ /(deepspeed|train.py)/){ c++; r+=rss; v+=vsz }
        }
        END{ printf "%d %d %d", c, r, v }
      ')"

      echo "${ts},${epoch},${mt},${ma},${st},${sf},${l1},${l5},${l15},${pc},${pr},${pv}"
      sleep "${interval_sec}"
    done
  } > "${out_csv}" &
  ACTIVE_HOST_MONITOR_PID="$!"
}

stop_host_telemetry() {
  if [[ -n "${ACTIVE_HOST_MONITOR_PID}" ]] && kill -0 "${ACTIVE_HOST_MONITOR_PID}" 2>/dev/null; then
    kill "${ACTIVE_HOST_MONITOR_PID}" 2>/dev/null || true
    wait "${ACTIVE_HOST_MONITOR_PID}" 2>/dev/null || true
  fi
  ACTIVE_HOST_MONITOR_PID=""
}

cleanup_on_exit() {
  stop_host_telemetry
}

run_case_round() {
  local case_name="$1"
  local round="$2"
  local order_idx="$3"

  local zero_stage speech_attn text_attn
  zero_stage="$(case_zero_stage "${case_name}")"
  speech_attn="$(case_speech_attn "${case_name}")"
  text_attn="$(case_text_attn "${case_name}")"

  local out_dir="${SUITE_ROOT}/${case_name}_r${round}"
  mkdir -p "${out_dir}"
  RUN_DIRS+=("${out_dir}")

  local start_ts end_ts start_epoch end_epoch duration status
  start_ts="$(date -Is)"
  start_epoch="$(date +%s)"
  status="success"

  echo "[INFO] round=${round} order=${order_idx} case=${case_name} z=${zero_stage} speech_attn=${speech_attn} text_attn=${text_attn}"
  echo "[INFO] out_dir=${out_dir}"

  start_host_telemetry "${out_dir}/host_telemetry.csv" "${HOST_TELEMETRY_INTERVAL_SEC}"

  if ! MODE=sweep \
    INCLUDE="${INCLUDE}" \
    REPEATS=1 \
    STOP_ON_ERROR="${STOP_ON_ERROR}" \
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
    "${ROOT_DIR}/scripts/run_stage1_ab_bench.sh"; then
    status="failed"
  fi

  stop_host_telemetry

  end_ts="$(date -Is)"
  end_epoch="$(date +%s)"
  duration=$((end_epoch - start_epoch))

  {
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "${round}" "${order_idx}" "${case_name}" "${zero_stage}" "${speech_attn}" "${text_attn}" \
      "${status}" "${start_ts}" "${end_ts}" "${duration}" "${out_dir}" "${out_dir}/host_telemetry.csv"
  } >> "${SUITE_ROOT}/run_order.tsv"

  if [[ "${status}" != "success" ]]; then
    echo "[ERROR] case failed: ${case_name} (round=${round})" >&2
    return 1
  fi
}

check_flash2_dtype_warning_all() {
  local out_txt="${SUITE_ROOT}/flash2_dtype_check.txt"
  : > "${out_txt}"

  local found_any="false"
  local d=""
  for d in "${SUITE_ROOT}"/z0_flash_attention_2_r*; do
    if [[ ! -d "${d}" ]]; then
      continue
    fi
    if rg -n "Flash Attention 2 only supports|current dype|current dtype.*float32" "${d}" -g "launcher.log" >/dev/null 2>&1; then
      found_any="true"
      {
        echo "=== ${d} ==="
        rg -n "Flash Attention 2 only supports|current dype|current dtype.*float32" "${d}" -g "launcher.log" || true
        echo
      } >> "${out_txt}"
    fi
  done

  if [[ "${found_any}" == "true" ]]; then
    echo "[WARN] flash2 dtype warning detected. See ${out_txt}" >&2
  else
    echo "[INFO] no flash2 dtype warning detected." | tee -a "${out_txt}" >/dev/null
  fi
}

run_quick_leak_diagnose() {
  if [[ "${#RUN_DIRS[@]}" -eq 0 ]]; then
    return 0
  fi
  python "${ROOT_DIR}/scripts/quick_leak_diagnose.py" "${RUN_DIRS[@]}" > "${SUITE_ROOT}/quick_leak_diagnose.csv"
}

main() {
  require_pos_int "ROUNDS" "${ROUNDS}"
  require_pos_int "MAX_STEPS" "${MAX_STEPS}"
  require_pos_int "MICRO_BATCH" "${MICRO_BATCH}"
  require_pos_int "NUM_WORKERS" "${NUM_WORKERS}"
  require_pos_int "PREFETCH_FACTOR" "${PREFETCH_FACTOR}"
  require_pos_int "HOST_TELEMETRY_INTERVAL_SEC" "${HOST_TELEMETRY_INTERVAL_SEC}"
  require_pos_int "STEP_SPAN_TARGET" "${STEP_SPAN_TARGET}"

  mkdir -p "${SUITE_ROOT}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "round" "order" "case" "zero_stage" "speech_attn" "text_attn" \
    "status" "start_ts" "end_ts" "duration_sec" "output_root" "host_telemetry_csv" \
    > "${SUITE_ROOT}/run_order.tsv"

  {
    echo "SUITE_ROOT=${SUITE_ROOT}"
    echo "INCLUDE=${INCLUDE}"
    echo "ROUNDS=${ROUNDS}"
    echo "MAX_STEPS=${MAX_STEPS}"
    echo "MICRO_BATCH=${MICRO_BATCH}"
    echo "NUM_WORKERS=${NUM_WORKERS}"
    echo "PREFETCH_FACTOR=${PREFETCH_FACTOR}"
    echo "PRECISION_MODE=${PRECISION_MODE}"
    echo "MODEL_LOAD_DTYPE=${MODEL_LOAD_DTYPE}"
    echo "ENABLE_TF32=${ENABLE_TF32}"
    echo "MATMUL_PRECISION=${MATMUL_PRECISION}"
    echo "ENABLE_TORCH_COMPILE=${ENABLE_TORCH_COMPILE}"
    echo "TORCH_COMPILE_MODE=${TORCH_COMPILE_MODE}"
    echo "TORCH_COMPILE_DYNAMIC=${TORCH_COMPILE_DYNAMIC}"
    echo "ENABLE_LENGTH_FIXED_SLICE=${ENABLE_LENGTH_FIXED_SLICE}"
    echo "FIXED_SLICE_SECONDS=${FIXED_SLICE_SECONDS}"
    echo "ENABLE_GPU_TELEMETRY=${ENABLE_GPU_TELEMETRY}"
    echo "GPU_TELEMETRY_INTERVAL_SEC=${GPU_TELEMETRY_INTERVAL_SEC}"
    echo "HOST_TELEMETRY_INTERVAL_SEC=${HOST_TELEMETRY_INTERVAL_SEC}"
    echo "STEP_SPAN_TARGET=${STEP_SPAN_TARGET}"
  } > "${SUITE_ROOT}/suite_config.env"

  echo "[INFO] Suite root: ${SUITE_ROOT}"
  echo "[INFO] rounds=${ROUNDS}, max_steps=${MAX_STEPS}, mb=${MICRO_BATCH}, nw=${NUM_WORKERS}, pf=${PREFETCH_FACTOR}"

  local round=0
  for round in $(seq 1 "${ROUNDS}"); do
    mapfile -t order < <(shuffle_cases)
    echo "[INFO] Round ${round} order: ${order[*]}"
    local idx=0
    local case_name=""
    for case_name in "${order[@]}"; do
      idx=$((idx + 1))
      run_case_round "${case_name}" "${round}" "${idx}"
    done
  done

  check_flash2_dtype_warning_all
  run_quick_leak_diagnose

  python "${ROOT_DIR}/scripts/analyze_4090_min_matrix.py" \
    --suite-root "${SUITE_ROOT}" \
    --step-span-target "${STEP_SPAN_TARGET}" \
    --output-json "${SUITE_ROOT}/suite_report.json" \
    --output-csv "${SUITE_ROOT}/suite_case_summary.csv" \
    --output-md "${SUITE_ROOT}/suite_report.md"

  echo "[INFO] Done."
  echo "[INFO] run_order: ${SUITE_ROOT}/run_order.tsv"
  echo "[INFO] quick leak: ${SUITE_ROOT}/quick_leak_diagnose.csv"
  echo "[INFO] final report: ${SUITE_ROOT}/suite_report.md"
}

trap cleanup_on_exit EXIT INT TERM
main "$@"
