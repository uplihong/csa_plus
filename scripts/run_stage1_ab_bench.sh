#!/usr/bin/env bash
set -euo pipefail

# Unified benchmark runner for Stage1 distributed runs on multi-GPU.
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
STOP_ON_ERROR="${STOP_ON_ERROR:-0}"
CONDA_ENV="${CONDA_ENV:-}"
RUN_TIMEOUT_SEC="${RUN_TIMEOUT_SEC:-0}"               # 0 means disabled
HEARTBEAT_EVERY_SEC="${HEARTBEAT_EVERY_SEC:-30}"      # 0 means disabled
FAILURE_DUMP_TAIL="${FAILURE_DUMP_TAIL:-true}"        # print log tails on failure
FAIL_TAIL_LINES="${FAIL_TAIL_LINES:-80}"
RESUME_RUNS="${RESUME_RUNS:-true}"                    # reuse existing run_manifest.tsv

DATASET_ROOT="${DATASET_ROOT:-data/LibriSpeech/LibriSpeech}"
DATASET_MANIFEST_PATH="${DATASET_MANIFEST_PATH:-}"
DATASET_USE_TRIM="${DATASET_USE_TRIM:-false}"
DATASET_OFFLINE_TRIMMED="${DATASET_OFFLINE_TRIMMED:-true}"
PRETRAINED_CKPT="${PRETRAINED_CKPT:-data/weights/csa/ckpt_epoch_8.pth}"
SPEECH_MODEL_PATH="${SPEECH_MODEL_PATH:-data/weights/wav2vec2-base}"
TEXT_MODEL_PATH="${TEXT_MODEL_PATH:-data/weights/bert-base-uncased}"
ENABLE_CUDA_SYNC_TIMING="${ENABLE_CUDA_SYNC_TIMING:-false}"
TIMING_RANK_SCOPE="${TIMING_RANK_SCOPE:-rank0}" # rank0 | all
PRECISION_MODE="${PRECISION_MODE:-auto}"        # auto | bf16 | fp16
ATTN_IMPL="${ATTN_IMPL:-auto}"                  # auto | eager | sdpa | flash_attention_2
ENABLE_TORCH_COMPILE="${ENABLE_TORCH_COMPILE:-false}"
TORCH_COMPILE_MODE="${TORCH_COMPILE_MODE:-max-autotune}"
TORCH_COMPILE_DYNAMIC="${TORCH_COMPILE_DYNAMIC:-true}"
ENABLE_LENGTH_FIXED_SLICE="${ENABLE_LENGTH_FIXED_SLICE:-false}"
FIXED_SLICE_SECONDS="${FIXED_SLICE_SECONDS:-}"

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
SWEEP_MICRO_BATCHES="${SWEEP_MICRO_BATCHES:-128,160,192}"
SWEEP_LOG_EVERY="${SWEEP_LOG_EVERY:-100}"
SWEEP_VALIDATION_EVERY="${SWEEP_VALIDATION_EVERY:-1000000}"
SWEEP_CHECKPOINT_EVERY="${SWEEP_CHECKPOINT_EVERY:-1000000}"
SWEEP_DETERMINISTIC="${SWEEP_DETERMINISTIC:-false}"
SWEEP_CUDNN_BENCHMARK="${SWEEP_CUDNN_BENCHMARK:-false}"
SWEEP_WALL_CLOCK_BREAKDOWN="${SWEEP_WALL_CLOCK_BREAKDOWN:-false}"
SWEEP_NUM_WORKERS="${SWEEP_NUM_WORKERS:-2}"
SWEEP_PREFETCH_FACTOR="${SWEEP_PREFETCH_FACTOR:-2}"
SWEEP_NUM_WORKERS_LIST="${SWEEP_NUM_WORKERS_LIST:-}"
SWEEP_PREFETCH_LIST="${SWEEP_PREFETCH_LIST:-}"

mkdir -p "${OUTPUT_ROOT}"
MANIFEST_PATH="${OUTPUT_ROOT}/run_manifest.tsv"

MANIFEST_HEADER=$'mode\tgroup\trepeat\tstatus\texit_code\tduration_sec\tlast_step\tlast_iter_ms_p50\tport\trun_dir\tlauncher_log\ttrain_log\tzero_stage\tmicro_batch\tworld_size\tglobal_batch\tdeterministic\tcudnn_benchmark\twall_clock_breakdown\tlog_every_steps\tvalidation_every_steps\tcheckpoint_every_steps\tnum_workers\tprefetch_factor\tdataset_manifest_path\tdataset_use_trim\tdataset_offline_trimmed\tenable_cuda_sync_timing\ttiming_rank_scope\tprecision_mode_req\tprecision_mode_effective\tattn_impl_effective\ttorch_compile_enabled\tgpu_name\tgpu_cc'
declare -A COMPLETED_RUNS

PYTHON_CMD=(python)
if [[ -n "${CONDA_ENV}" ]]; then
  PYTHON_CMD=(conda run -n "${CONDA_ENV}" python)
fi

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

require_timing_rank_scope() {
  local value="$1"
  if [[ "${value}" != "rank0" && "${value}" != "all" ]]; then
    echo "TIMING_RANK_SCOPE must be one of: rank0, all. got: ${value}" >&2
    exit 1
  fi
}

require_precision_mode() {
  local value="$1"
  if [[ "${value}" != "auto" && "${value}" != "bf16" && "${value}" != "fp16" ]]; then
    echo "PRECISION_MODE must be one of: auto, bf16, fp16. got: ${value}" >&2
    exit 1
  fi
}

require_attn_impl() {
  local value="$1"
  if [[ "${value}" != "auto" && "${value}" != "eager" && "${value}" != "sdpa" && "${value}" != "flash_attention_2" ]]; then
    echo "ATTN_IMPL must be one of: auto, eager, sdpa, flash_attention_2. got: ${value}" >&2
    exit 1
  fi
}

require_nonempty() {
  local name="$1"
  local value="$2"
  if [[ -z "${value}" ]]; then
    echo "${name} must be non-empty" >&2
    exit 1
  fi
}

require_pos_float() {
  local name="$1"
  local value="$2"
  if [[ ! "${value}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "${name} must be a positive float, got: ${value}" >&2
    exit 1
  fi
  awk -v v="${value}" 'BEGIN{ exit !(v > 0) }' || {
    echo "${name} must be > 0, got: ${value}" >&2
    exit 1
  }
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
require_bool "DATASET_USE_TRIM" "${DATASET_USE_TRIM}"
require_bool "DATASET_OFFLINE_TRIMMED" "${DATASET_OFFLINE_TRIMMED}"
require_bool "ENABLE_CUDA_SYNC_TIMING" "${ENABLE_CUDA_SYNC_TIMING}"
require_bool "FAILURE_DUMP_TAIL" "${FAILURE_DUMP_TAIL}"
require_bool "RESUME_RUNS" "${RESUME_RUNS}"
require_bool "ENABLE_TORCH_COMPILE" "${ENABLE_TORCH_COMPILE}"
require_bool "TORCH_COMPILE_DYNAMIC" "${TORCH_COMPILE_DYNAMIC}"
require_bool "ENABLE_LENGTH_FIXED_SLICE" "${ENABLE_LENGTH_FIXED_SLICE}"
require_timing_rank_scope "${TIMING_RANK_SCOPE}"
require_precision_mode "${PRECISION_MODE}"
require_attn_impl "${ATTN_IMPL}"
require_nonneg_int "RUN_TIMEOUT_SEC" "${RUN_TIMEOUT_SEC}"
require_nonneg_int "HEARTBEAT_EVERY_SEC" "${HEARTBEAT_EVERY_SEC}"
require_pos_int "FAIL_TAIL_LINES" "${FAIL_TAIL_LINES}"
require_nonempty "TORCH_COMPILE_MODE" "${TORCH_COMPILE_MODE}"

if [[ "${ENABLE_LENGTH_FIXED_SLICE}" == "true" ]]; then
  require_nonempty "FIXED_SLICE_SECONDS" "${FIXED_SLICE_SECONDS}"
  require_pos_float "FIXED_SLICE_SECONDS" "${FIXED_SLICE_SECONDS}"
fi

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

DETECTED_GPU_NAME="unknown"
DETECTED_GPU_CC="unknown"
DETECTED_MIN_CC_MAJOR="0"
FLASH_ATTN2_AVAILABLE="false"
EFFECTIVE_PRECISION_MODE=""
EFFECTIVE_ATTN_IMPL=""

detect_hardware_profile() {
  local include="$1"
  local raw=""
  if ! raw="$("${PYTHON_CMD[@]}" - "${include}" <<'PY'
import importlib.util
import sys

try:
    import torch
except Exception:
    print("ok=false")
    print("gpu_name=unknown")
    print("gpu_cc=unknown")
    print("min_cc_major=0")
    print("flash_attn2_available=false")
    raise SystemExit(0)

include_arg = sys.argv[1] if len(sys.argv) > 1 else ""
requested_ids = []
if ":" in include_arg:
    after_colon = include_arg.split(":", 1)[1]
    for token in after_colon.split(","):
        token = token.strip()
        if token.isdigit():
            requested_ids.append(int(token))

if not torch.cuda.is_available():
    print("ok=false")
    print("gpu_name=unknown")
    print("gpu_cc=unknown")
    print("min_cc_major=0")
    print("flash_attn2_available=false")
    raise SystemExit(0)

device_count = torch.cuda.device_count()
if device_count <= 0:
    print("ok=false")
    print("gpu_name=unknown")
    print("gpu_cc=unknown")
    print("min_cc_major=0")
    print("flash_attn2_available=false")
    raise SystemExit(0)

if requested_ids:
    valid_ids = [i for i in requested_ids if 0 <= i < device_count]
else:
    valid_ids = list(range(device_count))
if not valid_ids:
    valid_ids = [0]

names = []
cc_values = []
for idx in valid_ids:
    major, minor = torch.cuda.get_device_capability(idx)
    names.append(torch.cuda.get_device_name(idx).replace("\t", " "))
    cc_values.append((idx, major, minor))

min_cc_major = min(item[1] for item in cc_values)
gpu_name = " | ".join(sorted(set(names)))
gpu_cc = ",".join(f"{idx}:{major}.{minor}" for idx, major, minor in cc_values)
flash_available = importlib.util.find_spec("flash_attn") is not None

print("ok=true")
print(f"gpu_name={gpu_name}")
print(f"gpu_cc={gpu_cc}")
print(f"min_cc_major={min_cc_major}")
print(f"flash_attn2_available={'true' if flash_available else 'false'}")
PY
)"; then
    echo "[WARN] Failed to probe GPU capability from Python. Fallback to conservative defaults." >&2
    return 0
  fi

  while IFS='=' read -r key value; do
    case "${key}" in
      gpu_name)
        DETECTED_GPU_NAME="${value}"
        ;;
      gpu_cc)
        DETECTED_GPU_CC="${value}"
        ;;
      min_cc_major)
        DETECTED_MIN_CC_MAJOR="${value}"
        ;;
      flash_attn2_available)
        FLASH_ATTN2_AVAILABLE="${value}"
        ;;
      *)
        :
        ;;
    esac
  done <<< "${raw}"
}

resolve_precision_mode() {
  local requested="$1"
  local min_cc_major="$2"
  case "${requested}" in
    bf16)
      echo "bf16"
      ;;
    fp16)
      echo "fp16"
      ;;
    auto)
      if [[ "${min_cc_major}" =~ ^[0-9]+$ ]] && [[ "${min_cc_major}" -ge 8 ]]; then
        echo "bf16"
      else
        echo "fp16"
      fi
      ;;
    *)
      echo "bf16"
      ;;
  esac
}

resolve_attn_impl() {
  local requested="$1"
  local precision_mode="$2"
  local flash_available="$3"
  local min_cc_major="$4"
  if [[ "${requested}" == "auto" ]]; then
    if [[ "${precision_mode}" == "fp16" ]]; then
      echo "eager"
    else
      echo "sdpa"
    fi
    return 0
  fi

  if [[ "${requested}" == "flash_attention_2" ]] && { [[ "${flash_available}" != "true" ]] || [[ ! "${min_cc_major}" =~ ^[0-9]+$ ]] || [[ "${min_cc_major}" -lt 8 ]]; }; then
    echo "[WARN] ATTN_IMPL=flash_attention_2 requested, but flash-attn is unavailable or GPU capability is < 8.0. Fallback to sdpa." >&2
    echo "sdpa"
    return 0
  fi
  echo "${requested}"
}

detect_hardware_profile "${INCLUDE}"
EFFECTIVE_PRECISION_MODE="$(resolve_precision_mode "${PRECISION_MODE}" "${DETECTED_MIN_CC_MAJOR}")"
EFFECTIVE_ATTN_IMPL="$(resolve_attn_impl "${ATTN_IMPL}" "${EFFECTIVE_PRECISION_MODE}" "${FLASH_ATTN2_AVAILABLE}" "${DETECTED_MIN_CC_MAJOR}")"

PORT_COUNTER=0

init_manifest() {
  if [[ "${RESUME_RUNS}" == "true" && -f "${MANIFEST_PATH}" ]]; then
    local existing_header
    existing_header="$(head -n 1 "${MANIFEST_PATH}" || true)"
    if [[ "${existing_header}" != "${MANIFEST_HEADER}" ]]; then
      local backup_path="${MANIFEST_PATH}.bak_$(date +%Y%m%d_%H%M%S)"
      cp "${MANIFEST_PATH}" "${backup_path}" || true
      echo "[WARN] Existing manifest header mismatch. Backed up old manifest to ${backup_path}, then resetting ${MANIFEST_PATH}" >&2
      echo "${MANIFEST_HEADER}" > "${MANIFEST_PATH}"
      return 0
    fi

    local loaded=0
    while IFS=$'\t' read -r mode group repeat status _rest; do
      if [[ "${mode}" == "mode" ]]; then
        continue
      fi
      if [[ "${status}" == "success" ]]; then
        COMPLETED_RUNS["${mode}::${group}::${repeat}"]=1
        loaded=$((loaded + 1))
      fi
    done < "${MANIFEST_PATH}"
    echo "[INFO] Resuming with existing manifest: ${MANIFEST_PATH} (successful runs cached=${loaded})"
  else
    echo "${MANIFEST_HEADER}" > "${MANIFEST_PATH}"
  fi
}

extract_last_step() {
  local log_path="$1"
  if [[ ! -f "${log_path}" ]]; then
    echo ""
    return 0
  fi
  grep -Eo 'Step [0-9]+:' "${log_path}" | tail -n 1 | grep -Eo '[0-9]+' || true
}

extract_last_iter_ms() {
  local log_path="$1"
  if [[ ! -f "${log_path}" ]]; then
    echo ""
    return 0
  fi
  grep 'Timing(window=' "${log_path}" | tail -n 1 | sed -n 's/.*iter_ms:p50=\([0-9.]*\).*/\1/p' || true
}

start_heartbeat() {
  local child_pid="$1"
  local group="$2"
  local repeat="$3"
  local train_log="$4"
  local interval="$5"

  if [[ "${interval}" -le 0 ]]; then
    echo ""
    return 0
  fi

  (
    while kill -0 "${child_pid}" 2>/dev/null; do
      sleep "${interval}"
      if [[ ! -f "${train_log}" ]]; then
        echo "[INFO] heartbeat ${group} r${repeat}: waiting for ${train_log}" >&2
        continue
      fi
      local step=""
      local iter_ms=""
      step="$(extract_last_step "${train_log}")"
      iter_ms="$(extract_last_iter_ms "${train_log}")"
      echo "[INFO] heartbeat ${group} r${repeat}: last_step=${step:-N/A}, iter_ms_p50=${iter_ms:-N/A}" >&2
    done
  ) &
  echo "$!"
}

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
  local run_key="${mode_name}::${group}::${repeat}"
  if [[ "${RESUME_RUNS}" == "true" && -n "${COMPLETED_RUNS["${run_key}"]+x}" ]]; then
    echo "[INFO] SKIP ${group} r${repeat}: already successful in manifest"
    return 0
  fi

  mkdir -p "${run_dir}"

  local port=$((MASTER_PORT_BASE + PORT_COUNTER))
  PORT_COUNTER=$((PORT_COUNTER + 1))

  local global_batch=""
  if [[ "${WORLD_SIZE}" -gt 0 ]]; then
    global_batch=$((micro_batch * WORLD_SIZE))
  fi

  local ds_bf16_enabled="false"
  local ds_fp16_enabled="false"
  if [[ "${EFFECTIVE_PRECISION_MODE}" == "bf16" ]]; then
    ds_bf16_enabled="true"
  else
    ds_fp16_enabled="true"
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
    "++train.enable_cuda_sync_timing=${ENABLE_CUDA_SYNC_TIMING}"
    "++train.timing_rank_scope=${TIMING_RANK_SCOPE}"
    "++train.enable_torch_compile=${ENABLE_TORCH_COMPILE}"
    "++train.torch_compile_mode=${TORCH_COMPILE_MODE}"
    "++train.torch_compile_dynamic=${TORCH_COMPILE_DYNAMIC}"
    "++train.pretrained_model_checkpoint=${PRETRAINED_CKPT}"
    "++dataset.root_dir=${DATASET_ROOT}"
    "++dataset.use_trim=${DATASET_USE_TRIM}"
    "++dataset.offline_trimmed=${DATASET_OFFLINE_TRIMMED}"
    "++model.speech_encoder.pretrained_path=${SPEECH_MODEL_PATH}"
    "++model.text_encoder.pretrained_path=${TEXT_MODEL_PATH}"
    "++model.speech_encoder.attn_implementation=${EFFECTIVE_ATTN_IMPL}"
    "++model.text_encoder.attn_implementation=${EFFECTIVE_ATTN_IMPL}"
    "deepspeed_config_yaml.train_micro_batch_size_per_gpu=${micro_batch}"
    "deepspeed_config_yaml.zero_optimization.stage=${zero_stage}"
    "deepspeed_config_yaml.wall_clock_breakdown=${wall_clock_breakdown}"
    "deepspeed_config_yaml.bf16.enabled=${ds_bf16_enabled}"
    "deepspeed_config_yaml.fp16.enabled=${ds_fp16_enabled}"
  )
  if [[ -n "${DATASET_MANIFEST_PATH}" ]]; then
    cmd+=("++dataset.manifest_path=${DATASET_MANIFEST_PATH}")
  fi
  if [[ "${ENABLE_LENGTH_FIXED_SLICE}" == "true" ]]; then
    cmd+=("++train.fixed_slice_seconds=${FIXED_SLICE_SECONDS}")
  fi

  local status="success"
  local exit_code=0
  local run_start_epoch=0
  local run_end_epoch=0
  local duration_sec=0
  local last_step=""
  local last_iter_ms_p50=""

  echo "[INFO] START ${group} r${repeat}: z=${zero_stage}, mb=${micro_batch}, nw=${num_workers}, pf=${prefetch_factor}, port=${port}, precision=${EFFECTIVE_PRECISION_MODE}, attn_impl=${EFFECTIVE_ATTN_IMPL}, compile=${ENABLE_TORCH_COMPILE}"
  echo "[INFO] Logs: launcher=${launcher_log}, train=${train_log}"

  local exec_cmd=()
  if [[ -n "${CONDA_ENV}" ]]; then
    exec_cmd=(conda run -n "${CONDA_ENV}" "${cmd[@]}")
  else
    exec_cmd=("${cmd[@]}")
  fi

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
    echo "precision_mode_req=${PRECISION_MODE}"
    echo "precision_mode_effective=${EFFECTIVE_PRECISION_MODE}"
    echo "attn_impl_req=${ATTN_IMPL}"
    echo "attn_impl_effective=${EFFECTIVE_ATTN_IMPL}"
    echo "torch_compile_enabled=${ENABLE_TORCH_COMPILE}"
    echo "torch_compile_mode=${TORCH_COMPILE_MODE}"
    echo "torch_compile_dynamic=${TORCH_COMPILE_DYNAMIC}"
    echo "gpu_name=${DETECTED_GPU_NAME}"
    echo "gpu_cc=${DETECTED_GPU_CC}"
    echo "enable_length_fixed_slice=${ENABLE_LENGTH_FIXED_SLICE}"
    echo "fixed_slice_seconds=${FIXED_SLICE_SECONDS}"
    echo "run_dir=${run_dir}"
    echo "=== CMD ==="
    printf '%q ' "${cmd[@]}"
    echo
    echo "============"
  } > "${launcher_log}" 2>&1

  run_start_epoch="$(date +%s)"

  if [[ "${RUN_TIMEOUT_SEC}" -gt 0 ]]; then
    timeout --signal=TERM --kill-after=60 "${RUN_TIMEOUT_SEC}" "${exec_cmd[@]}" >> "${launcher_log}" 2>&1 &
  else
    "${exec_cmd[@]}" >> "${launcher_log}" 2>&1 &
  fi
  local run_pid="$!"
  local heartbeat_pid=""
  heartbeat_pid="$(start_heartbeat "${run_pid}" "${group}" "${repeat}" "${train_log}" "${HEARTBEAT_EVERY_SEC}")"

  if wait "${run_pid}"; then
    exit_code=0
  else
    exit_code=$?
    status="failed"
  fi

  if [[ -n "${heartbeat_pid}" ]]; then
    kill "${heartbeat_pid}" 2>/dev/null || true
    wait "${heartbeat_pid}" 2>/dev/null || true
  fi

  run_end_epoch="$(date +%s)"
  duration_sec=$((run_end_epoch - run_start_epoch))
  last_step="$(extract_last_step "${train_log}")"
  last_iter_ms_p50="$(extract_last_iter_ms "${train_log}")"

  local manifest_row=(
    "${mode_name}" "${group}" "${repeat}" "${status}" "${exit_code}" "${duration_sec}" "${last_step}" "${last_iter_ms_p50}" "${port}" "${run_dir}" "${launcher_log}" "${train_log}"
    "${zero_stage}" "${micro_batch}" "${WORLD_SIZE}" "${global_batch}"
    "${deterministic}" "${cudnn_benchmark}" "${wall_clock_breakdown}"
    "${log_every}" "${val_every}" "${ckpt_every}" "${num_workers}" "${prefetch_factor}"
    "${DATASET_MANIFEST_PATH}" "${DATASET_USE_TRIM}" "${DATASET_OFFLINE_TRIMMED}" "${ENABLE_CUDA_SYNC_TIMING}" "${TIMING_RANK_SCOPE}"
    "${PRECISION_MODE}" "${EFFECTIVE_PRECISION_MODE}" "${EFFECTIVE_ATTN_IMPL}" "${ENABLE_TORCH_COMPILE}" "${DETECTED_GPU_NAME}" "${DETECTED_GPU_CC}"
  )
  (
    IFS=$'\t'
    printf '%s\n' "${manifest_row[*]}"
  ) >> "${MANIFEST_PATH}"

  if [[ "${status}" == "failed" ]]; then
    echo "[WARN] ${group} repeat ${repeat} failed (exit_code=${exit_code}, duration_sec=${duration_sec}, last_step=${last_step:-N/A})" >&2
    echo "[WARN] see ${launcher_log}" >&2
    if [[ "${exit_code}" -eq 124 ]]; then
      echo "[WARN] timeout hit: RUN_TIMEOUT_SEC=${RUN_TIMEOUT_SEC}" >&2
    elif [[ "${exit_code}" -eq 137 ]]; then
      echo "[WARN] exit_code=137 often indicates OOM-kill or external SIGKILL." >&2
    fi
    if [[ "${FAILURE_DUMP_TAIL}" == "true" ]]; then
      echo "[WARN] ===== launcher tail (last ${FAIL_TAIL_LINES} lines) =====" >&2
      tail -n "${FAIL_TAIL_LINES}" "${launcher_log}" >&2 || true
      if [[ -f "${train_log}" ]]; then
        echo "[WARN] ===== train tail (last ${FAIL_TAIL_LINES} lines) =====" >&2
        tail -n "${FAIL_TAIL_LINES}" "${train_log}" >&2 || true
      fi
    fi
    if [[ "${STOP_ON_ERROR}" == "1" ]]; then
      exit 1
    fi
  else
    COMPLETED_RUNS["${run_key}"]=1
    echo "[INFO] DONE ${group} r${repeat}: duration_sec=${duration_sec}, last_step=${last_step:-N/A}, last_iter_ms_p50=${last_iter_ms_p50:-N/A}" >&2
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUMMARY_SCRIPT="${SCRIPT_DIR}/summarize_stage1_bench.py"

SUMMARY_RAN=0
INTERRUPTED=0

handle_interrupt() {
  INTERRUPTED=1
  echo "[WARN] Received interrupt signal. Writing partial benchmark summary..." >&2
  exit 130
}

generate_summary() {
  if [[ "${SUMMARY_RAN}" -eq 1 ]]; then
    return 0
  fi
  SUMMARY_RAN=1

  if [[ ! -f "${MANIFEST_PATH}" ]]; then
    echo "[WARN] Manifest not found. Skip summary generation: ${MANIFEST_PATH}" >&2
    return 0
  fi
  local line_count
  line_count=$(wc -l < "${MANIFEST_PATH}")
  if [[ "${line_count}" -le 1 ]]; then
    echo "[WARN] Manifest has no run rows yet. Skip summary generation." >&2
    return 0
  fi

  "${PYTHON_CMD[@]}" "${SUMMARY_SCRIPT}" \
    --manifest "${MANIFEST_PATH}" \
    --output-root "${OUTPUT_ROOT}" \
    --tail-points "${TAIL_TIMING_POINTS}" \
    --baseline-group "${BASELINE_GROUP}"
}

finalize_on_exit() {
  local exit_code="$1"
  if [[ "${SUMMARY_RAN}" -eq 0 ]]; then
    if ! generate_summary; then
      echo "[WARN] Failed to generate benchmark summary during exit." >&2
    fi
  fi
  if [[ "${INTERRUPTED}" -eq 1 || "${exit_code}" -eq 130 ]]; then
    echo "[WARN] Benchmark interrupted. Partial outputs are under: ${OUTPUT_ROOT}" >&2
  fi
}

trap handle_interrupt INT TERM
trap 'finalize_on_exit $?' EXIT

init_manifest

echo "[INFO] Output root: ${OUTPUT_ROOT}"
echo "[INFO] Mode=${MODE}, REPEATS=${REPEATS}, MAX_STEPS=${MAX_STEPS}, INCLUDE=${INCLUDE}, WORLD_SIZE=${WORLD_SIZE}"
echo "[INFO] Dataset root=${DATASET_ROOT}, manifest=${DATASET_MANIFEST_PATH:-<none>}, use_trim=${DATASET_USE_TRIM}, offline_trimmed=${DATASET_OFFLINE_TRIMMED}"
echo "[INFO] Timing flags: enable_cuda_sync_timing=${ENABLE_CUDA_SYNC_TIMING}, timing_rank_scope=${TIMING_RANK_SCOPE}"
echo "[INFO] Runtime profile: gpu_name=${DETECTED_GPU_NAME}, gpu_cc=${DETECTED_GPU_CC}, min_cc_major=${DETECTED_MIN_CC_MAJOR}, flash_attn2_available=${FLASH_ATTN2_AVAILABLE}"
echo "[INFO] Precision/attention: requested_precision=${PRECISION_MODE}, effective_precision=${EFFECTIVE_PRECISION_MODE}, requested_attn_impl=${ATTN_IMPL}, effective_attn_impl=${EFFECTIVE_ATTN_IMPL}"
echo "[INFO] Compile/fixed-slice: enable_torch_compile=${ENABLE_TORCH_COMPILE}, torch_compile_mode=${TORCH_COMPILE_MODE}, torch_compile_dynamic=${TORCH_COMPILE_DYNAMIC}, enable_length_fixed_slice=${ENABLE_LENGTH_FIXED_SLICE}, fixed_slice_seconds=${FIXED_SLICE_SECONDS:-<none>}"

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
  if [[ -n "${SWEEP_NUM_WORKERS_LIST}" ]]; then
    split_csv_to_array "${SWEEP_NUM_WORKERS_LIST}" SWEEP_NWS
  else
    SWEEP_NWS=("${SWEEP_NUM_WORKERS}")
  fi
  if [[ -n "${SWEEP_PREFETCH_LIST}" ]]; then
    split_csv_to_array "${SWEEP_PREFETCH_LIST}" SWEEP_PFS
  else
    SWEEP_PFS=("${SWEEP_PREFETCH_FACTOR}")
  fi

  for z in "${SWEEP_ZS[@]}"; do
    require_nonneg_int "SWEEP_ZERO_STAGE item" "${z}"
  done
  for mb in "${SWEEP_MBS[@]}"; do
    require_pos_int "SWEEP_MICRO_BATCH item" "${mb}"
  done
  for nw in "${SWEEP_NWS[@]}"; do
    require_pos_int "SWEEP_NUM_WORKERS item" "${nw}"
  done
  for pf in "${SWEEP_PFS[@]}"; do
    require_pos_int "SWEEP_PREFETCH_FACTOR item" "${pf}"
  done

  for z in "${SWEEP_ZS[@]}"; do
    for mb in "${SWEEP_MBS[@]}"; do
      for nw in "${SWEEP_NWS[@]}"; do
        for pf in "${SWEEP_PFS[@]}"; do
          local_group="sweep_z${z}_mb${mb}_nw${nw}_pf${pf}"
          for repeat in $(seq 1 "${REPEATS}"); do
            run_case "sweep" "${local_group}" "${repeat}" \
              "${SWEEP_LOG_EVERY}" "${SWEEP_VALIDATION_EVERY}" "${SWEEP_CHECKPOINT_EVERY}" \
              "${SWEEP_DETERMINISTIC}" "${SWEEP_CUDNN_BENCHMARK}" "${SWEEP_WALL_CLOCK_BREAKDOWN}" \
              "${mb}" "${z}" "${nw}" "${pf}"
          done
        done
      done
    done
  done
fi

if ! generate_summary; then
  echo "[WARN] Summary generation failed." >&2
fi

echo "[INFO] Benchmark finished."
echo "[INFO] Manifest: ${MANIFEST_PATH}"
echo "[INFO] Report: ${OUTPUT_ROOT}/summary.md"
