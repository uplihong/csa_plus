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
# - <group_run>/gpu_telemetry.csv (optional)
# - <group_run>/host_telemetry.csv (optional)

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/bench_stage1_ab_${TIMESTAMP}}"
DRIVER_LOG_PATH="${DRIVER_LOG_PATH:-}"

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
MAX_WAIT_TRAIN_LOG_SEC="${MAX_WAIT_TRAIN_LOG_SEC:-900}" # 0 means disabled
FAILURE_DUMP_TAIL="${FAILURE_DUMP_TAIL:-true}"        # print log tails on failure
FAIL_TAIL_LINES="${FAIL_TAIL_LINES:-80}"
RESUME_RUNS="${RESUME_RUNS:-true}"                    # reuse existing run_manifest.tsv
MANIFEST_WRITE_RETRIES="${MANIFEST_WRITE_RETRIES:-3}"
MANIFEST_WRITE_RETRY_SLEEP_SEC="${MANIFEST_WRITE_RETRY_SLEEP_SEC:-2}"

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
SPEECH_ATTN_IMPL="${SPEECH_ATTN_IMPL:-${ATTN_IMPL}}"
TEXT_ATTN_IMPL="${TEXT_ATTN_IMPL:-${ATTN_IMPL}}"
MODEL_LOAD_DTYPE="${MODEL_LOAD_DTYPE:-auto}"    # auto | bf16 | fp16 | fp32
ENABLE_TF32="${ENABLE_TF32:-true}"              # only effective on Ampere/Ada/Hopper
MATMUL_PRECISION="${MATMUL_PRECISION:-high}"    # high | medium | highest
ENABLE_TORCH_COMPILE="${ENABLE_TORCH_COMPILE:-false}"
TORCH_COMPILE_MODE="${TORCH_COMPILE_MODE:-max-autotune}"
TORCH_COMPILE_DYNAMIC="${TORCH_COMPILE_DYNAMIC:-true}"
ENABLE_LENGTH_FIXED_SLICE="${ENABLE_LENGTH_FIXED_SLICE:-false}"
FIXED_SLICE_SECONDS="${FIXED_SLICE_SECONDS:-}"
ENABLE_LENGTH_BUCKET="${ENABLE_LENGTH_BUCKET:-false}"
LENGTH_BUCKET_BOUNDARIES_SEC="${LENGTH_BUCKET_BOUNDARIES_SEC:-1.0,1.5,2.0,2.5,3.0}"
ENABLE_GPU_TELEMETRY="${ENABLE_GPU_TELEMETRY:-true}"
GPU_TELEMETRY_INTERVAL_SEC="${GPU_TELEMETRY_INTERVAL_SEC:-2}"
ENABLE_HOST_TELEMETRY="${ENABLE_HOST_TELEMETRY:-false}"
HOST_TELEMETRY_INTERVAL_SEC="${HOST_TELEMETRY_INTERVAL_SEC:-2}"
STALL_ALERT_RATIO="${STALL_ALERT_RATIO:-2.0}"

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
MANIFEST_FALLBACK_PATH="${MANIFEST_FALLBACK_PATH:-/tmp/csa_plus_run_manifest_fallback_${TIMESTAMP}_$$.tsv}"

MANIFEST_HEADER=$'mode\tgroup\trepeat\tstatus\texit_code\tduration_sec\tlast_step\tlast_iter_ms_p50\titer_p90_over_p50\tdata_p90_over_p50\tstep_p90_over_p50\tunstable_run_flag\tport\trun_dir\tlauncher_log\ttrain_log\tzero_stage\tmicro_batch\tworld_size\tglobal_batch\tdeterministic\tcudnn_benchmark\twall_clock_breakdown\tlog_every_steps\tvalidation_every_steps\tcheckpoint_every_steps\tnum_workers\tprefetch_factor\tdataset_manifest_path\tdataset_use_trim\tdataset_offline_trimmed\tenable_cuda_sync_timing\ttiming_rank_scope\tprecision_mode_req\tprecision_mode_effective\tmodel_load_dtype_effective\tattn_impl_effective\tspeech_attn_impl_effective\ttext_attn_impl_effective\ttorch_compile_enabled\ttf32_enabled\tgpu_name\tgpu_cc\tgpu_uuid_list\tgpu_power_limit_w\tpcie_gen\tdriver_version\tgit_commit_hash\tgit_commit_short\tgit_branch\tgit_dirty\tgpu_telemetry_rows\tgpu_telemetry_empty_flag\thost_telemetry_rows\thost_telemetry_empty_flag'
declare -A COMPLETED_RUNS

PYTHON_CMD=(python)
if [[ -n "${CONDA_ENV}" ]]; then
  PYTHON_CMD=(conda run -n "${CONDA_ENV}" python)
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GIT_COMMIT_HASH="unknown"
GIT_COMMIT_SHORT="unknown"
GIT_BRANCH="unknown"
GIT_DIRTY="unknown"

detect_git_metadata() {
  if ! command -v git >/dev/null 2>&1; then
    return 0
  fi
  if ! git -C "${REPO_ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    return 0
  fi

  local commit_hash=""
  local commit_short=""
  local branch=""
  commit_hash="$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || true)"
  commit_short="$(git -C "${REPO_ROOT}" rev-parse --short=12 HEAD 2>/dev/null || true)"
  branch="$(git -C "${REPO_ROOT}" rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
  if [[ -n "${commit_hash}" ]]; then
    GIT_COMMIT_HASH="${commit_hash}"
  fi
  if [[ -n "${commit_short}" ]]; then
    GIT_COMMIT_SHORT="${commit_short}"
  fi
  if [[ -n "${branch}" ]]; then
    GIT_BRANCH="${branch}"
  fi
  if [[ -n "$(git -C "${REPO_ROOT}" status --porcelain 2>/dev/null || true)" ]]; then
    GIT_DIRTY="true"
  else
    GIT_DIRTY="false"
  fi
}

detect_git_metadata

# Optional internal driver log. Prefer this over external `| tee ...` to avoid
# missing-directory issues before the script starts.
if [[ -n "${DRIVER_LOG_PATH}" ]]; then
  mkdir -p "$(dirname "${DRIVER_LOG_PATH}")"
  exec > >(tee -a "${DRIVER_LOG_PATH}") 2>&1
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

require_model_load_dtype() {
  local value="$1"
  if [[ "${value}" != "auto" && "${value}" != "bf16" && "${value}" != "fp16" && "${value}" != "fp32" ]]; then
    echo "MODEL_LOAD_DTYPE must be one of: auto, bf16, fp16, fp32. got: ${value}" >&2
    exit 1
  fi
}

require_matmul_precision() {
  local value="$1"
  if [[ "${value}" != "high" && "${value}" != "medium" && "${value}" != "highest" ]]; then
    echo "MATMUL_PRECISION must be one of: high, medium, highest. got: ${value}" >&2
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

normalize_positive_float_csv() {
  local name="$1"
  local raw="$2"
  local normalized=()
  local token=""
  IFS=',' read -r -a _tokens <<< "${raw}"
  if [[ "${#_tokens[@]}" -eq 0 ]]; then
    echo "${name} must contain at least one item." >&2
    exit 1
  fi
  for token in "${_tokens[@]}"; do
    token="${token//[[:space:]]/}"
    if [[ -z "${token}" ]]; then
      continue
    fi
    if [[ ! "${token}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
      echo "${name} contains invalid float item: ${token}" >&2
      exit 1
    fi
    awk -v v="${token}" 'BEGIN{ exit !(v > 0) }' || {
      echo "${name} item must be > 0, got: ${token}" >&2
      exit 1
    }
    normalized+=("${token}")
  done
  if [[ "${#normalized[@]}" -eq 0 ]]; then
    echo "${name} must contain at least one positive float item." >&2
    exit 1
  fi
  local IFS=','
  echo "${normalized[*]}"
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
require_bool "ENABLE_LENGTH_BUCKET" "${ENABLE_LENGTH_BUCKET}"
require_bool "ENABLE_GPU_TELEMETRY" "${ENABLE_GPU_TELEMETRY}"
require_bool "ENABLE_HOST_TELEMETRY" "${ENABLE_HOST_TELEMETRY}"
require_bool "ENABLE_TF32" "${ENABLE_TF32}"
require_timing_rank_scope "${TIMING_RANK_SCOPE}"
require_precision_mode "${PRECISION_MODE}"
require_attn_impl "${ATTN_IMPL}"
require_attn_impl "${SPEECH_ATTN_IMPL}"
require_attn_impl "${TEXT_ATTN_IMPL}"
require_model_load_dtype "${MODEL_LOAD_DTYPE}"
require_matmul_precision "${MATMUL_PRECISION}"
require_nonneg_int "RUN_TIMEOUT_SEC" "${RUN_TIMEOUT_SEC}"
require_nonneg_int "HEARTBEAT_EVERY_SEC" "${HEARTBEAT_EVERY_SEC}"
require_nonneg_int "MAX_WAIT_TRAIN_LOG_SEC" "${MAX_WAIT_TRAIN_LOG_SEC}"
require_pos_int "GPU_TELEMETRY_INTERVAL_SEC" "${GPU_TELEMETRY_INTERVAL_SEC}"
require_pos_int "HOST_TELEMETRY_INTERVAL_SEC" "${HOST_TELEMETRY_INTERVAL_SEC}"
require_pos_int "FAIL_TAIL_LINES" "${FAIL_TAIL_LINES}"
require_pos_int "MANIFEST_WRITE_RETRIES" "${MANIFEST_WRITE_RETRIES}"
require_nonneg_int "MANIFEST_WRITE_RETRY_SLEEP_SEC" "${MANIFEST_WRITE_RETRY_SLEEP_SEC}"
require_nonempty "TORCH_COMPILE_MODE" "${TORCH_COMPILE_MODE}"
require_pos_float "STALL_ALERT_RATIO" "${STALL_ALERT_RATIO}"

if [[ "${ENABLE_LENGTH_FIXED_SLICE}" == "true" ]]; then
  require_nonempty "FIXED_SLICE_SECONDS" "${FIXED_SLICE_SECONDS}"
  require_pos_float "FIXED_SLICE_SECONDS" "${FIXED_SLICE_SECONDS}"
fi

if [[ "${ENABLE_LENGTH_BUCKET}" == "true" ]]; then
  require_nonempty "LENGTH_BUCKET_BOUNDARIES_SEC" "${LENGTH_BUCKET_BOUNDARIES_SEC}"
  EFFECTIVE_LENGTH_BUCKET_BOUNDARIES_SEC="$(normalize_positive_float_csv "LENGTH_BUCKET_BOUNDARIES_SEC" "${LENGTH_BUCKET_BOUNDARIES_SEC}")"
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
DETECTED_GPU_UUID_LIST="unknown"
DETECTED_GPU_POWER_LIMIT_W="unknown"
DETECTED_PCIE_GEN="unknown"
DETECTED_DRIVER_VERSION="unknown"
DETECTED_MIN_CC_MAJOR="0"
FLASH_ATTN2_AVAILABLE="false"
EFFECTIVE_PRECISION_MODE=""
EFFECTIVE_MODEL_LOAD_DTYPE=""
EFFECTIVE_ATTN_IMPL=""
EFFECTIVE_SPEECH_ATTN_IMPL=""
EFFECTIVE_TEXT_ATTN_IMPL=""
EFFECTIVE_TF32_ENABLED="false"
EFFECTIVE_LENGTH_BUCKET_BOUNDARIES_SEC=""

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

detect_gpu_runtime_snapshot() {
  local include="$1"
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 0
  fi

  local id_filter=""
  local include_after_colon="${include#*:}"
  if [[ "${include_after_colon}" != "${include}" ]]; then
    local token=""
    local requested_ids=()
    IFS=',' read -r -a _ids <<< "${include_after_colon}"
    for token in "${_ids[@]}"; do
      token="${token//[[:space:]]/}"
      if [[ "${token}" =~ ^[0-9]+$ ]]; then
        requested_ids+=("${token}")
      fi
    done
    if [[ "${#requested_ids[@]}" -gt 0 ]]; then
      local IFS=','
      id_filter="${requested_ids[*]}"
    fi
  fi

  local query_cmd=(
    nvidia-smi
    --query-gpu=index,uuid,power.limit,pcie.link.gen.max,driver_version
    --format=csv,noheader,nounits
  )
  if [[ -n "${id_filter}" ]]; then
    query_cmd+=(--id="${id_filter}")
  fi

  local raw=""
  raw="$("${query_cmd[@]}" 2>/dev/null || true)"
  if [[ -z "${raw}" && -n "${id_filter}" ]]; then
    echo "[WARN] Runtime snapshot query returned empty with --id=${id_filter}; retrying without id filter." >&2
    raw="$(nvidia-smi --query-gpu=index,uuid,power.limit,pcie.link.gen.max,driver_version --format=csv,noheader,nounits 2>/dev/null || true)"
  fi
  if [[ -z "${raw}" ]]; then
    return 0
  fi

  local uuids=()
  local power_limits=()
  local pcie_gens=()
  local drivers=()
  local line=""
  while IFS= read -r line; do
    [[ -z "${line}" ]] && continue
    local idx="" uuid="" power_limit="" pcie_gen="" driver=""
    IFS=',' read -r idx uuid power_limit pcie_gen driver <<< "${line}"
    uuid="$(echo "${uuid}" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
    power_limit="$(echo "${power_limit}" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
    pcie_gen="$(echo "${pcie_gen}" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
    driver="$(echo "${driver}" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
    [[ -n "${uuid}" ]] && uuids+=("${uuid}")
    [[ -n "${power_limit}" ]] && power_limits+=("${power_limit}")
    [[ -n "${pcie_gen}" ]] && pcie_gens+=("${pcie_gen}")
    [[ -n "${driver}" ]] && drivers+=("${driver}")
  done <<< "${raw}"

  if [[ "${#uuids[@]}" -gt 0 ]]; then
    local IFS=','
    DETECTED_GPU_UUID_LIST="${uuids[*]}"
  fi
  if [[ "${#power_limits[@]}" -gt 0 ]]; then
    local IFS=','
    DETECTED_GPU_POWER_LIMIT_W="${power_limits[*]}"
  fi
  if [[ "${#pcie_gens[@]}" -gt 0 ]]; then
    local IFS=','
    DETECTED_PCIE_GEN="${pcie_gens[*]}"
  fi
  if [[ "${#drivers[@]}" -gt 0 ]]; then
    DETECTED_DRIVER_VERSION="${drivers[0]}"
  fi
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
  local source_name="${5:-ATTN_IMPL}"
  if [[ "${requested}" == "auto" ]]; then
    if [[ "${precision_mode}" == "fp16" ]]; then
      echo "eager"
    else
      echo "sdpa"
    fi
    return 0
  fi

  if [[ "${requested}" == "flash_attention_2" ]] && { [[ "${flash_available}" != "true" ]] || [[ ! "${min_cc_major}" =~ ^[0-9]+$ ]] || [[ "${min_cc_major}" -lt 8 ]]; }; then
    echo "[WARN] ${source_name}=flash_attention_2 requested, but flash-attn is unavailable or GPU capability is < 8.0. Fallback to sdpa." >&2
    echo "sdpa"
    return 0
  fi
  echo "${requested}"
}

resolve_model_load_dtype() {
  local requested="$1"
  local precision_mode="$2"
  case "${requested}" in
    auto)
      if [[ "${precision_mode}" == "bf16" || "${precision_mode}" == "fp16" ]]; then
        echo "${precision_mode}"
      else
        echo "fp32"
      fi
      ;;
    bf16|fp16|fp32)
      echo "${requested}"
      ;;
    *)
      echo "fp32"
      ;;
  esac
}

resolve_tf32_enabled() {
  local requested="$1"
  local min_cc_major="$2"
  if [[ "${requested}" != "true" ]]; then
    echo "false"
    return 0
  fi
  if [[ "${min_cc_major}" =~ ^[0-9]+$ ]] && [[ "${min_cc_major}" -ge 8 ]]; then
    echo "true"
  else
    echo "[WARN] ENABLE_TF32=true requested, but GPU capability is < 8.0. Force disable TF32." >&2
    echo "false"
  fi
}

harmonize_flash_dtype() {
  local encoder_name="$1"
  local attn_impl="$2"
  local model_dtype="$3"
  local precision_mode="$4"

  if [[ "${attn_impl}" == "flash_attention_2" && ( -z "${model_dtype}" || "${model_dtype}" == "fp32" ) ]]; then
    if [[ "${precision_mode}" == "bf16" || "${precision_mode}" == "fp16" ]]; then
      echo "[WARN] ${encoder_name}: flash_attention_2 requires bf16/fp16. Override model dtype ${model_dtype:-<unset>} -> ${precision_mode}." >&2
      echo "${precision_mode}"
      return 0
    fi
    echo "[WARN] ${encoder_name}: flash_attention_2 requested but no compatible dtype could be derived from precision=${precision_mode}; keeping dtype=${model_dtype:-<unset>}." >&2
  fi
  echo "${model_dtype}"
}

derive_unified_attn_impl() {
  local speech_impl="$1"
  local text_impl="$2"
  if [[ "${speech_impl}" == "${text_impl}" ]]; then
    echo "${speech_impl}"
  else
    echo "split"
  fi
}

detect_hardware_profile "${INCLUDE}"
detect_gpu_runtime_snapshot "${INCLUDE}"
EFFECTIVE_PRECISION_MODE="$(resolve_precision_mode "${PRECISION_MODE}" "${DETECTED_MIN_CC_MAJOR}")"
EFFECTIVE_MODEL_LOAD_DTYPE="$(resolve_model_load_dtype "${MODEL_LOAD_DTYPE}" "${EFFECTIVE_PRECISION_MODE}")"
EFFECTIVE_SPEECH_ATTN_IMPL="$(resolve_attn_impl "${SPEECH_ATTN_IMPL}" "${EFFECTIVE_PRECISION_MODE}" "${FLASH_ATTN2_AVAILABLE}" "${DETECTED_MIN_CC_MAJOR}" "SPEECH_ATTN_IMPL")"
EFFECTIVE_TEXT_ATTN_IMPL="$(resolve_attn_impl "${TEXT_ATTN_IMPL}" "${EFFECTIVE_PRECISION_MODE}" "${FLASH_ATTN2_AVAILABLE}" "${DETECTED_MIN_CC_MAJOR}" "TEXT_ATTN_IMPL")"
EFFECTIVE_MODEL_LOAD_DTYPE="$(harmonize_flash_dtype "speech_encoder" "${EFFECTIVE_SPEECH_ATTN_IMPL}" "${EFFECTIVE_MODEL_LOAD_DTYPE}" "${EFFECTIVE_PRECISION_MODE}")"
EFFECTIVE_MODEL_LOAD_DTYPE="$(harmonize_flash_dtype "text_encoder" "${EFFECTIVE_TEXT_ATTN_IMPL}" "${EFFECTIVE_MODEL_LOAD_DTYPE}" "${EFFECTIVE_PRECISION_MODE}")"
EFFECTIVE_ATTN_IMPL="$(derive_unified_attn_impl "${EFFECTIVE_SPEECH_ATTN_IMPL}" "${EFFECTIVE_TEXT_ATTN_IMPL}")"
EFFECTIVE_TF32_ENABLED="$(resolve_tf32_enabled "${ENABLE_TF32}" "${DETECTED_MIN_CC_MAJOR}")"

PORT_COUNTER=0

is_port_available() {
  local port="$1"
  if [[ ! "${port}" =~ ^[0-9]+$ ]]; then
    return 1
  fi
  if [[ "${port}" -lt 1 || "${port}" -gt 65535 ]]; then
    return 1
  fi
  "${PYTHON_CMD[@]}" - "${port}" <<'PY'
import socket
import sys

port = int(sys.argv[1])
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    sock.bind(("0.0.0.0", port))
except OSError:
    raise SystemExit(1)
finally:
    sock.close()
PY
}

reserve_master_port() {
  local attempts=0
  local max_attempts=200
  while [[ "${attempts}" -lt "${max_attempts}" ]]; do
    local candidate=$((MASTER_PORT_BASE + PORT_COUNTER))
    PORT_COUNTER=$((PORT_COUNTER + 1))
    if is_port_available "${candidate}"; then
      echo "${candidate}"
      return 0
    fi
    echo "[WARN] master_port ${candidate} is already in use, trying next..." >&2
    attempts=$((attempts + 1))
  done
  echo "[ERROR] Failed to find an available master port after ${max_attempts} attempts starting at ${MASTER_PORT_BASE}" >&2
  exit 1
}

init_manifest() {
  if [[ "${RESUME_RUNS}" == "true" && -f "${MANIFEST_PATH}" ]]; then
    local existing_header
    existing_header="$(head -n 1 "${MANIFEST_PATH}" || true)"
    if [[ "${existing_header}" != "${MANIFEST_HEADER}" ]]; then
      local backup_path="${MANIFEST_PATH}.bak_$(date +%Y%m%d_%H%M%S)"
      cp "${MANIFEST_PATH}" "${backup_path}" || true
      echo "[WARN] Existing manifest header mismatch. Backed up old manifest to ${backup_path}, then resetting ${MANIFEST_PATH}" >&2
      if ! write_manifest_header_with_retry "${MANIFEST_PATH}"; then
        echo "[WARN] Failed to reset primary manifest header at ${MANIFEST_PATH}; will use fallback manifest if needed." >&2
      fi
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
    if ! write_manifest_header_with_retry "${MANIFEST_PATH}"; then
      echo "[WARN] Failed to initialize primary manifest at ${MANIFEST_PATH}; will use fallback manifest if needed." >&2
    fi
  fi
}

write_manifest_header_with_retry() {
  local target_path="$1"
  local attempts=0
  while [[ "${attempts}" -lt "${MANIFEST_WRITE_RETRIES}" ]]; do
    if printf '%s\n' "${MANIFEST_HEADER}" > "${target_path}"; then
      return 0
    fi
    attempts=$((attempts + 1))
    echo "[WARN] Failed to write manifest header to ${target_path} (attempt ${attempts}/${MANIFEST_WRITE_RETRIES})." >&2
    if [[ "${attempts}" -lt "${MANIFEST_WRITE_RETRIES}" && "${MANIFEST_WRITE_RETRY_SLEEP_SEC}" -gt 0 ]]; then
      sleep "${MANIFEST_WRITE_RETRY_SLEEP_SEC}"
    fi
  done
  return 1
}

append_manifest_line_with_retry() {
  local line="$1"
  local target_path="$2"
  local attempts=0
  while [[ "${attempts}" -lt "${MANIFEST_WRITE_RETRIES}" ]]; do
    if printf '%s\n' "${line}" >> "${target_path}"; then
      return 0
    fi
    attempts=$((attempts + 1))
    echo "[WARN] Failed to append manifest line to ${target_path} (attempt ${attempts}/${MANIFEST_WRITE_RETRIES})." >&2
    if [[ "${attempts}" -lt "${MANIFEST_WRITE_RETRIES}" && "${MANIFEST_WRITE_RETRY_SLEEP_SEC}" -gt 0 ]]; then
      sleep "${MANIFEST_WRITE_RETRY_SLEEP_SEC}"
    fi
  done
  return 1
}

ensure_fallback_manifest_ready() {
  if [[ -z "${MANIFEST_FALLBACK_PATH}" ]]; then
    return 1
  fi
  local fallback_dir
  fallback_dir="$(dirname "${MANIFEST_FALLBACK_PATH}")"
  if ! mkdir -p "${fallback_dir}"; then
    echo "[WARN] Failed to create fallback manifest directory: ${fallback_dir}" >&2
    return 1
  fi
  if [[ ! -f "${MANIFEST_FALLBACK_PATH}" ]]; then
    if ! write_manifest_header_with_retry "${MANIFEST_FALLBACK_PATH}"; then
      echo "[WARN] Failed to initialize fallback manifest: ${MANIFEST_FALLBACK_PATH}" >&2
      return 1
    fi
  fi
  return 0
}

append_manifest_row_resilient() {
  local -n manifest_row_ref="$1"
  local IFS=$'\t'
  local row_line=""
  row_line="${manifest_row_ref[*]}"

  if append_manifest_line_with_retry "${row_line}" "${MANIFEST_PATH}"; then
    return 0
  fi

  echo "[WARN] Primary manifest append failed. Falling back to ${MANIFEST_FALLBACK_PATH}" >&2
  if ! ensure_fallback_manifest_ready; then
    return 1
  fi
  if append_manifest_line_with_retry "${row_line}" "${MANIFEST_FALLBACK_PATH}"; then
    echo "[WARN] Persisted manifest row into fallback manifest: ${MANIFEST_FALLBACK_PATH}" >&2
    return 0
  fi
  return 1
}

merge_fallback_manifest_rows() {
  if [[ -z "${MANIFEST_FALLBACK_PATH}" || ! -f "${MANIFEST_FALLBACK_PATH}" ]]; then
    return 0
  fi

  local fallback_lines=0
  fallback_lines="$(wc -l < "${MANIFEST_FALLBACK_PATH}" || echo 0)"
  if [[ ! "${fallback_lines}" =~ ^[0-9]+$ ]] || [[ "${fallback_lines}" -le 1 ]]; then
    return 0
  fi

  if [[ ! -f "${MANIFEST_PATH}" ]]; then
    if ! write_manifest_header_with_retry "${MANIFEST_PATH}"; then
      echo "[WARN] Cannot create primary manifest for fallback merge: ${MANIFEST_PATH}" >&2
      return 1
    fi
  fi

  local rows_to_merge=$((fallback_lines - 1))
  echo "[WARN] Attempting to merge fallback manifest rows into primary manifest (${rows_to_merge} rows)." >&2
  local merge_failed=0
  local line=""
  local merge_input=""
  merge_input="$(mktemp /tmp/csa_plus_manifest_merge.XXXXXX)"
  if ! tail -n +2 "${MANIFEST_FALLBACK_PATH}" > "${merge_input}"; then
    echo "[WARN] Failed to read fallback manifest rows from ${MANIFEST_FALLBACK_PATH}" >&2
    rm -f "${merge_input}" || true
    return 1
  fi

  while IFS= read -r line; do
    [[ -z "${line}" ]] && continue
    if ! append_manifest_line_with_retry "${line}" "${MANIFEST_PATH}"; then
      merge_failed=1
      break
    fi
  done < "${merge_input}"
  rm -f "${merge_input}" || true

  if [[ "${merge_failed}" -eq 1 ]]; then
    echo "[WARN] Failed to fully merge fallback manifest rows; keeping fallback file: ${MANIFEST_FALLBACK_PATH}" >&2
    return 1
  fi

  rm -f "${MANIFEST_FALLBACK_PATH}" || true
  echo "[INFO] Merged fallback manifest rows into ${MANIFEST_PATH}" >&2
  return 0
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

extract_last_timing_ratios() {
  local log_path="$1"
  if [[ ! -f "${log_path}" ]]; then
    echo -e "\t\t\tfalse"
    return 0
  fi
  "${PYTHON_CMD[@]}" - "${log_path}" <<'PY'
import re
import sys

path = sys.argv[1]
pattern = re.compile(
    r"Timing\(window=\d+\): "
    r"data_wait_ms:p50=([0-9]+(?:\.[0-9]+)?) p90=([0-9]+(?:\.[0-9]+)?) \| "
    r"preprocess_ms:p50=[0-9]+(?:\.[0-9]+)? p90=[0-9]+(?:\.[0-9]+)? \| "
    r"fwd_ms:p50=[0-9]+(?:\.[0-9]+)? p90=[0-9]+(?:\.[0-9]+)? \| "
    r"bwd_ms:p50=[0-9]+(?:\.[0-9]+)? p90=[0-9]+(?:\.[0-9]+)? \| "
    r"step_ms:p50=([0-9]+(?:\.[0-9]+)?) p90=([0-9]+(?:\.[0-9]+)?) \| "
    r"iter_ms:p50=([0-9]+(?:\.[0-9]+)?) p90=([0-9]+(?:\.[0-9]+)?)"
)
last = None
with open(path, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        m = pattern.search(line)
        if m:
            last = m

if last is None:
    print("\t\t\tfalse")
    raise SystemExit(0)

data_p50, data_p90, step_p50, step_p90, iter_p50, iter_p90 = [float(v) for v in last.groups()]

def ratio(p90, p50):
    if p50 <= 0:
        return ""
    return f"{(p90 / p50):.6f}"

iter_ratio = ratio(iter_p90, iter_p50)
data_ratio = ratio(data_p90, data_p50)
step_ratio = ratio(step_p90, step_p50)
unstable = "true" if ((iter_p50 > 0 and iter_p90 / iter_p50 > 2.0) or (step_p50 > 0 and step_p90 / step_p50 > 3.0)) else "false"
print(f"{iter_ratio}\t{data_ratio}\t{step_ratio}\t{unstable}")
PY
}

count_telemetry_rows() {
  local telemetry_csv="$1"
  if [[ ! -f "${telemetry_csv}" ]]; then
    echo ""
    return 0
  fi
  local line_count=0
  line_count="$(wc -l < "${telemetry_csv}" || echo 0)"
  if [[ ! "${line_count}" =~ ^[0-9]+$ ]]; then
    echo ""
    return 0
  fi
  if [[ "${line_count}" -le 1 ]]; then
    echo "0"
    return 0
  fi
  echo $((line_count - 1))
}

start_heartbeat() {
  local out_pid_var="$1"
  local child_pid="$2"
  local child_pgid="$3"
  local group="$4"
  local repeat="$5"
  local train_log="$6"
  local interval="$7"
  local stall_alert_ratio="$8"
  local max_wait_train_log_sec="$9"

  printf -v "${out_pid_var}" "%s" ""
  if [[ "${interval}" -le 0 ]]; then
    return 0
  fi

  (
    local prev_step=""
    local prev_iter_ms=""
    local stagnant_count=0
    local train_log_missing_sec=0
    while kill -0 "${child_pid}" 2>/dev/null; do
      sleep "${interval}"
      if [[ ! -f "${train_log}" ]]; then
        train_log_missing_sec=$((train_log_missing_sec + interval))
        echo "[INFO] heartbeat ${group} r${repeat}: waiting for ${train_log}" >&2
        if [[ "${max_wait_train_log_sec}" -gt 0 && "${train_log_missing_sec}" -ge "${max_wait_train_log_sec}" ]]; then
          echo "[WARN] heartbeat ${group} r${repeat}: train.log did not appear within ${max_wait_train_log_sec}s, terminating run pid=${child_pid} pgid=${child_pgid:-N/A}" >&2
          if [[ -n "${child_pgid}" ]]; then
            kill -TERM -- "-${child_pgid}" 2>/dev/null || true
            sleep 2
            kill -KILL -- "-${child_pgid}" 2>/dev/null || true
          else
            kill -TERM "${child_pid}" 2>/dev/null || true
            sleep 2
            kill -KILL "${child_pid}" 2>/dev/null || true
          fi
          break
        fi
        continue
      fi
      train_log_missing_sec=0
      local step=""
      local iter_ms=""
      step="$(extract_last_step "${train_log}")"
      iter_ms="$(extract_last_iter_ms "${train_log}")"
      echo "[INFO] heartbeat ${group} r${repeat}: last_step=${step:-N/A}, iter_ms_p50=${iter_ms:-N/A}" >&2

      if [[ -n "${step}" && -n "${prev_step}" && "${step}" == "${prev_step}" ]]; then
        stagnant_count=$((stagnant_count + 1))
        if [[ "${stagnant_count}" -eq 2 ]] || [[ $((stagnant_count % 4)) -eq 0 ]]; then
          echo "[WARN] heartbeat ${group} r${repeat}: step has not advanced for $((stagnant_count * interval))s (step=${step})" >&2
        fi
      elif [[ -n "${step}" ]]; then
        stagnant_count=0
      fi

      if [[ -n "${iter_ms}" && -n "${prev_iter_ms}" ]]; then
        if awk -v cur="${iter_ms}" -v prev="${prev_iter_ms}" -v ratio="${stall_alert_ratio}" 'BEGIN {exit !(prev > 0 && cur > prev * ratio)}'; then
          echo "[WARN] heartbeat ${group} r${repeat}: iter_ms_p50 spike detected (${prev_iter_ms} -> ${iter_ms}, ratio>${stall_alert_ratio})" >&2
        fi
      fi

      if [[ -n "${step}" ]]; then
        prev_step="${step}"
      fi
      if [[ -n "${iter_ms}" ]]; then
        prev_iter_ms="${iter_ms}"
      fi
    done
  ) &
  printf -v "${out_pid_var}" "%s" "$!"
}

extract_include_gpu_ids() {
  local include="$1"
  local after_colon="${include#*:}"
  if [[ "${after_colon}" == "${include}" ]]; then
    echo ""
    return 0
  fi
  local result=""
  local token=""
  IFS=',' read -r -a _ids <<< "${after_colon}"
  for token in "${_ids[@]}"; do
    token="${token//[[:space:]]/}"
    if [[ "${token}" =~ ^[0-9]+$ ]]; then
      if [[ -z "${result}" ]]; then
        result="${token}"
      else
        result="${result},${token}"
      fi
    fi
  done
  echo "${result}"
}

start_gpu_telemetry() {
  local out_pid_var="$1"
  local child_pid="$2"
  local telemetry_csv="$3"
  local interval="$4"
  local include="$5"
  local gpu_ids="$6"

  printf -v "${out_pid_var}" "%s" ""
  if [[ "${ENABLE_GPU_TELEMETRY}" != "true" ]]; then
    return 0
  fi
  if [[ "${interval}" -le 0 ]]; then
    return 0
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[WARN] ENABLE_GPU_TELEMETRY=true but nvidia-smi is not available. Skip telemetry for ${telemetry_csv}" >&2
    return 0
  fi

  (
    local id_filter="${gpu_ids}"
    local used_id_fallback="false"
    local sample_rows=0
    local empty_ticks=0
    local telemetry_target="${telemetry_csv}"
    if [[ -z "${id_filter}" && "${include}" == *":"* ]]; then
      id_filter="$(extract_include_gpu_ids "${include}")"
    fi
    if ! echo "timestamp,epoch_sec,gpu_index,gpu_uuid,gpu_name,temp_c,util_gpu_pct,util_mem_pct,power_w,sm_clock_mhz,mem_clock_mhz,mem_used_mib,mem_total_mib" > "${telemetry_target}"; then
      local fallback_csv="/tmp/csa_plus_gpu_telemetry_${TIMESTAMP}_$$_${child_pid}.csv"
      echo "[WARN] Failed to initialize telemetry CSV at ${telemetry_target}; fallback to ${fallback_csv}" >&2
      if ! echo "timestamp,epoch_sec,gpu_index,gpu_uuid,gpu_name,temp_c,util_gpu_pct,util_mem_pct,power_w,sm_clock_mhz,mem_clock_mhz,mem_used_mib,mem_total_mib" > "${fallback_csv}"; then
        echo "[WARN] Failed to initialize fallback telemetry CSV ${fallback_csv}; skip telemetry for this run." >&2
        exit 0
      fi
      telemetry_target="${fallback_csv}"
    fi
    while kill -0 "${child_pid}" 2>/dev/null; do
      local ts
      local epoch
      local raw=""
      ts="$(date -Is)"
      epoch="$(date +%s)"

      local nvsmi_cmd=(
        nvidia-smi
        --query-gpu=index,uuid,name,temperature.gpu,utilization.gpu,utilization.memory,power.draw,clocks.sm,clocks.mem,memory.used,memory.total
        --format=csv,noheader,nounits
      )
      if [[ -n "${id_filter}" ]]; then
        nvsmi_cmd+=(--id="${id_filter}")
      fi
      raw="$("${nvsmi_cmd[@]}" 2>/dev/null || true)"
      if [[ -z "${raw}" && -n "${id_filter}" && "${used_id_fallback}" == "false" ]]; then
        echo "[WARN] Telemetry query returned empty with --id=${id_filter}; retrying without id filter for ${telemetry_target}" >&2
        used_id_fallback="true"
        id_filter=""
        raw="$(nvidia-smi --query-gpu=index,uuid,name,temperature.gpu,utilization.gpu,utilization.memory,power.draw,clocks.sm,clocks.mem,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || true)"
      fi

      if [[ -n "${raw}" ]]; then
        local line=""
        empty_ticks=0
        while IFS= read -r line; do
          [[ -z "${line}" ]] && continue
          local idx uuid name temp util_gpu util_mem power sm_clk mem_clk mem_used mem_total
          IFS=',' read -r idx uuid name temp util_gpu util_mem power sm_clk mem_clk mem_used mem_total <<< "${line}"
          echo "${ts},${epoch},${idx//[[:space:]]/},${uuid//[[:space:]]/},$(echo "${name}" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//'),${temp//[[:space:]]/},${util_gpu//[[:space:]]/},${util_mem//[[:space:]]/},${power//[[:space:]]/},${sm_clk//[[:space:]]/},${mem_clk//[[:space:]]/},${mem_used//[[:space:]]/},${mem_total//[[:space:]]/}" >> "${telemetry_target}"
          sample_rows=$((sample_rows + 1))
        done <<< "${raw}"
      else
        empty_ticks=$((empty_ticks + 1))
        if [[ "${empty_ticks}" -eq 1 || $((empty_ticks % 15)) -eq 0 ]]; then
          echo "[WARN] Telemetry query produced no rows (tick=${empty_ticks}): ${telemetry_target}" >&2
        fi
      fi
      sleep "${interval}"
    done
    if [[ "${sample_rows}" -eq 0 ]]; then
      echo "[WARN] Telemetry exited with zero samples: ${telemetry_target}" >&2
    fi
    if [[ "${telemetry_target}" != "${telemetry_csv}" ]]; then
      if cp "${telemetry_target}" "${telemetry_csv}" 2>/dev/null; then
        echo "[INFO] Telemetry fallback copied to requested path: ${telemetry_csv}" >&2
      else
        echo "[WARN] Telemetry data only available at fallback path: ${telemetry_target}" >&2
      fi
    fi
  ) &
  printf -v "${out_pid_var}" "%s" "$!"
}

start_host_telemetry() {
  local out_pid_var="$1"
  local child_pid="$2"
  local child_pgid="$3"
  local telemetry_csv="$4"
  local interval="$5"

  printf -v "${out_pid_var}" "%s" ""
  if [[ "${ENABLE_HOST_TELEMETRY}" != "true" ]]; then
    return 0
  fi
  if [[ "${interval}" -le 0 ]]; then
    return 0
  fi
  if [[ ! -r /proc/stat || ! -r /proc/meminfo || ! -r /proc/loadavg ]]; then
    echo "[WARN] ENABLE_HOST_TELEMETRY=true but /proc metrics are unavailable. Skip host telemetry for ${telemetry_csv}" >&2
    return 0
  fi

  (
    local telemetry_target="${telemetry_csv}"
    local sample_rows=0
    local prev_user=""
    local prev_system=""
    local prev_idle=""
    local prev_total=""
    if ! echo "timestamp,epoch_sec,load1,load5,load15,cpu_user_pct,cpu_system_pct,cpu_idle_pct,mem_total_kib,mem_available_kib,mem_used_kib,swap_total_kib,swap_free_kib,rss_kib" > "${telemetry_target}"; then
      local fallback_csv="/tmp/csa_plus_host_telemetry_${TIMESTAMP}_$$_${child_pid}.csv"
      echo "[WARN] Failed to initialize host telemetry CSV at ${telemetry_target}; fallback to ${fallback_csv}" >&2
      if ! echo "timestamp,epoch_sec,load1,load5,load15,cpu_user_pct,cpu_system_pct,cpu_idle_pct,mem_total_kib,mem_available_kib,mem_used_kib,swap_total_kib,swap_free_kib,rss_kib" > "${fallback_csv}"; then
        echo "[WARN] Failed to initialize fallback host telemetry CSV ${fallback_csv}; skip host telemetry for this run." >&2
        exit 0
      fi
      telemetry_target="${fallback_csv}"
    fi

    while kill -0 "${child_pid}" 2>/dev/null; do
      local ts epoch
      ts="$(date -Is)"
      epoch="$(date +%s)"

      local cpu_line=""
      cpu_line="$(head -n 1 /proc/stat 2>/dev/null || true)"
      local _cpu user nice system idle iowait irq softirq steal _guest _guest_nice
      read -r _cpu user nice system idle iowait irq softirq steal _guest _guest_nice <<< "${cpu_line}"
      local idle_all=$((idle + iowait))
      local non_idle=$((user + nice + system + irq + softirq + steal))
      local total=$((idle_all + non_idle))

      local cpu_user_pct="" cpu_system_pct="" cpu_idle_pct=""
      if [[ -n "${prev_total}" ]]; then
        local totald=$((total - prev_total))
        local idled=$((idle_all - prev_idle))
        local userd=$((user - prev_user))
        local systemd=$((system - prev_system))
        if [[ "${totald}" -gt 0 ]]; then
          cpu_user_pct="$(awk -v n="${userd}" -v d="${totald}" 'BEGIN{printf "%.2f", (n*100.0)/d}')"
          cpu_system_pct="$(awk -v n="${systemd}" -v d="${totald}" 'BEGIN{printf "%.2f", (n*100.0)/d}')"
          cpu_idle_pct="$(awk -v n="${idled}" -v d="${totald}" 'BEGIN{printf "%.2f", (n*100.0)/d}')"
        fi
      fi
      prev_user="${user}"
      prev_system="${system}"
      prev_idle="${idle_all}"
      prev_total="${total}"

      local load_line load1 load5 load15
      load_line="$(cat /proc/loadavg 2>/dev/null || true)"
      read -r load1 load5 load15 _rest <<< "${load_line}"

      local mem_total mem_available swap_total swap_free mem_used
      mem_total="$(awk '/^MemTotal:/ {print $2}' /proc/meminfo 2>/dev/null || true)"
      mem_available="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo 2>/dev/null || true)"
      swap_total="$(awk '/^SwapTotal:/ {print $2}' /proc/meminfo 2>/dev/null || true)"
      swap_free="$(awk '/^SwapFree:/ {print $2}' /proc/meminfo 2>/dev/null || true)"
      mem_used=""
      if [[ "${mem_total}" =~ ^[0-9]+$ && "${mem_available}" =~ ^[0-9]+$ ]]; then
        mem_used=$((mem_total - mem_available))
      fi

      local rss_kib=""
      if [[ -n "${child_pgid}" ]]; then
        rss_kib="$(ps -eo pgid=,rss= 2>/dev/null | awk -v pg="${child_pgid}" '$1==pg{sum+=$2} END{printf "%.0f", sum+0}')"
      else
        rss_kib="$(ps -o rss= -p "${child_pid}" 2>/dev/null | awk '{sum+=$1} END{printf "%.0f", sum+0}')"
      fi

      echo "${ts},${epoch},${load1:-},${load5:-},${load15:-},${cpu_user_pct},${cpu_system_pct},${cpu_idle_pct},${mem_total:-},${mem_available:-},${mem_used},${swap_total:-},${swap_free:-},${rss_kib}" >> "${telemetry_target}"
      sample_rows=$((sample_rows + 1))
      sleep "${interval}"
    done

    if [[ "${sample_rows}" -eq 0 ]]; then
      echo "[WARN] Host telemetry exited with zero samples: ${telemetry_target}" >&2
    fi
    if [[ "${telemetry_target}" != "${telemetry_csv}" ]]; then
      if cp "${telemetry_target}" "${telemetry_csv}" 2>/dev/null; then
        echo "[INFO] Host telemetry fallback copied to requested path: ${telemetry_csv}" >&2
      else
        echo "[WARN] Host telemetry data only available at fallback path: ${telemetry_target}" >&2
      fi
    fi
  ) &
  printf -v "${out_pid_var}" "%s" "$!"
}

ACTIVE_RUN_PID=""
ACTIVE_RUN_PGID=""
ACTIVE_HEARTBEAT_PID=""
ACTIVE_TELEMETRY_PID=""
ACTIVE_HOST_TELEMETRY_PID=""

stop_active_heartbeat() {
  if [[ -z "${ACTIVE_HEARTBEAT_PID}" ]]; then
    return 0
  fi
  if kill -0 "${ACTIVE_HEARTBEAT_PID}" 2>/dev/null; then
    kill "${ACTIVE_HEARTBEAT_PID}" 2>/dev/null || true
    wait "${ACTIVE_HEARTBEAT_PID}" 2>/dev/null || true
  fi
  ACTIVE_HEARTBEAT_PID=""
}

stop_active_telemetry() {
  if [[ -z "${ACTIVE_TELEMETRY_PID}" ]]; then
    return 0
  fi
  if kill -0 "${ACTIVE_TELEMETRY_PID}" 2>/dev/null; then
    kill "${ACTIVE_TELEMETRY_PID}" 2>/dev/null || true
    wait "${ACTIVE_TELEMETRY_PID}" 2>/dev/null || true
  fi
  ACTIVE_TELEMETRY_PID=""
}

stop_active_host_telemetry() {
  if [[ -z "${ACTIVE_HOST_TELEMETRY_PID}" ]]; then
    return 0
  fi
  if kill -0 "${ACTIVE_HOST_TELEMETRY_PID}" 2>/dev/null; then
    kill "${ACTIVE_HOST_TELEMETRY_PID}" 2>/dev/null || true
    wait "${ACTIVE_HOST_TELEMETRY_PID}" 2>/dev/null || true
  fi
  ACTIVE_HOST_TELEMETRY_PID=""
}

terminate_active_run() {
  local reason="${1:-cleanup}"
  if [[ -z "${ACTIVE_RUN_PID}" ]]; then
    return 0
  fi

  if ! kill -0 "${ACTIVE_RUN_PID}" 2>/dev/null; then
    ACTIVE_RUN_PID=""
    ACTIVE_RUN_PGID=""
    return 0
  fi

  echo "[WARN] ${reason}: terminating active training process (pid=${ACTIVE_RUN_PID}, pgid=${ACTIVE_RUN_PGID:-N/A})" >&2
  if [[ -n "${ACTIVE_RUN_PGID}" ]]; then
    # Prefer process-group kill so all deepspeed ranks are terminated together.
    kill -TERM -- "-${ACTIVE_RUN_PGID}" 2>/dev/null || true
    sleep 2
    kill -KILL -- "-${ACTIVE_RUN_PGID}" 2>/dev/null || true
  else
    # Fallback when setsid is unavailable and we only have the launcher PID.
    kill -TERM "${ACTIVE_RUN_PID}" 2>/dev/null || true
    sleep 2
    kill -KILL "${ACTIVE_RUN_PID}" 2>/dev/null || true
  fi

  ACTIVE_RUN_PID=""
  ACTIVE_RUN_PGID=""
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
  local telemetry_csv="${run_dir}/gpu_telemetry.csv"
  local host_telemetry_csv="${run_dir}/host_telemetry.csv"
  local run_key="${mode_name}::${group}::${repeat}"
  if [[ "${RESUME_RUNS}" == "true" && -n "${COMPLETED_RUNS["${run_key}"]+x}" ]]; then
    echo "[INFO] SKIP ${group} r${repeat}: already successful in manifest"
    return 0
  fi

  mkdir -p "${run_dir}"

  local port=""
  port="$(reserve_master_port)"

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
    "++train.data.use_length_bucket=${ENABLE_LENGTH_BUCKET}"
    "++train.enable_cuda_sync_timing=${ENABLE_CUDA_SYNC_TIMING}"
    "++train.timing_rank_scope=${TIMING_RANK_SCOPE}"
    "++train.enable_torch_compile=${ENABLE_TORCH_COMPILE}"
    "++train.torch_compile_mode=${TORCH_COMPILE_MODE}"
    "++train.torch_compile_dynamic=${TORCH_COMPILE_DYNAMIC}"
    "++train.enable_tf32=${EFFECTIVE_TF32_ENABLED}"
    "++train.matmul_precision=${MATMUL_PRECISION}"
    "++train.pretrained_model_checkpoint=${PRETRAINED_CKPT}"
    "++dataset.root_dir=${DATASET_ROOT}"
    "++dataset.use_trim=${DATASET_USE_TRIM}"
    "++dataset.offline_trimmed=${DATASET_OFFLINE_TRIMMED}"
    "++model.speech_encoder.pretrained_path=${SPEECH_MODEL_PATH}"
    "++model.text_encoder.pretrained_path=${TEXT_MODEL_PATH}"
    "++model.speech_encoder.attn_implementation=${EFFECTIVE_SPEECH_ATTN_IMPL}"
    "++model.text_encoder.attn_implementation=${EFFECTIVE_TEXT_ATTN_IMPL}"
    "++model.speech_encoder.torch_dtype=${EFFECTIVE_MODEL_LOAD_DTYPE}"
    "++model.text_encoder.torch_dtype=${EFFECTIVE_MODEL_LOAD_DTYPE}"
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
  if [[ "${ENABLE_LENGTH_BUCKET}" == "true" ]]; then
    cmd+=("++train.data.bucket_boundaries_second=[${EFFECTIVE_LENGTH_BUCKET_BOUNDARIES_SEC}]")
  fi

  local status="success"
  local exit_code=0
  local run_start_epoch=0
  local run_end_epoch=0
  local duration_sec=0
  local last_step=""
  local last_iter_ms_p50=""
  local iter_p90_over_p50=""
  local data_p90_over_p50=""
  local step_p90_over_p50=""
  local unstable_run_flag="false"
  local gpu_telemetry_rows=""
  local gpu_telemetry_empty_flag="false"
  local host_telemetry_rows=""
  local host_telemetry_empty_flag="false"
  local include_gpu_ids=""
  include_gpu_ids="$(extract_include_gpu_ids "${INCLUDE}")"

  echo "[INFO] START ${group} r${repeat}: z=${zero_stage}, mb=${micro_batch}, nw=${num_workers}, pf=${prefetch_factor}, port=${port}, precision=${EFFECTIVE_PRECISION_MODE}, model_dtype=${EFFECTIVE_MODEL_LOAD_DTYPE}, speech_attn=${EFFECTIVE_SPEECH_ATTN_IMPL}, text_attn=${EFFECTIVE_TEXT_ATTN_IMPL}, tf32=${EFFECTIVE_TF32_ENABLED}, compile=${ENABLE_TORCH_COMPILE}"
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
    echo "model_load_dtype_req=${MODEL_LOAD_DTYPE}"
    echo "model_load_dtype_effective=${EFFECTIVE_MODEL_LOAD_DTYPE}"
    echo "attn_impl_req=${ATTN_IMPL}"
    echo "attn_impl_effective=${EFFECTIVE_ATTN_IMPL}"
    echo "speech_attn_impl_req=${SPEECH_ATTN_IMPL}"
    echo "speech_attn_impl_effective=${EFFECTIVE_SPEECH_ATTN_IMPL}"
    echo "text_attn_impl_req=${TEXT_ATTN_IMPL}"
    echo "text_attn_impl_effective=${EFFECTIVE_TEXT_ATTN_IMPL}"
    echo "torch_compile_enabled=${ENABLE_TORCH_COMPILE}"
    echo "torch_compile_mode=${TORCH_COMPILE_MODE}"
    echo "torch_compile_dynamic=${TORCH_COMPILE_DYNAMIC}"
    echo "tf32_enabled=${EFFECTIVE_TF32_ENABLED}"
    echo "matmul_precision=${MATMUL_PRECISION}"
    echo "gpu_name=${DETECTED_GPU_NAME}"
    echo "gpu_cc=${DETECTED_GPU_CC}"
    echo "gpu_uuid_list=${DETECTED_GPU_UUID_LIST}"
    echo "gpu_power_limit_w=${DETECTED_GPU_POWER_LIMIT_W}"
    echo "pcie_gen=${DETECTED_PCIE_GEN}"
    echo "driver_version=${DETECTED_DRIVER_VERSION}"
    echo "git_commit_hash=${GIT_COMMIT_HASH}"
    echo "git_commit_short=${GIT_COMMIT_SHORT}"
    echo "git_branch=${GIT_BRANCH}"
    echo "git_dirty=${GIT_DIRTY}"
    echo "enable_gpu_telemetry=${ENABLE_GPU_TELEMETRY}"
    echo "gpu_telemetry_interval_sec=${GPU_TELEMETRY_INTERVAL_SEC}"
    echo "gpu_telemetry_csv=${telemetry_csv}"
    echo "enable_host_telemetry=${ENABLE_HOST_TELEMETRY}"
    echo "host_telemetry_interval_sec=${HOST_TELEMETRY_INTERVAL_SEC}"
    echo "host_telemetry_csv=${host_telemetry_csv}"
    echo "stall_alert_ratio=${STALL_ALERT_RATIO}"
    echo "enable_length_fixed_slice=${ENABLE_LENGTH_FIXED_SLICE}"
    echo "fixed_slice_seconds=${FIXED_SLICE_SECONDS}"
    echo "enable_length_bucket=${ENABLE_LENGTH_BUCKET}"
    echo "length_bucket_boundaries_sec=${EFFECTIVE_LENGTH_BUCKET_BOUNDARIES_SEC}"
    echo "run_dir=${run_dir}"
    echo "=== CMD ==="
    printf '%q ' "${cmd[@]}"
    echo
    echo "============"
  } > "${launcher_log}" 2>&1

  run_start_epoch="$(date +%s)"

  local run_pid=""
  local run_pgid=""
  if [[ "${RUN_TIMEOUT_SEC}" -gt 0 ]]; then
    if command -v setsid >/dev/null 2>&1; then
      setsid timeout --signal=TERM --kill-after=60 "${RUN_TIMEOUT_SEC}" "${exec_cmd[@]}" >> "${launcher_log}" 2>&1 &
      run_pid="$!"
      run_pgid="${run_pid}"
    else
      timeout --signal=TERM --kill-after=60 "${RUN_TIMEOUT_SEC}" "${exec_cmd[@]}" >> "${launcher_log}" 2>&1 &
      run_pid="$!"
    fi
  else
    if command -v setsid >/dev/null 2>&1; then
      setsid "${exec_cmd[@]}" >> "${launcher_log}" 2>&1 &
      run_pid="$!"
      run_pgid="${run_pid}"
    else
      "${exec_cmd[@]}" >> "${launcher_log}" 2>&1 &
      run_pid="$!"
    fi
  fi
  ACTIVE_RUN_PID="${run_pid}"
  ACTIVE_RUN_PGID="${run_pgid}"

  local heartbeat_pid=""
  start_heartbeat heartbeat_pid "${run_pid}" "${run_pgid}" "${group}" "${repeat}" "${train_log}" "${HEARTBEAT_EVERY_SEC}" "${STALL_ALERT_RATIO}" "${MAX_WAIT_TRAIN_LOG_SEC}"
  ACTIVE_HEARTBEAT_PID="${heartbeat_pid}"
  local telemetry_pid=""
  start_gpu_telemetry telemetry_pid "${run_pid}" "${telemetry_csv}" "${GPU_TELEMETRY_INTERVAL_SEC}" "${INCLUDE}" "${include_gpu_ids}"
  ACTIVE_TELEMETRY_PID="${telemetry_pid}"
  local host_telemetry_pid=""
  start_host_telemetry host_telemetry_pid "${run_pid}" "${run_pgid}" "${host_telemetry_csv}" "${HOST_TELEMETRY_INTERVAL_SEC}"
  ACTIVE_HOST_TELEMETRY_PID="${host_telemetry_pid}"

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
  if [[ -n "${telemetry_pid}" ]]; then
    kill "${telemetry_pid}" 2>/dev/null || true
    wait "${telemetry_pid}" 2>/dev/null || true
  fi
  if [[ -n "${host_telemetry_pid}" ]]; then
    kill "${host_telemetry_pid}" 2>/dev/null || true
    wait "${host_telemetry_pid}" 2>/dev/null || true
  fi
  ACTIVE_HEARTBEAT_PID=""
  ACTIVE_TELEMETRY_PID=""
  ACTIVE_HOST_TELEMETRY_PID=""
  ACTIVE_RUN_PID=""
  ACTIVE_RUN_PGID=""

  run_end_epoch="$(date +%s)"
  duration_sec=$((run_end_epoch - run_start_epoch))
  last_step="$(extract_last_step "${train_log}")"
  last_iter_ms_p50="$(extract_last_iter_ms "${train_log}")"
  IFS=$'\t' read -r iter_p90_over_p50 data_p90_over_p50 step_p90_over_p50 unstable_run_flag <<< "$(extract_last_timing_ratios "${train_log}")"
  if [[ -z "${unstable_run_flag}" ]]; then
    unstable_run_flag="false"
  fi
  gpu_telemetry_rows="$(count_telemetry_rows "${telemetry_csv}")"
  if [[ "${ENABLE_GPU_TELEMETRY}" == "true" ]]; then
    if [[ -z "${gpu_telemetry_rows}" || ! "${gpu_telemetry_rows}" =~ ^[0-9]+$ ]]; then
      gpu_telemetry_empty_flag="true"
      echo "[WARN] Telemetry enabled but no samples were collected: ${telemetry_csv}" >&2
      gpu_telemetry_rows="0"
    elif [[ "${gpu_telemetry_rows}" -le 0 ]]; then
      gpu_telemetry_empty_flag="true"
      echo "[WARN] Telemetry enabled but no samples were collected: ${telemetry_csv}" >&2
      gpu_telemetry_rows="0"
    fi
  fi
  host_telemetry_rows="$(count_telemetry_rows "${host_telemetry_csv}")"
  if [[ "${ENABLE_HOST_TELEMETRY}" == "true" ]]; then
    if [[ -z "${host_telemetry_rows}" || ! "${host_telemetry_rows}" =~ ^[0-9]+$ ]]; then
      host_telemetry_empty_flag="true"
      echo "[WARN] Host telemetry enabled but no samples were collected: ${host_telemetry_csv}" >&2
      host_telemetry_rows="0"
    elif [[ "${host_telemetry_rows}" -le 0 ]]; then
      host_telemetry_empty_flag="true"
      echo "[WARN] Host telemetry enabled but no samples were collected: ${host_telemetry_csv}" >&2
      host_telemetry_rows="0"
    fi
  fi

  local manifest_row=(
    "${mode_name}" "${group}" "${repeat}" "${status}" "${exit_code}" "${duration_sec}" "${last_step}" "${last_iter_ms_p50}" "${iter_p90_over_p50}" "${data_p90_over_p50}" "${step_p90_over_p50}" "${unstable_run_flag}" "${port}" "${run_dir}" "${launcher_log}" "${train_log}"
    "${zero_stage}" "${micro_batch}" "${WORLD_SIZE}" "${global_batch}"
    "${deterministic}" "${cudnn_benchmark}" "${wall_clock_breakdown}"
    "${log_every}" "${val_every}" "${ckpt_every}" "${num_workers}" "${prefetch_factor}"
    "${DATASET_MANIFEST_PATH}" "${DATASET_USE_TRIM}" "${DATASET_OFFLINE_TRIMMED}" "${ENABLE_CUDA_SYNC_TIMING}" "${TIMING_RANK_SCOPE}"
    "${PRECISION_MODE}" "${EFFECTIVE_PRECISION_MODE}" "${EFFECTIVE_MODEL_LOAD_DTYPE}" "${EFFECTIVE_ATTN_IMPL}" "${EFFECTIVE_SPEECH_ATTN_IMPL}" "${EFFECTIVE_TEXT_ATTN_IMPL}" "${ENABLE_TORCH_COMPILE}" "${EFFECTIVE_TF32_ENABLED}" "${DETECTED_GPU_NAME}" "${DETECTED_GPU_CC}" "${DETECTED_GPU_UUID_LIST}" "${DETECTED_GPU_POWER_LIMIT_W}" "${DETECTED_PCIE_GEN}" "${DETECTED_DRIVER_VERSION}" "${GIT_COMMIT_HASH}" "${GIT_COMMIT_SHORT}" "${GIT_BRANCH}" "${GIT_DIRTY}" "${gpu_telemetry_rows}" "${gpu_telemetry_empty_flag}" "${host_telemetry_rows}" "${host_telemetry_empty_flag}"
  )
  if ! append_manifest_row_resilient manifest_row; then
    echo "[WARN] Unable to persist manifest row for ${group} r${repeat}. run status may be missing from summary." >&2
  fi

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
    echo "[INFO] DONE ${group} r${repeat}: duration_sec=${duration_sec}, last_step=${last_step:-N/A}, last_iter_ms_p50=${last_iter_ms_p50:-N/A}, unstable_run_flag=${unstable_run_flag}, iter_p90_over_p50=${iter_p90_over_p50:-N/A}" >&2
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
  stop_active_heartbeat
  stop_active_telemetry
  stop_active_host_telemetry
  terminate_active_run "interrupt signal received"
  exit 130
}

generate_summary() {
  if [[ "${SUMMARY_RAN}" -eq 1 ]]; then
    return 0
  fi
  SUMMARY_RAN=1

  if ! merge_fallback_manifest_rows; then
    echo "[WARN] Proceeding with available manifest data after fallback-merge failure." >&2
  fi

  local manifest_for_summary="${MANIFEST_PATH}"
  if [[ ! -f "${manifest_for_summary}" ]]; then
    if [[ -n "${MANIFEST_FALLBACK_PATH}" && -f "${MANIFEST_FALLBACK_PATH}" ]]; then
      echo "[WARN] Primary manifest missing; using fallback manifest for summary: ${MANIFEST_FALLBACK_PATH}" >&2
      manifest_for_summary="${MANIFEST_FALLBACK_PATH}"
    else
      echo "[WARN] Manifest not found. Skip summary generation: ${MANIFEST_PATH}" >&2
      return 0
    fi
  fi

  if [[ ! -f "${manifest_for_summary}" ]]; then
    echo "[WARN] No readable manifest for summary generation." >&2
    return 0
  fi
  local line_count=0
  if ! line_count="$(wc -l < "${manifest_for_summary}" 2>/dev/null)"; then
    echo "[WARN] Failed to read manifest line count: ${manifest_for_summary}" >&2
    if [[ "${manifest_for_summary}" != "${MANIFEST_FALLBACK_PATH}" && -n "${MANIFEST_FALLBACK_PATH}" && -f "${MANIFEST_FALLBACK_PATH}" ]]; then
      manifest_for_summary="${MANIFEST_FALLBACK_PATH}"
      if ! line_count="$(wc -l < "${manifest_for_summary}" 2>/dev/null)"; then
        echo "[WARN] Failed to read fallback manifest line count: ${manifest_for_summary}" >&2
        return 0
      fi
    else
      return 0
    fi
  fi
  if [[ ! "${line_count}" =~ ^[0-9]+$ ]]; then
    echo "[WARN] Invalid manifest line count: ${line_count}" >&2
    return 0
  fi
  if [[ "${line_count}" -le 1 ]]; then
    echo "[WARN] Manifest has no run rows yet. Skip summary generation." >&2
    return 0
  fi

  "${PYTHON_CMD[@]}" "${SUMMARY_SCRIPT}" \
    --manifest "${manifest_for_summary}" \
    --output-root "${OUTPUT_ROOT}" \
    --tail-points "${TAIL_TIMING_POINTS}" \
    --baseline-group "${BASELINE_GROUP}"
}

finalize_on_exit() {
  local exit_code="$1"
  stop_active_heartbeat
  stop_active_telemetry
  stop_active_host_telemetry
  terminate_active_run "script exiting"
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
echo "[INFO] Runtime snapshot: gpu_uuid_list=${DETECTED_GPU_UUID_LIST}, gpu_power_limit_w=${DETECTED_GPU_POWER_LIMIT_W}, pcie_gen=${DETECTED_PCIE_GEN}, driver_version=${DETECTED_DRIVER_VERSION}"
echo "[INFO] Precision/attention: requested_precision=${PRECISION_MODE}, effective_precision=${EFFECTIVE_PRECISION_MODE}, requested_model_dtype=${MODEL_LOAD_DTYPE}, effective_model_dtype=${EFFECTIVE_MODEL_LOAD_DTYPE}, requested_attn_impl=${ATTN_IMPL}, effective_attn_impl=${EFFECTIVE_ATTN_IMPL}, speech_attn_impl=${EFFECTIVE_SPEECH_ATTN_IMPL}, text_attn_impl=${EFFECTIVE_TEXT_ATTN_IMPL}"
echo "[INFO] Math backend: tf32_requested=${ENABLE_TF32}, tf32_effective=${EFFECTIVE_TF32_ENABLED}, matmul_precision=${MATMUL_PRECISION}"
echo "[INFO] Code revision: git_commit=${GIT_COMMIT_SHORT}, branch=${GIT_BRANCH}, dirty=${GIT_DIRTY}"
echo "[INFO] Compile/fixed-slice: enable_torch_compile=${ENABLE_TORCH_COMPILE}, torch_compile_mode=${TORCH_COMPILE_MODE}, torch_compile_dynamic=${TORCH_COMPILE_DYNAMIC}, enable_length_fixed_slice=${ENABLE_LENGTH_FIXED_SLICE}, fixed_slice_seconds=${FIXED_SLICE_SECONDS:-<none>}"
echo "[INFO] Length bucketing: enable_length_bucket=${ENABLE_LENGTH_BUCKET}, boundaries=${EFFECTIVE_LENGTH_BUCKET_BOUNDARIES_SEC:-<none>}"
echo "[INFO] Telemetry/stall-alert: enable_gpu_telemetry=${ENABLE_GPU_TELEMETRY}, gpu_telemetry_interval_sec=${GPU_TELEMETRY_INTERVAL_SEC}, enable_host_telemetry=${ENABLE_HOST_TELEMETRY}, host_telemetry_interval_sec=${HOST_TELEMETRY_INTERVAL_SEC}, stall_alert_ratio=${STALL_ALERT_RATIO}, max_wait_train_log_sec=${MAX_WAIT_TRAIN_LOG_SEC}"
echo "[INFO] Manifest durability: primary=${MANIFEST_PATH}, fallback=${MANIFEST_FALLBACK_PATH}, write_retries=${MANIFEST_WRITE_RETRIES}, retry_sleep_sec=${MANIFEST_WRITE_RETRY_SLEEP_SEC}"
echo "[INFO] Driver log path: ${DRIVER_LOG_PATH:-<disabled>}"

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
