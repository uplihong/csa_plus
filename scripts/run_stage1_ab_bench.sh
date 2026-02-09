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
STOP_ON_ERROR="${STOP_ON_ERROR:-0}"
CONDA_ENV="${CONDA_ENV:-}"

DATASET_ROOT="${DATASET_ROOT:-/code/data/LibriSpeech/LibriSpeech}"
DATASET_MANIFEST_PATH="${DATASET_MANIFEST_PATH:-}"
DATASET_USE_TRIM="${DATASET_USE_TRIM:-false}"
DATASET_OFFLINE_TRIMMED="${DATASET_OFFLINE_TRIMMED:-true}"
PRETRAINED_CKPT="${PRETRAINED_CKPT:-/code/data/weights/csa/ckpt_epoch_8.pth}"
SPEECH_MODEL_PATH="${SPEECH_MODEL_PATH:-/code/data/weights/wav2vec2-base}"
TEXT_MODEL_PATH="${TEXT_MODEL_PATH:-/code/data/weights/bert-base-uncased}"
ENABLE_CUDA_SYNC_TIMING="${ENABLE_CUDA_SYNC_TIMING:-false}"
TIMING_RANK_SCOPE="${TIMING_RANK_SCOPE:-rank0}" # rank0 | all

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

echo -e "mode\tgroup\trepeat\tstatus\tport\trun_dir\tlauncher_log\ttrain_log\tzero_stage\tmicro_batch\tworld_size\tglobal_batch\tdeterministic\tcudnn_benchmark\twall_clock_breakdown\tlog_every_steps\tvalidation_every_steps\tcheckpoint_every_steps\tnum_workers\tprefetch_factor\tdataset_manifest_path\tdataset_use_trim\tdataset_offline_trimmed\tenable_cuda_sync_timing\ttiming_rank_scope" > "${MANIFEST_PATH}"

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
require_timing_rank_scope "${TIMING_RANK_SCOPE}"

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
    "++train.enable_cuda_sync_timing=${ENABLE_CUDA_SYNC_TIMING}"
    "++train.timing_rank_scope=${TIMING_RANK_SCOPE}"
    "++train.pretrained_model_checkpoint=${PRETRAINED_CKPT}"
    "++dataset.root_dir=${DATASET_ROOT}"
    "++dataset.use_trim=${DATASET_USE_TRIM}"
    "++dataset.offline_trimmed=${DATASET_OFFLINE_TRIMMED}"
    "++model.speech_encoder.pretrained_path=${SPEECH_MODEL_PATH}"
    "++model.text_encoder.pretrained_path=${TEXT_MODEL_PATH}"
    "deepspeed_config_yaml.train_micro_batch_size_per_gpu=${micro_batch}"
    "deepspeed_config_yaml.zero_optimization.stage=${zero_stage}"
    "deepspeed_config_yaml.wall_clock_breakdown=${wall_clock_breakdown}"
    "deepspeed_config_yaml.bf16.enabled=true"
    "deepspeed_config_yaml.fp16.enabled=false"
  )
  if [[ -n "${DATASET_MANIFEST_PATH}" ]]; then
    cmd+=("++dataset.manifest_path=${DATASET_MANIFEST_PATH}")
  fi

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

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${mode_name}" "${group}" "${repeat}" "${status}" "${port}" "${run_dir}" "${launcher_log}" "${train_log}" \
    "${zero_stage}" "${micro_batch}" "${WORLD_SIZE}" "${global_batch}" \
    "${deterministic}" "${cudnn_benchmark}" "${wall_clock_breakdown}" \
    "${log_every}" "${val_every}" "${ckpt_every}" "${num_workers}" "${prefetch_factor}" \
    "${DATASET_MANIFEST_PATH}" "${DATASET_USE_TRIM}" "${DATASET_OFFLINE_TRIMMED}" "${ENABLE_CUDA_SYNC_TIMING}" "${TIMING_RANK_SCOPE}" >> "${MANIFEST_PATH}"

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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUMMARY_SCRIPT="${SCRIPT_DIR}/summarize_stage1_bench.py"
PYTHON_CMD=(python)
if [[ -n "${CONDA_ENV}" ]]; then
  PYTHON_CMD=(conda run -n "${CONDA_ENV}" python)
fi

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

echo "[INFO] Output root: ${OUTPUT_ROOT}"
echo "[INFO] Mode=${MODE}, REPEATS=${REPEATS}, MAX_STEPS=${MAX_STEPS}, INCLUDE=${INCLUDE}, WORLD_SIZE=${WORLD_SIZE}"
echo "[INFO] Dataset root=${DATASET_ROOT}, manifest=${DATASET_MANIFEST_PATH:-<none>}, use_trim=${DATASET_USE_TRIM}, offline_trimmed=${DATASET_OFFLINE_TRIMMED}"
echo "[INFO] Timing flags: enable_cuda_sync_timing=${ENABLE_CUDA_SYNC_TIMING}, timing_rank_scope=${TIMING_RANK_SCOPE}"

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
