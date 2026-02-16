# Stage1 Distributed SOP (Mainline)

This SOP is the production/mainline workflow for Stage1 throughput benchmarking.

Scope:
- Throughput-first, stable lane
- Single-node multi-GPU (2x/4x) with DeepSpeed
- Mainline only: `z0 + sdpa + precision auto`

Out of scope:
- Historical exploratory matrices
- `z1` throughput sweeps
- `flash_attention_2` / `fixed_slice` / length-bucket as default paths

## 1. Mainline Freeze (Decision)

Use this policy unless you have a specific troubleshooting target.

- `zero_stage=0`
- `PRECISION_MODE=auto`
  - cc >= 8.0 (4090/H100/A100): effective `bf16`
  - cc < 8.0 (V100): effective `fp16`
- `ATTN_IMPL=auto` (effective `sdpa` on 4090/H100)
- `MODEL_LOAD_DTYPE=auto` (follows effective precision)
- `ENABLE_TF32=true`
- `MATMUL_PRECISION=high`
- `ENABLE_TORCH_COMPILE=false`
- `ENABLE_LENGTH_FIXED_SLICE=false`
- `ENABLE_LENGTH_BUCKET=false`

Mainline intent:
- Maximize stable throughput with reproducible operations.
- Avoid spending benchmark budget on low-confidence branches.

## 2. Prerequisites

## 2.1 Dataset (offline 16k trimmed recommended)

```bash
python scripts/preprocess_librispeech_16k.py \
  --input-root data/LibriSpeech/LibriSpeech \
  --output-root data/LibriSpeech/LibriSpeech_16k_trim \
  --target-sr 16000 \
  --trim \
  --manifest-path data/LibriSpeech/LibriSpeech_16k_trim/manifest_16k_trim.tsv \
  --workers 16 \
  --log-every 1000
```

## 2.2 Runtime expectations

- `deepspeed` executable available
- NCCL + CUDA driver healthy
- no heavy concurrent CPU tasks during run
- enough disk space for `OUTPUT_ROOT`
- if `deepspeed` is not in current shell `PATH`, set `CONDA_ENV` (e.g. `CONDA_ENV=csref_2`)

## 3. Mainline Command Templates

All templates use:
- rank0 timing for lower overhead (`TIMING_RANK_SCOPE=rank0`)
- ETA logging enabled with low-overhead mode (`ETA_DISTRIBUTED_MODE=rank0`)
- GPU telemetry enabled
- host telemetry optional (`ENABLE_HOST_TELEMETRY`)

### 3.1 2x4090 convergence sweep

```bash
TS=$(date +%Y%m%d_%H%M%S)
OUT=outputs/bench_2x4090_z0_mainline_${TS}
mkdir -p "${OUT}"

MODE=sweep \
INCLUDE=localhost:0,1 \
REPEATS=2 \
STOP_ON_ERROR=0 \
MAX_STEPS=1200 \
SWEEP_ZERO_STAGES=0 \
SWEEP_MICRO_BATCHES=160,192 \
SWEEP_NUM_WORKERS_LIST=6,8 \
SWEEP_PREFETCH_LIST=4 \
SWEEP_LOG_EVERY=50 \
TAIL_TIMING_POINTS=10 \
HEARTBEAT_EVERY_SEC=30 \
RUN_TIMEOUT_SEC=2400 \
FAILURE_DUMP_TAIL=true \
FAIL_TAIL_LINES=120 \
RESUME_RUNS=true \
SWEEP_VALIDATION_EVERY=1000000 \
SWEEP_CHECKPOINT_EVERY=1000000 \
CONDA_ENV=csref_2 \
DATASET_ROOT=data/LibriSpeech/LibriSpeech_16k_trim \
DATASET_MANIFEST_PATH=data/LibriSpeech/LibriSpeech_16k_trim/manifest_16k_trim.tsv \
DATASET_USE_TRIM=false \
DATASET_OFFLINE_TRIMMED=true \
ENABLE_CUDA_SYNC_TIMING=false \
TIMING_RANK_SCOPE=rank0 \
ENABLE_ETA_LOGGING=true \
ETA_DISTRIBUTED_MODE=rank0 \
ETA_MIN_SAMPLES=10 \
PRECISION_MODE=auto \
ATTN_IMPL=auto \
MODEL_LOAD_DTYPE=auto \
ENABLE_TF32=true \
MATMUL_PRECISION=high \
ENABLE_TORCH_COMPILE=false \
ENABLE_LENGTH_FIXED_SLICE=false \
ENABLE_LENGTH_BUCKET=false \
ENABLE_GPU_TELEMETRY=true \
GPU_TELEMETRY_INTERVAL_SEC=1 \
ENABLE_HOST_TELEMETRY=true \
HOST_TELEMETRY_INTERVAL_SEC=2 \
OUTPUT_ROOT="${OUT}" \
./scripts/run_stage1_ab_bench.sh 2>&1 | tee "${OUT}/driver.log"
```

### 3.2 4x4090 convergence sweep

```bash
TS=$(date +%Y%m%d_%H%M%S)
OUT=outputs/bench_4x4090_z0_mainline_${TS}
mkdir -p "${OUT}"

MODE=sweep \
INCLUDE=localhost:0,1,2,3 \
REPEATS=2 \
STOP_ON_ERROR=0 \
MAX_STEPS=1200 \
SWEEP_ZERO_STAGES=0 \
SWEEP_MICRO_BATCHES=160,192 \
SWEEP_NUM_WORKERS_LIST=6,8 \
SWEEP_PREFETCH_LIST=4 \
SWEEP_LOG_EVERY=50 \
TAIL_TIMING_POINTS=10 \
HEARTBEAT_EVERY_SEC=30 \
RUN_TIMEOUT_SEC=2400 \
FAILURE_DUMP_TAIL=true \
FAIL_TAIL_LINES=120 \
RESUME_RUNS=true \
SWEEP_VALIDATION_EVERY=1000000 \
SWEEP_CHECKPOINT_EVERY=1000000 \
CONDA_ENV=csref_2 \
DATASET_ROOT=data/LibriSpeech/LibriSpeech_16k_trim \
DATASET_MANIFEST_PATH=data/LibriSpeech/LibriSpeech_16k_trim/manifest_16k_trim.tsv \
DATASET_USE_TRIM=false \
DATASET_OFFLINE_TRIMMED=true \
ENABLE_CUDA_SYNC_TIMING=false \
TIMING_RANK_SCOPE=rank0 \
ENABLE_ETA_LOGGING=true \
ETA_DISTRIBUTED_MODE=rank0 \
ETA_MIN_SAMPLES=10 \
PRECISION_MODE=auto \
ATTN_IMPL=auto \
MODEL_LOAD_DTYPE=auto \
ENABLE_TF32=true \
MATMUL_PRECISION=high \
ENABLE_TORCH_COMPILE=false \
ENABLE_LENGTH_FIXED_SLICE=false \
ENABLE_LENGTH_BUCKET=false \
ENABLE_GPU_TELEMETRY=true \
GPU_TELEMETRY_INTERVAL_SEC=1 \
ENABLE_HOST_TELEMETRY=true \
HOST_TELEMETRY_INTERVAL_SEC=2 \
OUTPUT_ROOT="${OUT}" \
./scripts/run_stage1_ab_bench.sh 2>&1 | tee "${OUT}/driver.log"
```

### 3.3 Fast smoke run (10-20 min)

Use before large sweeps or after branch merges.

```bash
TS=$(date +%Y%m%d_%H%M%S)
OUT=outputs/smoke_stage1_${TS}
mkdir -p "${OUT}"

MODE=sweep \
INCLUDE=localhost:0,1 \
REPEATS=1 \
STOP_ON_ERROR=1 \
MAX_STEPS=300 \
SWEEP_ZERO_STAGES=0 \
SWEEP_MICRO_BATCHES=160 \
SWEEP_NUM_WORKERS_LIST=8 \
SWEEP_PREFETCH_LIST=4 \
SWEEP_LOG_EVERY=50 \
TAIL_TIMING_POINTS=10 \
HEARTBEAT_EVERY_SEC=30 \
RUN_TIMEOUT_SEC=1800 \
CONDA_ENV=csref_2 \
DATASET_ROOT=data/LibriSpeech/LibriSpeech_16k_trim \
DATASET_MANIFEST_PATH=data/LibriSpeech/LibriSpeech_16k_trim/manifest_16k_trim.tsv \
DATASET_USE_TRIM=false \
DATASET_OFFLINE_TRIMMED=true \
TIMING_RANK_SCOPE=rank0 \
ENABLE_ETA_LOGGING=true \
ETA_DISTRIBUTED_MODE=rank0 \
ETA_MIN_SAMPLES=10 \
PRECISION_MODE=auto \
ATTN_IMPL=auto \
MODEL_LOAD_DTYPE=auto \
ENABLE_TF32=true \
MATMUL_PRECISION=high \
ENABLE_TORCH_COMPILE=false \
ENABLE_LENGTH_FIXED_SLICE=false \
ENABLE_LENGTH_BUCKET=false \
ENABLE_GPU_TELEMETRY=true \
ENABLE_HOST_TELEMETRY=true \
OUTPUT_ROOT="${OUT}" \
./scripts/run_stage1_ab_bench.sh 2>&1 | tee "${OUT}/driver.log"
```

### 3.4 Low-overhead distributed ETA (recommended)

Default (lowest overhead):

- `ENABLE_ETA_LOGGING=true`
- `ETA_DISTRIBUTED_MODE=rank0`
- `ETA_MIN_SAMPLES=10`

Conservative multi-rank estimate:

- switch `ETA_DISTRIBUTED_MODE=global_max`
- this adds low-frequency scalar `all_reduce(max)` only at log steps

Single-machine compatibility fallback (when distributed NCCL P2P is unstable):

- add `NCCL_P2P_DISABLE=1` for the benchmark command
- keep this as host-specific override, not a default policy

## 4. Output Artifacts

Each run/sweep writes to `OUTPUT_ROOT`.

Core files:
- `run_manifest.tsv`: run-level status and last metrics (including `last_eta_*`)
- `timing_points.csv`: parsed timing windows
- `eta_points.csv`: parsed ETA points from `TimingETA`
- `per_run_metrics.csv`: aggregated per-run metrics
- `group_summary.csv`: group-level aggregation and ranking fields
- `summary.md` / `summary.json`: human/machine summaries
- `ranked_groups.csv`: sortable ranking output

Telemetry (if enabled):
- per-run `gpu_telemetry.csv`
- per-run `host_telemetry.csv`

## 5. Reading Results

Primary KPIs:
- Throughput: `tail_samples_per_sec_mean`
- Iter latency: `tail_mean_iter_p50_ms_mean`

Stability KPIs:
- `iter_p90_over_p50_mean`
- `step_p90_over_p50_mean` (from per-run/timing interpretation)
- `unstable_run_ratio`

Heuristics:
- prefer groups with lower p50 and controlled p90/p50
- treat very high p90/p50 as straggler/sync issue, not pure compute limit
- compare 2x vs 4x using both total throughput and per-GPU efficiency

## 6. Troubleshooting

### 6.1 `EADDRINUSE` (master port in use)

- Root cause: port collision from stale/parallel jobs.
- Action:
  - stop stale jobs
  - rerun with a different `MASTER_PORT`/auto-selected port path

### 6.2 `train.log` not appearing + heartbeat waiting

- Common causes:
  - launcher failed before training started
  - filesystem/storage issue
- Action:
  - inspect `launcher.log` first
  - verify output mount is writable and healthy

### 6.3 telemetry file empty warning

- Confirm collector availability (`nvidia-smi`, `/proc` for host telemetry).
- If job ended too early, empty telemetry can be expected.

### 6.4 platform I/O instability (`Input/output error`)

- Treat as infra issue first.
- Re-run on healthy node/storage before comparing code-level performance.

### 6.5 `deepspeed: No such file or directory`

- Root cause: `deepspeed` not in current shell path.
- Action:
  - set `CONDA_ENV=csref_2` (or your training env)
  - rerun command without changing benchmark configs

### 6.6 distributed stall/hang on this host (NCCL P2P path)

- Symptom:
  - multi-GPU job stalls early or heartbeat cannot observe normal step progress.
- Action:
  - rerun with `NCCL_P2P_DISABLE=1`
  - keep this setting host-scoped; do not force as global default

## 7. Optional Diagnostic Tool

`quick_leak_diagnose.py` is retained as optional diagnosis only.
It is not part of normal throughput benchmarking workflow.

Example:

```bash
python scripts/quick_leak_diagnose.py \
  outputs/bench_2x4090_z0_mainline_xxx/sweep_z0_mb160_nw8_pf4_r1 \
  outputs/bench_2x4090_z0_mainline_xxx/sweep_z0_mb192_nw8_pf4_r1
```

## 8. What Is Not Mainline

These options remain in code but are off the default lane:
- `ENABLE_LENGTH_FIXED_SLICE=true`
- `ENABLE_LENGTH_BUCKET=true`
- `ENABLE_TORCH_COMPILE=true`
- routine throughput sweeps with `zero_stage=1`
- defaulting to `flash_attention_2`

Use them only for isolated A/B diagnostics, then report separately.

## 9. Historical Experiments

Historical Phase A/B/C command sets and exploratory branches are archived in:
- `docs/archive_stage1_experiments.md`
