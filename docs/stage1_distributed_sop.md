# Stage1 Distributed SOP

## 1. Offline Data Preparation (recommended)

```bash
python scripts/preprocess_librispeech_16k.py \
  --input-root data/LibriSpeech/LibriSpeech \
  --output-root data/LibriSpeech/LibriSpeech_16k_trim \
  --target-sr 16000 \
  --trim \
  --manifest-path data/LibriSpeech/LibriSpeech_16k_trim/manifest_16k_trim.tsv \
  --workers 49 \
  --log-every 1000
```

Set runtime env:

```bash
export LIBRISPEECH_MANIFEST_PATH=./data/LibriSpeech/LibriSpeech_16k_trim/manifest_16k_trim.tsv
```

## 2. Precision/Attention Policy (cross-platform)

| Platform | `PRECISION_MODE=auto` result | Recommended `ATTN_IMPL` |
|---|---|---|
| V100 | `fp16` (cc < 8.0) | `eager` |
| RTX 4090 | `bf16` (cc >= 8.0) | `sdpa` |
| H100 | `bf16` (cc >= 8.0) | `sdpa` |

The benchmark script records requested/effective precision + attention mode in `run_manifest.tsv`.

## 3. Sweep (complete matrix + partial summary on interrupt)

### 2x4090
```bash
MODE=sweep \
INCLUDE=localhost:0,1,2,3 \
REPEATS=2 \
STOP_ON_ERROR=0 \
MAX_STEPS=1000 \
SWEEP_ZERO_STAGES=0,1 \
SWEEP_MICRO_BATCHES=128,160,192 \
SWEEP_NUM_WORKERS_LIST=4,6,8 \
SWEEP_PREFETCH_LIST=2,4 \
SWEEP_LOG_EVERY=50 \
TAIL_TIMING_POINTS=10 \
HEARTBEAT_EVERY_SEC=30 \
RUN_TIMEOUT_SEC=2400 \
FAILURE_DUMP_TAIL=true \
FAIL_TAIL_LINES=80 \
RESUME_RUNS=true \
SWEEP_VALIDATION_EVERY=1000000 \
SWEEP_CHECKPOINT_EVERY=1000000 \
DATASET_ROOT=data/LibriSpeech/LibriSpeech_16k_trim \
DATASET_MANIFEST_PATH=data/LibriSpeech/LibriSpeech_16k_trim/manifest_16k_trim.tsv \
DATASET_USE_TRIM=false \
DATASET_OFFLINE_TRIMMED=true \
ENABLE_CUDA_SYNC_TIMING=false \
TIMING_RANK_SCOPE=rank0 \
PRECISION_MODE=auto \
ATTN_IMPL=auto \
ENABLE_TORCH_COMPILE=false \
TORCH_COMPILE_MODE=max-autotune \
TORCH_COMPILE_DYNAMIC=true \
ENABLE_LENGTH_FIXED_SLICE=false \
OUTPUT_ROOT=outputs/bench_4090_sweep_stage1_v4 \
./scripts/run_stage1_ab_bench.sh 2>&1 | tee outputs/bench_4090_sweep_stage1_v4/driver.log
```

### 2xv100
```bash
MODE=sweep \
INCLUDE=localhost:0,4 \
REPEATS=1 \
STOP_ON_ERROR=0 \
MAX_STEPS=300 \
SWEEP_ZERO_STAGES=0,1 \
SWEEP_MICRO_BATCHES=128,160,192 \
SWEEP_NUM_WORKERS_LIST=4,6,8 \
SWEEP_PREFETCH_LIST=2,4 \
SWEEP_LOG_EVERY=10 \
TAIL_TIMING_POINTS=10 \
HEARTBEAT_EVERY_SEC=30 \
RUN_TIMEOUT_SEC=2400 \
FAILURE_DUMP_TAIL=true \
FAIL_TAIL_LINES=80 \
RESUME_RUNS=true \
SWEEP_VALIDATION_EVERY=1000000 \
SWEEP_CHECKPOINT_EVERY=1000000 \
DATASET_ROOT=data/LibriSpeech/LibriSpeech_16k_trim \
DATASET_MANIFEST_PATH=data/LibriSpeech/LibriSpeech_16k_trim/manifest_16k_trim.tsv \
DATASET_USE_TRIM=false \
DATASET_OFFLINE_TRIMMED=true \
ENABLE_CUDA_SYNC_TIMING=false \
TIMING_RANK_SCOPE=rank0 \
PRECISION_MODE=auto \
ATTN_IMPL=auto \
ENABLE_TORCH_COMPILE=false \
TORCH_COMPILE_MODE=max-autotune \
TORCH_COMPILE_DYNAMIC=true \
ENABLE_LENGTH_FIXED_SLICE=false \
OUTPUT_ROOT=outputs/bench_4090_sweep_stage1_v4 \
./scripts/run_stage1_ab_bench.sh 2>&1 | tee outputs/bench_4090_sweep_stage1_v4/driver.log
```

Notes:

- `HEARTBEAT_EVERY_SEC=30` prints latest observed step periodically to avoid "silent hanging" perception.
- `RUN_TIMEOUT_SEC=2400` marks a run as failed if it exceeds 40 minutes (tune up/down by your platform quota).
- `RESUME_RUNS=true` keeps `run_manifest.tsv` and skips already-successful groups when rerunning after interruption.
- On failure, script now records `exit_code`, `duration_sec`, `last_step` in `run_manifest.tsv` and prints launcher/train tail automatically.
- `ENABLE_LENGTH_FIXED_SLICE=false` by default. Enable it only for throughput-only A/B tests, e.g. `ENABLE_LENGTH_FIXED_SLICE=true FIXED_SLICE_SECONDS=2.0`.

Outputs:

- `outputs/.../run_manifest.tsv`
- `outputs/.../per_run_metrics.csv`
- `outputs/.../group_summary.csv`
- `outputs/.../ranked_groups.csv`
- `outputs/.../best_config.json`
- `outputs/.../summary.md`

### Recommended Phase Schedule (throughput-first)

- Phase A (pilot): `REPEATS=1`, `MAX_STEPS=300`; sweep `zero_stage in {0,1}`, `micro_batch in {128,160,192}`, `num_workers in {4,6}`, `prefetch in {2,4}`.
- Phase B (retest top2): keep only best two groups from Phase A, run `REPEATS=2`, `MAX_STEPS=1000`.
- Phase C (optional, best group only): test `ENABLE_TORCH_COMPILE=true` and/or `ENABLE_LENGTH_FIXED_SLICE=true FIXED_SLICE_SECONDS=2.0`.

### Acceptance Criteria (throughput-first)

- Primary metric: `tail_mean_iter_p50_ms` (smaller is better).
- Secondary metric: `tail_samples_per_sec_mean` (larger is better).
- Suggested target: V100 speedup >= 20% vs BF16 baseline; H100/4090 speedup >= 8% with comparable loss trend.

## 4. Rank Straggler Diagnosis Run

```bash
MODE=sweep \
INCLUDE=localhost:0,1 \
REPEATS=1 \
MAX_STEPS=1000 \
SWEEP_ZERO_STAGES=0,1 \
SWEEP_MICRO_BATCHES=160 \
SWEEP_NUM_WORKERS_LIST=6 \
SWEEP_PREFETCH_LIST=4 \
ENABLE_CUDA_SYNC_TIMING=true \
TIMING_RANK_SCOPE=all \
SWEEP_LOG_EVERY=50 \
SWEEP_VALIDATION_EVERY=1000000 \
SWEEP_CHECKPOINT_EVERY=1000000 \
DATASET_ROOT=/code/data/LibriSpeech/LibriSpeech_16k_trim \
DATASET_MANIFEST_PATH=/code/data/LibriSpeech/LibriSpeech_16k_trim/manifest_16k_trim.tsv \
DATASET_USE_TRIM=false \
DATASET_OFFLINE_TRIMMED=true \
OUTPUT_ROOT=outputs/bench_4090_rank_diag \
./scripts/run_stage1_ab_bench.sh
```

Inspect `train.log` `TimingRank` lines for rank imbalance.

## 5. Final Training (single machine)

```bash
deepspeed --include localhost:0,1 train.py \
  +experiment=limit_longest_1-3_stage1_bf16 \
  '++dataset.root_dir=/code/data/LibriSpeech/LibriSpeech_16k_trim' \
  '++dataset.manifest_path=/code/data/LibriSpeech_16k_trim/LibriSpeech/manifest_16k_trim.tsv' \
  '++dataset.use_trim=false' \
  '++dataset.offline_trimmed=true' \
  'deepspeed_config_yaml.zero_optimization.stage=1' \
  'deepspeed_config_yaml.train_micro_batch_size_per_gpu=160' \
  '++train.data.num_workers=6' \
  '++train.data.prefetch_factor=4' \
  '++train.enable_cuda_sync_timing=false' \
  '++train.timing_rank_scope=rank0'
```

## 6. Final Training (multi-node template)

```bash
deepspeed --hostfile /etc/deepspeed/hostfile train.py \
  +experiment=limit_longest_1-3_stage1_bf16 \
  '++dataset.root_dir=/code/data/LibriSpeech/LibriSpeech_16k_trim' \
  '++dataset.manifest_path=/code/data/LibriSpeech/LibriSpeech_16k_trim/manifest_16k_trim.tsv' \
  '++dataset.use_trim=false' \
  '++dataset.offline_trimmed=true' \
  'deepspeed_config_yaml.zero_optimization.stage=1' \
  'deepspeed_config_yaml.train_micro_batch_size_per_gpu=160' \
  '++train.data.num_workers=6' \
  '++train.data.prefetch_factor=4' \
  '++train.enable_cuda_sync_timing=false' \
  '++train.timing_rank_scope=rank0'
```
