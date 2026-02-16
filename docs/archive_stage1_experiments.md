# Stage1 Historical Experiments Archive

This document archives historical Stage1 experiments that are no longer the default mainline workflow.

Current mainline SOP lives at:
- `docs/stage1_distributed_sop.md`

Use this file for traceability and postmortem context only.

## 1. Historical Timeline (Condensed)

- Phase A: broad baseline/matrix sweeps across V100/4090 variants.
- Phase B: stability-focused sentinel runs with rank-level timing and telemetry.
- Phase C: targeted A/B on `fixed_slice` and length bucketing.
- Revalidation rounds: repeated 2x/4x 4090 runs to separate code effects from platform effects.

## 2. Historical Conclusions (Non-Mainline)

1. `z1` often improved stability metrics but usually hurt throughput significantly in 4090 PCIe environments.
2. `flash_attention_2` dtype mismatch issue was fixed; however, throughput gains were inconsistent in this project context.
3. `fixed_slice` and `length_bucket` could increase raw throughput in some runs, but tail behavior and variability often worsened.
4. Platform factors (node topology, storage, transient host load) produced larger variance than several operator-level tweaks.

These points are retained for context; they are not default policy.

## 3. Historical Command Snippets

## 3.1 Large sweep (legacy example)

```bash
MODE=sweep \
INCLUDE=localhost:0,1 \
REPEATS=1 \
STOP_ON_ERROR=0 \
MAX_STEPS=600 \
SWEEP_ZERO_STAGES=0,1 \
SWEEP_MICRO_BATCHES=128,160,192 \
SWEEP_NUM_WORKERS_LIST=4,6,8 \
SWEEP_PREFETCH_LIST=2,4 \
SWEEP_LOG_EVERY=10 \
TAIL_TIMING_POINTS=10 \
HEARTBEAT_EVERY_SEC=30 \
RUN_TIMEOUT_SEC=2400 \
RESUME_RUNS=true \
PRECISION_MODE=auto \
ATTN_IMPL=auto \
ENABLE_TORCH_COMPILE=false \
ENABLE_LENGTH_FIXED_SLICE=false \
OUTPUT_ROOT=outputs/bench_legacy_matrix \
./scripts/run_stage1_ab_bench.sh
```

## 3.2 Phase B sentinel style (legacy example)

```bash
MODE=sweep \
INCLUDE=localhost:0,1 \
REPEATS=3 \
MAX_STEPS=2000 \
SWEEP_ZERO_STAGES=0,1 \
SWEEP_MICRO_BATCHES=128,160 \
SWEEP_NUM_WORKERS_LIST=6 \
SWEEP_PREFETCH_LIST=2,4 \
TIMING_RANK_SCOPE=all \
ENABLE_GPU_TELEMETRY=true \
ENABLE_HOST_TELEMETRY=true \
OUTPUT_ROOT=outputs/bench_phaseb_legacy \
./scripts/run_stage1_ab_bench.sh
```

## 3.3 Fixed-slice and length-bucket A/B (legacy example)

Baseline:

```bash
MODE=sweep \
INCLUDE=localhost:0,1 \
REPEATS=2 \
MAX_STEPS=2000 \
SWEEP_ZERO_STAGES=0 \
SWEEP_MICRO_BATCHES=192 \
SWEEP_NUM_WORKERS_LIST=6 \
SWEEP_PREFETCH_LIST=4 \
ENABLE_LENGTH_FIXED_SLICE=false \
ENABLE_LENGTH_BUCKET=false \
TIMING_RANK_SCOPE=all \
OUTPUT_ROOT=outputs/bench_phasec_baseline \
./scripts/run_stage1_ab_bench.sh
```

Fixed slice:

```bash
MODE=sweep \
INCLUDE=localhost:0,1 \
REPEATS=2 \
MAX_STEPS=2000 \
SWEEP_ZERO_STAGES=0 \
SWEEP_MICRO_BATCHES=192 \
SWEEP_NUM_WORKERS_LIST=6 \
SWEEP_PREFETCH_LIST=4 \
ENABLE_LENGTH_FIXED_SLICE=true \
FIXED_SLICE_SECONDS=2.0 \
ENABLE_LENGTH_BUCKET=false \
TIMING_RANK_SCOPE=all \
OUTPUT_ROOT=outputs/bench_phasec_fixed_slice \
./scripts/run_stage1_ab_bench.sh
```

Length bucket:

```bash
MODE=sweep \
INCLUDE=localhost:0,1 \
REPEATS=2 \
MAX_STEPS=2000 \
SWEEP_ZERO_STAGES=0 \
SWEEP_MICRO_BATCHES=192 \
SWEEP_NUM_WORKERS_LIST=6 \
SWEEP_PREFETCH_LIST=4 \
ENABLE_LENGTH_FIXED_SLICE=false \
ENABLE_LENGTH_BUCKET=true \
LENGTH_BUCKET_BOUNDARIES_SEC=1.0,1.5,2.0,2.5,3.0 \
TIMING_RANK_SCOPE=all \
OUTPUT_ROOT=outputs/bench_phasec_length_bucket \
./scripts/run_stage1_ab_bench.sh
```

## 3.4 Experimental min-matrix wrapper (retired)

Legacy min-matrix wrapper scripts and their analyzer/checker helpers were retired from mainline.
Equivalent analysis can be done with:
- `scripts/run_stage1_ab_bench.sh`
- `scripts/summarize_stage1_bench.py`
- optional `scripts/quick_leak_diagnose.py`

## 4. Retired / Non-Mainline Paths

The following are retained in code/config for optional diagnostics, but not mainline:

- `zero_stage=1` throughput sweeps
- default `flash_attention_2`
- `ENABLE_TORCH_COMPILE=true` as default
- `ENABLE_LENGTH_FIXED_SLICE=true` as default
- `ENABLE_LENGTH_BUCKET=true` as default

## 5. How to Reproduce Legacy Findings

1. Check historical outputs under `outputs/` directories.
2. Match git commit from run metadata (`git_commit_short` in manifest/summary).
3. Re-run with the exact env block captured in historical `driver.log`.
4. Compare with current mainline only after confirming platform parity.

## 6. Why This Archive Exists

- Preserve historical traceability for prior decisions.
- Keep mainline SOP concise and operational.
- Avoid mixing exploratory branches with default production workflow.
