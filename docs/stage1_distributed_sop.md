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

Throughput-first defaults:
- 4090/H100: prefer `zero_stage=0`; use `zero_stage=1` only when memory pressure forces it.
- `MODEL_LOAD_DTYPE=auto` follows effective precision (`bf16` or `fp16`), which avoids FlashAttention2 dtype mismatch.
- `SPEECH_ATTN_IMPL` and `TEXT_ATTN_IMPL` can be set independently; if omitted they inherit `ATTN_IMPL`.
- `ENABLE_TF32=true` is only effective on cc >= 8.0 GPUs.

## 3. Sweep (complete matrix + partial summary on interrupt)

### 3.1 2x4090
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
ENABLE_GPU_TELEMETRY=true \
GPU_TELEMETRY_INTERVAL_SEC=2 \
STALL_ALERT_RATIO=2.0 \
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
OUTPUT_ROOT=outputs/bench_4090x2_24h_sweep_stage1_v6_flash-attn \
DRIVER_LOG_PATH=outputs/bench_4090x2_24h_sweep_stage1_v6_flash-attn/driver.log \
./scripts/run_stage1_ab_bench.sh
```

#### 12h
```bash
MODE=sweep \
INCLUDE=localhost:0,1 \
REPEATS=1 \
STOP_ON_ERROR=0 \
MAX_STEPS=500 \
SWEEP_ZERO_STAGES=0,1 \
SWEEP_MICRO_BATCHES=128,160,192,224 \
SWEEP_NUM_WORKERS_LIST=4,6,8 \
SWEEP_PREFETCH_LIST=2,4 \
SWEEP_LOG_EVERY=10 \
TAIL_TIMING_POINTS=10 \
HEARTBEAT_EVERY_SEC=30 \
ENABLE_GPU_TELEMETRY=true \
GPU_TELEMETRY_INTERVAL_SEC=2 \
STALL_ALERT_RATIO=2.0 \
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
OUTPUT_ROOT=outputs/bench_4090x2_12h_sweep_stage1_v5 \
DRIVER_LOG_PATH=outputs/bench_4090x2_12h_sweep_stage1_v5/driver.log \
./scripts/run_stage1_ab_bench.sh
```

### 3.2 4x4090
```bash
MODE=sweep \
INCLUDE=localhost:0,1,2,3 \
REPEATS=2 \
STOP_ON_ERROR=0 \
MAX_STEPS=1000 \
SWEEP_ZERO_STAGES=0,1 \
SWEEP_MICRO_BATCHES=128,160,192,224,256 \
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
OUTPUT_ROOT=outputs/bench_4090x4_sweep_stage1_v5 \
DRIVER_LOG_PATH=outputs/bench_4090x4_sweep_stage1_v5/driver.log \
./scripts/run_stage1_ab_bench.sh
```

### 3.3 2xv100
```bash
MODE=sweep \
INCLUDE=localhost:0,4 \
REPEATS=1 \
STOP_ON_ERROR=0 \
MAX_STEPS=1000 \
SWEEP_ZERO_STAGES=0,1 \
SWEEP_MICRO_BATCHES=128,160,192,224 \
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
OUTPUT_ROOT=outputs/bench_v100x2_sweep_stage1_v5 \
DRIVER_LOG_PATH=outputs/bench_v100x2_sweep_stage1_v5/driver.log \
./scripts/run_stage1_ab_bench.sh
```

### 3.4 2x4090

> Phase B：稳定性优先的 4090 诊断（1-2 天）
固定两个哨兵组：z0_mb128_nw6_pf4 与 z1_mb160_nw6_pf2，每组 3 次。
强制 TIMING_RANK_SCOPE=all，采集 rank 级 timing 与 gpu_telemetry.csv。
输出判定：若 iter_p90/p50 持续 >2 或双卡 util 反相关明显，则判定为平台抖动主导，不进入算子优化阶段。

> Phase C：参数收敛与生产候选（1 天）
V100 候选：吞吐优先 z0_mb160_nw6_pf2，时延优先 z0_mb128_nw6_pf2。
4090 候选：吞吐优先先用 z0_mb192_nw4_pf2（24h 完成集中最优 samples/s），时延优先 z0_mb128_nw6_pf4。
若 12h 平台可复现高功率稳定态，再追加 mb224 作为吞吐特化候选。

> Phase D：算子级增益（在 B 稳定后再做，1-2 天）
先 A/B flash_attention_2（仅 4090/H100）；不可用自动回退 sdpa。
再 A/B torch.compile（仅对稳定平台），保留失败回退。
最后单独评估 fixed_slice_seconds=2.0 作为吞吐特化配置，不混入默认基线。

#### 3.4.1 B

先统一设置一组公共参数（按 2x4090）：

```bash
export INCLUDE=localhost:0,1
export STOP_ON_ERROR=0
export FAILURE_DUMP_TAIL=true
export FAIL_TAIL_LINES=120
export RESUME_RUNS=true
export HEARTBEAT_EVERY_SEC=30
export RUN_TIMEOUT_SEC=2400
export TAIL_TIMING_POINTS=10

export DATASET_ROOT=data/LibriSpeech/LibriSpeech_16k_trim
export DATASET_MANIFEST_PATH=data/LibriSpeech/LibriSpeech_16k_trim/manifest_16k_trim.tsv
export DATASET_USE_TRIM=false
export DATASET_OFFLINE_TRIMMED=true

export PRECISION_MODE=auto
export ATTN_IMPL=auto
export MODEL_LOAD_DTYPE=auto
export SPEECH_ATTN_IMPL=auto
export TEXT_ATTN_IMPL=auto
export ENABLE_TF32=true
export MATMUL_PRECISION=high
export ENABLE_CUDA_SYNC_TIMING=false
export ENABLE_GPU_TELEMETRY=true
export GPU_TELEMETRY_INTERVAL_SEC=2
export STALL_ALERT_RATIO=2.0

export SWEEP_LOG_EVERY=50
export SWEEP_VALIDATION_EVERY=1000000
export SWEEP_CHECKPOINT_EVERY=1000000

export ENABLE_TORCH_COMPILE=false
export TORCH_COMPILE_MODE=max-autotune
export TORCH_COMPILE_DYNAMIC=true
export ENABLE_LENGTH_FIXED_SLICE=false
```

哨兵组1: z0_mb128_nw6_pf4, repeat=3

```bash
MODE=sweep REPEATS=3 MAX_STEPS=1000 TIMING_RANK_SCOPE=all \
SWEEP_ZERO_STAGES=0 SWEEP_MICRO_BATCHES=128 SWEEP_NUM_WORKERS_LIST=6 SWEEP_PREFETCH_LIST=4 \
OUTPUT_ROOT=outputs/bench_4090_phaseB_diag \
DRIVER_LOG_PATH=outputs/bench_4090_phaseB_diag/driver.log \
./scripts/run_stage1_ab_bench.sh
```

哨兵组2: z1_mb160_nw6_pf2, repeat=3（同一个 OUTPUT_ROOT，直接追加）
```bash
MODE=sweep REPEATS=3 MAX_STEPS=1000 TIMING_RANK_SCOPE=all \
SWEEP_ZERO_STAGES=1 SWEEP_MICRO_BATCHES=160 SWEEP_NUM_WORKERS_LIST=6 SWEEP_PREFETCH_LIST=2 \
OUTPUT_ROOT=outputs/bench_4090_phaseB_diag \
DRIVER_LOG_PATH=outputs/bench_4090_phaseB_diag/driver.log \
./scripts/run_stage1_ab_bench.sh
```

一键跑完整 4090 Phase-B 最小矩阵（含自动验收）：
```bash
OUTPUT_ROOT=outputs/bench_4090_phaseB_diag \
INCLUDE=localhost:0,1 \
./scripts/run_phaseb_4090_stability.sh
```

单独执行验收脚本（可复用到任意目录）：
```bash
./scripts/check_phaseb_stability.py \
  --output-root outputs/bench_4090_phaseB_diag \
  --groups sweep_z0_mb128_nw6_pf4,sweep_z1_mb160_nw6_pf2 \
  --min-repeats 3 \
  --max-iter-ratio-median 1.5 \
  --max-iter-ratio-max 2.5 \
  --max-iter-p50-cv 0.05
```

#### 3.4.1.1 跑一次长时 leak 诊断（4090，z0_mb128_nw6_pf4，MAX_STEPS=10000，含主机内存采样）

```bash
OUT=outputs/leak_4090_z0_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"

(
  echo "timestamp,epoch_sec,mem_total_kib,mem_available_kib,swap_total_kib,swap_free_kib,load1,load5,load15,train_proc_count,train_rss_kib,train_vsz_kib" > "$OUT/host_telemetry.csv"
  while true; do
    ts=$(date -Is); epoch=$(date +%s)
    eval "$(awk '
      /^MemTotal:/ {print "mt="$2}
      /^MemAvailable:/ {print "ma="$2}
      /^SwapTotal:/ {print "st="$2}
      /^SwapFree:/ {print "sf="$2}
      END {print "mt="mt"\nma="ma"\nst="st"\nsf="sf}
    ' /proc/meminfo)"
    read l1 l5 l15 _ < /proc/loadavg
    read pc pr pv <<< "$(ps -eo pid=,rss=,vsz=,args= | awk 'BEGIN{c=0;r=0;v=0} {pid=$1;rss=$2;vsz=$3;$1=$2=$3=""; sub(/^ +/,"",$0); if($0 ~ /(deepspeed|train.py)/){c++; r+=rss; v+=vsz}} END{printf "%d %d %d", c,r,v}')"
    echo "$ts,$epoch,$mt,$ma,$st,$sf,$l1,$l5,$l15,$pc,$pr,$pv" >> "$OUT/host_telemetry.csv"
    sleep 5
  done
) &
MON=$!
trap 'kill $MON 2>/dev/null || true' EXIT

MODE=sweep \
INCLUDE=localhost:0,1 \
REPEATS=1 \
STOP_ON_ERROR=1 \
MAX_STEPS=10000 \
SWEEP_ZERO_STAGES=0 \
SWEEP_MICRO_BATCHES=128 \
SWEEP_NUM_WORKERS_LIST=6 \
SWEEP_PREFETCH_LIST=4 \
SWEEP_LOG_EVERY=50 \
TAIL_TIMING_POINTS=20 \
HEARTBEAT_EVERY_SEC=30 \
RUN_TIMEOUT_SEC=0 \
FAILURE_DUMP_TAIL=true \
FAIL_TAIL_LINES=120 \
RESUME_RUNS=false \
SWEEP_VALIDATION_EVERY=1000000 \
SWEEP_CHECKPOINT_EVERY=1000000 \
DATASET_ROOT=data/LibriSpeech/LibriSpeech_16k_trim \
DATASET_MANIFEST_PATH=data/LibriSpeech/LibriSpeech_16k_trim/manifest_16k_trim.tsv \
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
OUTPUT_ROOT="$OUT" \
DRIVER_LOG_PATH="$OUT/driver.log" \
./scripts/run_stage1_ab_bench.sh

kill $MON 2>/dev/null || true
wait $MON 2>/dev/null || true
trap - EXIT
echo "OUT=$OUT"
```

#### 3.4.1.2 跑一次长时 leak 诊断（4090，z1_mb128_nw6_pf4，MAX_STEPS=10000，含主机内存采样）

```bash
OUT=outputs/leak_4090_z1_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"

(
  echo "timestamp,epoch_sec,mem_total_kib,mem_available_kib,swap_total_kib,swap_free_kib,load1,load5,load15,train_proc_count,train_rss_kib,train_vsz_kib" > "$OUT/host_telemetry.csv"
  while true; do
    ts=$(date -Is); epoch=$(date +%s)
    eval "$(awk '
      /^MemTotal:/ {print "mt="$2}
      /^MemAvailable:/ {print "ma="$2}
      /^SwapTotal:/ {print "st="$2}
      /^SwapFree:/ {print "sf="$2}
      END {print "mt="mt"\nma="ma"\nst="st"\nsf="sf}
    ' /proc/meminfo)"
    read l1 l5 l15 _ < /proc/loadavg
    read pc pr pv <<< "$(ps -eo pid=,rss=,vsz=,args= | awk 'BEGIN{c=0;r=0;v=0} {pid=$1;rss=$2;vsz=$3;$1=$2=$3=""; sub(/^ +/,"",$0); if($0 ~ /(deepspeed|train.py)/){c++; r+=rss; v+=vsz}} END{printf "%d %d %d", c,r,v}')"
    echo "$ts,$epoch,$mt,$ma,$st,$sf,$l1,$l5,$l15,$pc,$pr,$pv" >> "$OUT/host_telemetry.csv"
    sleep 5
  done
) &
MON=$!
trap 'kill $MON 2>/dev/null || true' EXIT

MODE=sweep \
INCLUDE=localhost:0,1 \
REPEATS=1 \
STOP_ON_ERROR=1 \
MAX_STEPS=10000 \
SWEEP_ZERO_STAGES=1 \
SWEEP_MICRO_BATCHES=128 \
SWEEP_NUM_WORKERS_LIST=6 \
SWEEP_PREFETCH_LIST=4 \
SWEEP_LOG_EVERY=50 \
TAIL_TIMING_POINTS=20 \
HEARTBEAT_EVERY_SEC=30 \
RUN_TIMEOUT_SEC=0 \
FAILURE_DUMP_TAIL=true \
FAIL_TAIL_LINES=120 \
RESUME_RUNS=false \
SWEEP_VALIDATION_EVERY=1000000 \
SWEEP_CHECKPOINT_EVERY=1000000 \
DATASET_ROOT=data/LibriSpeech/LibriSpeech_16k_trim \
DATASET_MANIFEST_PATH=data/LibriSpeech/LibriSpeech_16k_trim/manifest_16k_trim.tsv \
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
OUTPUT_ROOT="$OUT" \
DRIVER_LOG_PATH="$OUT/driver.log" \
./scripts/run_stage1_ab_bench.sh

kill $MON 2>/dev/null || true
wait $MON 2>/dev/null || true
trap - EXIT
echo "OUT=$OUT"
```


#### 3.4.1.3 跑一次长时 leak 诊断（4090，z0_mb128_nw6_pf4，MAX_STEPS=10000，含主机内存采样，flash_attention_2）

```bash
OUT=outputs/leak_4090_z0_flash_attention_2_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"

(
  echo "timestamp,epoch_sec,mem_total_kib,mem_available_kib,swap_total_kib,swap_free_kib,load1,load5,load15,train_proc_count,train_rss_kib,train_vsz_kib" > "$OUT/host_telemetry.csv"
  while true; do
    ts=$(date -Is); epoch=$(date +%s)
    eval "$(awk '
      /^MemTotal:/ {print "mt="$2}
      /^MemAvailable:/ {print "ma="$2}
      /^SwapTotal:/ {print "st="$2}
      /^SwapFree:/ {print "sf="$2}
      END {print "mt="mt"\nma="ma"\nst="st"\nsf="sf}
    ' /proc/meminfo)"
    read l1 l5 l15 _ < /proc/loadavg
    read pc pr pv <<< "$(ps -eo pid=,rss=,vsz=,args= | awk 'BEGIN{c=0;r=0;v=0} {pid=$1;rss=$2;vsz=$3;$1=$2=$3=""; sub(/^ +/,"",$0); if($0 ~ /(deepspeed|train.py)/){c++; r+=rss; v+=vsz}} END{printf "%d %d %d", c,r,v}')"
    echo "$ts,$epoch,$mt,$ma,$st,$sf,$l1,$l5,$l15,$pc,$pr,$pv" >> "$OUT/host_telemetry.csv"
    sleep 5
  done
) &
MON=$!
trap 'kill $MON 2>/dev/null || true' EXIT

MODE=sweep \
INCLUDE=localhost:0,1 \
REPEATS=1 \
STOP_ON_ERROR=1 \
MAX_STEPS=10000 \
SWEEP_ZERO_STAGES=0 \
SWEEP_MICRO_BATCHES=128 \
SWEEP_NUM_WORKERS_LIST=6 \
SWEEP_PREFETCH_LIST=4 \
SWEEP_LOG_EVERY=50 \
TAIL_TIMING_POINTS=20 \
HEARTBEAT_EVERY_SEC=30 \
RUN_TIMEOUT_SEC=0 \
FAILURE_DUMP_TAIL=true \
FAIL_TAIL_LINES=120 \
RESUME_RUNS=false \
SWEEP_VALIDATION_EVERY=1000000 \
SWEEP_CHECKPOINT_EVERY=1000000 \
DATASET_ROOT=data/LibriSpeech/LibriSpeech_16k_trim \
DATASET_MANIFEST_PATH=data/LibriSpeech/LibriSpeech_16k_trim/manifest_16k_trim.tsv \
DATASET_USE_TRIM=false \
DATASET_OFFLINE_TRIMMED=true \
ENABLE_CUDA_SYNC_TIMING=false \
TIMING_RANK_SCOPE=all \
PRECISION_MODE=auto \
ATTN_IMPL=flash_attention_2 \
ENABLE_TORCH_COMPILE=false \
TORCH_COMPILE_MODE=max-autotune \
TORCH_COMPILE_DYNAMIC=true \
ENABLE_LENGTH_FIXED_SLICE=false \
ENABLE_GPU_TELEMETRY=true \
GPU_TELEMETRY_INTERVAL_SEC=2 \
STALL_ALERT_RATIO=2.0 \
OUTPUT_ROOT="$OUT" \
DRIVER_LOG_PATH="$OUT/driver.log" \
./scripts/run_stage1_ab_bench.sh

kill $MON 2>/dev/null || true
wait $MON 2>/dev/null || true
trap - EXIT
echo "OUT=$OUT"
```

#### 3.4.2 C


#### 3.4.2 D


### 3.5 2xv100

#### 3.5.1 B

先统一设置一组公共参数（按 2xv100）：

```bash
export INCLUDE=localhost:0,4
export STOP_ON_ERROR=0
export FAILURE_DUMP_TAIL=true
export FAIL_TAIL_LINES=120
export RESUME_RUNS=true
export HEARTBEAT_EVERY_SEC=30
export RUN_TIMEOUT_SEC=2400
export TAIL_TIMING_POINTS=10

export DATASET_ROOT=data/LibriSpeech/LibriSpeech_16k_trim
export DATASET_MANIFEST_PATH=data/LibriSpeech/LibriSpeech_16k_trim/manifest_16k_trim.tsv
export DATASET_USE_TRIM=false
export DATASET_OFFLINE_TRIMMED=true

export PRECISION_MODE=auto
export ATTN_IMPL=auto
export MODEL_LOAD_DTYPE=auto
export SPEECH_ATTN_IMPL=auto
export TEXT_ATTN_IMPL=auto
export ENABLE_TF32=true
export MATMUL_PRECISION=high
export ENABLE_CUDA_SYNC_TIMING=false
export ENABLE_GPU_TELEMETRY=true
export GPU_TELEMETRY_INTERVAL_SEC=2
export STALL_ALERT_RATIO=2.0

export SWEEP_LOG_EVERY=50
export SWEEP_VALIDATION_EVERY=1000000
export SWEEP_CHECKPOINT_EVERY=1000000

export ENABLE_TORCH_COMPILE=false
export TORCH_COMPILE_MODE=max-autotune
export TORCH_COMPILE_DYNAMIC=true
export ENABLE_LENGTH_FIXED_SLICE=false
```

哨兵组1: z0_mb128_nw6_pf4, repeat=3

```bash
MODE=sweep REPEATS=3 MAX_STEPS=1000 TIMING_RANK_SCOPE=all \
SWEEP_ZERO_STAGES=0 SWEEP_MICRO_BATCHES=128 SWEEP_NUM_WORKERS_LIST=6 SWEEP_PREFETCH_LIST=4 \
OUTPUT_ROOT=outputs/bench_v100_phaseB_diag \
DRIVER_LOG_PATH=outputs/bench_v100_phaseB_diag/driver.log \
./scripts/run_stage1_ab_bench.sh
```

哨兵组2: z1_mb160_nw6_pf2, repeat=3（同一个 OUTPUT_ROOT，直接追加）
```bash
MODE=sweep REPEATS=3 MAX_STEPS=1000 TIMING_RANK_SCOPE=all \
SWEEP_ZERO_STAGES=1 SWEEP_MICRO_BATCHES=160 SWEEP_NUM_WORKERS_LIST=6 SWEEP_PREFETCH_LIST=2 \
OUTPUT_ROOT=outputs/bench_v100_phaseB_diag \
DRIVER_LOG_PATH=outputs/bench_v100_phaseB_diag/driver.log \
./scripts/run_stage1_ab_bench.sh
```

#### 3.5.1.1 跑一次长时 leak 诊断（4090，z0_mb128_nw6_pf4，MAX_STEPS=10000，含主机内存采样）

```bash
OUT=outputs/leak_v100_z0_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"

(
  echo "timestamp,epoch_sec,mem_total_kib,mem_available_kib,swap_total_kib,swap_free_kib,load1,load5,load15,train_proc_count,train_rss_kib,train_vsz_kib" > "$OUT/host_telemetry.csv"
  while true; do
    ts=$(date -Is); epoch=$(date +%s)
    eval "$(awk '
      /^MemTotal:/ {print "mt="$2}
      /^MemAvailable:/ {print "ma="$2}
      /^SwapTotal:/ {print "st="$2}
      /^SwapFree:/ {print "sf="$2}
      END {print "mt="mt"\nma="ma"\nst="st"\nsf="sf}
    ' /proc/meminfo)"
    read l1 l5 l15 _ < /proc/loadavg
    read pc pr pv <<< "$(ps -eo pid=,rss=,vsz=,args= | awk 'BEGIN{c=0;r=0;v=0} {pid=$1;rss=$2;vsz=$3;$1=$2=$3=""; sub(/^ +/,"",$0); if($0 ~ /(deepspeed|train.py)/){c++; r+=rss; v+=vsz}} END{printf "%d %d %d", c,r,v}')"
    echo "$ts,$epoch,$mt,$ma,$st,$sf,$l1,$l5,$l15,$pc,$pr,$pv" >> "$OUT/host_telemetry.csv"
    sleep 5
  done
) &
MON=$!
trap 'kill $MON 2>/dev/null || true' EXIT

MODE=sweep \
INCLUDE=localhost:1,3 \
REPEATS=1 \
STOP_ON_ERROR=1 \
MAX_STEPS=10000 \
SWEEP_ZERO_STAGES=0 \
SWEEP_MICRO_BATCHES=128 \
SWEEP_NUM_WORKERS_LIST=6 \
SWEEP_PREFETCH_LIST=4 \
SWEEP_LOG_EVERY=50 \
TAIL_TIMING_POINTS=20 \
HEARTBEAT_EVERY_SEC=30 \
RUN_TIMEOUT_SEC=0 \
FAILURE_DUMP_TAIL=true \
FAIL_TAIL_LINES=120 \
RESUME_RUNS=false \
SWEEP_VALIDATION_EVERY=1000000 \
SWEEP_CHECKPOINT_EVERY=1000000 \
DATASET_ROOT=data/LibriSpeech/LibriSpeech_16k_trim \
DATASET_MANIFEST_PATH=data/LibriSpeech/LibriSpeech_16k_trim/manifest_16k_trim.tsv \
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
OUTPUT_ROOT="$OUT" \
DRIVER_LOG_PATH="$OUT/driver.log" \
./scripts/run_stage1_ab_bench.sh

kill $MON 2>/dev/null || true
wait $MON 2>/dev/null || true
trap - EXIT
echo "OUT=$OUT"
```

#### 3.5.1.1 跑一次长时 leak 诊断（4090，z1_mb128_nw6_pf4，MAX_STEPS=10000，含主机内存采样）

```bash
OUT=outputs/leak_v100_z1_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"

(
  echo "timestamp,epoch_sec,mem_total_kib,mem_available_kib,swap_total_kib,swap_free_kib,load1,load5,load15,train_proc_count,train_rss_kib,train_vsz_kib" > "$OUT/host_telemetry.csv"
  while true; do
    ts=$(date -Is); epoch=$(date +%s)
    eval "$(awk '
      /^MemTotal:/ {print "mt="$2}
      /^MemAvailable:/ {print "ma="$2}
      /^SwapTotal:/ {print "st="$2}
      /^SwapFree:/ {print "sf="$2}
      END {print "mt="mt"\nma="ma"\nst="st"\nsf="sf}
    ' /proc/meminfo)"
    read l1 l5 l15 _ < /proc/loadavg
    read pc pr pv <<< "$(ps -eo pid=,rss=,vsz=,args= | awk 'BEGIN{c=0;r=0;v=0} {pid=$1;rss=$2;vsz=$3;$1=$2=$3=""; sub(/^ +/,"",$0); if($0 ~ /(deepspeed|train.py)/){c++; r+=rss; v+=vsz}} END{printf "%d %d %d", c,r,v}')"
    echo "$ts,$epoch,$mt,$ma,$st,$sf,$l1,$l5,$l15,$pc,$pr,$pv" >> "$OUT/host_telemetry.csv"
    sleep 5
  done
) &
MON=$!
trap 'kill $MON 2>/dev/null || true' EXIT

MODE=sweep \
INCLUDE=localhost:1,3 \
REPEATS=1 \
STOP_ON_ERROR=1 \
MAX_STEPS=10000 \
SWEEP_ZERO_STAGES=1 \
SWEEP_MICRO_BATCHES=128 \
SWEEP_NUM_WORKERS_LIST=6 \
SWEEP_PREFETCH_LIST=4 \
SWEEP_LOG_EVERY=50 \
TAIL_TIMING_POINTS=20 \
HEARTBEAT_EVERY_SEC=30 \
RUN_TIMEOUT_SEC=0 \
FAILURE_DUMP_TAIL=true \
FAIL_TAIL_LINES=120 \
RESUME_RUNS=false \
SWEEP_VALIDATION_EVERY=1000000 \
SWEEP_CHECKPOINT_EVERY=1000000 \
DATASET_ROOT=data/LibriSpeech/LibriSpeech_16k_trim \
DATASET_MANIFEST_PATH=data/LibriSpeech/LibriSpeech_16k_trim/manifest_16k_trim.tsv \
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
OUTPUT_ROOT="$OUT" \
DRIVER_LOG_PATH="$OUT/driver.log" \
./scripts/run_stage1_ab_bench.sh

kill $MON 2>/dev/null || true
wait $MON 2>/dev/null || true
trap - EXIT
echo "OUT=$OUT"
```

Notes:

- `HEARTBEAT_EVERY_SEC=30` prints latest observed step periodically to avoid "silent hanging" perception.
- `STALL_ALERT_RATIO=2.0` enables heartbeat spike alerts when `iter_ms_p50` jumps abruptly.
- `RUN_TIMEOUT_SEC=2400` marks a run as failed if it exceeds 40 minutes (tune up/down by your platform quota).
- `RESUME_RUNS=true` keeps `run_manifest.tsv` and skips already-successful groups when rerunning after interruption.
- On failure, script now records `exit_code`, `duration_sec`, `last_step` in `run_manifest.tsv` and prints launcher/train tail automatically.
- `ENABLE_GPU_TELEMETRY=true` writes `gpu_telemetry.csv` for each run (power/utilization/clock snapshots), with interval controlled by `GPU_TELEMETRY_INTERVAL_SEC`.
- `run_manifest.tsv` now includes `gpu_telemetry_rows` and `gpu_telemetry_empty_flag` to quickly detect telemetry collection failures.
- `run_manifest.tsv` now includes `git_commit_hash`, `git_commit_short`, `git_branch`, `git_dirty` for reproducible run-to-code mapping.
- `run_manifest.tsv` now includes `iter_p90_over_p50`, `data_p90_over_p50`, `step_p90_over_p50`, and `unstable_run_flag` for stability diagnosis.
- `ENABLE_LENGTH_FIXED_SLICE=false` by default. Enable it only for throughput-only A/B tests, e.g. `ENABLE_LENGTH_FIXED_SLICE=true FIXED_SLICE_SECONDS=2.0`.
- Prefer `DRIVER_LOG_PATH=... ./scripts/run_stage1_ab_bench.sh` over external `| tee ...`. If you still use external `tee`, pre-create the directory first.

Outputs:

- `outputs/.../run_manifest.tsv`
- `outputs/.../per_run_metrics.csv`
- `outputs/.../group_summary.csv`
- `outputs/.../ranked_groups.csv`
- `outputs/.../*/gpu_telemetry.csv` (when telemetry is enabled)
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
