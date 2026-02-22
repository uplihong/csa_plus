# CSRef 2.0

Unified framework for speech-text pretraining with Contrastive Semantic Alignment (CSA), Hydra configuration, and DeepSpeed distributed training.

## Project Layout

```text
configs/                     # Hydra configs
csref/                       # core training/data/model code
train.py                     # training entrypoint
scripts/run_stage1_ab_bench.sh
scripts/summarize_stage1_bench.py
```

## Installation

## Conda

```bash
conda create -n csref_2 python=3.10 -y
conda activate csref_2
pip install -r requirements.txt
conda install -y openmpi mpi4py
```

Optional:

```bash
python -m pip install --use-pep517 --no-build-isolation flash-attn
```

## Docker

```bash
docker build -t csa_plus:cuda12.8 .
```

## Data Preparation

This repo uses LibriSpeech. Update dataset path in `configs/dataset/librispeech.yaml` or pass overrides at runtime.

Offline 16k preprocessing (recommended for stable throughput benchmarking):

```bash
python scripts/preprocess_librispeech_16k.py \
  --input-root data/LibriSpeech/LibriSpeech \
  --output-root data/LibriSpeech/LibriSpeech_16k_trim \
  --target-sr 16000 \
  --trim \
  --manifest-path data/LibriSpeech/LibriSpeech_16k_trim/manifest_16k_trim.tsv
```

## Pretrained Weights

```bash
git clone https://huggingface.co/google-bert/bert-base-uncased
git clone https://huggingface.co/facebook/wav2vec2-base
```

Set paths in:
- `configs/model/text_encoder/bert.yaml`
- `configs/model/speech_encoder/wav2vec2.yaml`

## Training Quick Start

Single GPU:

```bash
python train.py +experiment=limit_longest_1-3_stage1_bf16 ++train.max_step_iterations=50
```

Multi-GPU:

```bash
deepspeed --num_gpus 2 train.py +experiment=limit_longest_1-3_stage1_bf16
```

Multi-node (scheduler + hostfile):

```bash
REPO=/code/csa_plus
TS=$(date +%Y%m%d_%H%M%S)
OUT=${REPO}/outputs/smoke_stage1_multinode_${TS}
mkdir -p "${OUT}"
cd "${REPO}"

deepspeed \
  --hostfile /etc/deepspeed/hostfile \
  --launcher pdsh \
  train.py \
  +experiment=limit_longest_1-3_stage1_bf16 \
  experiment_output_dir="${OUT}" \
  hydra.run.dir="${OUT}" \
  ++train.max_step_iterations=100000 \
  ++train.log_every_steps=20 \
  ++train.validation_every_steps=500 \
  ++train.checkpoint_every_steps=500 \
  ++train.checkpoint_save_latest=true \
  ++train.checkpoint_exclude_frozen_parameters=true \
  ++train.checkpoint_tag_style=iter \
  ++train.deterministic=false \
  ++train.cudnn_benchmark=false \
  ++train.enable_cuda_sync_timing=false \
  ++train.timing_rank_scope=rank0 \
  ++train.enable_eta_logging=true \
  ++train.eta_distributed_mode=rank0 \
  ++train.eta_min_samples=10 \
  ++train.data.num_workers=8 \
  ++train.data.prefetch_factor=4 \
  ++dataset.root_dir=data/LibriSpeech/LibriSpeech_16k_trim \
  ++dataset.manifest_path=data/LibriSpeech/LibriSpeech_16k_trim/manifest_16k_trim.tsv \
  ++dataset.use_trim=false \
  ++dataset.offline_trimmed=true \
  ++train.pretrained_model_checkpoint=data/weights/csa/ckpt_epoch_8.pth \
  ++model.speech_encoder.pretrained_path=data/weights/wav2vec2-base \
  ++model.text_encoder.pretrained_path=data/weights/bert-base-uncased \
  ++model.speech_encoder.attn_implementation=eager \
  ++model.text_encoder.attn_implementation=eager \
  ++model.speech_encoder.torch_dtype=bf16 \
  ++model.text_encoder.torch_dtype=bf16 \
  deepspeed_config_yaml.zero_optimization.stage=0 \
  deepspeed_config_yaml.train_micro_batch_size_per_gpu=128 \
  deepspeed_config_yaml.wall_clock_breakdown=false \
  deepspeed_config_yaml.bf16.enabled=true \
  deepspeed_config_yaml.fp16.enabled=false \
  2>&1 | tee "${OUT}/driver.log"
```

Startup artifacts (written to `experiment_output_dir`):
- `resolved_config.yaml`: resolved Hydra configuration snapshot for run reproduction.
- `run_context.json`: launch-time runtime context, including git commit/branch/dirty state and a bounded `git status --porcelain` summary.
- if git working tree is dirty, training continues with a warning and the dirty state is recorded.

Checkpointing (DeepSpeed official flow):
- trainer calls `DeepSpeedEngine.save_checkpoint(save_dir, tag, client_state, save_latest, exclude_frozen_parameters)` on all ranks.
- reference:
  - https://deepspeed.readthedocs.io/en/latest/model-checkpointing.html
  - https://deepspeed.readthedocs.io/en/stable/training.html
- default config:
  - `train.checkpoint_save_latest=true`
  - `train.checkpoint_exclude_frozen_parameters=true` (recommended for multinode memory stability)
  - `train.checkpoint_tag_style=iter`
- output layout:
  - checkpoint payload under `<run_dir>/<tag>/` (for example: `<run_dir>/iter_500/`)
  - pointer file `<run_dir>/latest` tracks the newest tag when `checkpoint_save_latest=true`.
- if strict full-parameter snapshots are needed for compatibility checks, override:
  - `++train.checkpoint_exclude_frozen_parameters=false`

Emergency fallback for platforms that cannot provide `pdsh`:
- launch each node separately with DeepSpeed `--no_ssh`
- see: https://www.deepspeed.ai/getting-started/#launching-without-passwordless-ssh
- if fused optimizer JIT is unavailable, add override:
  - `deepspeed_config_yaml.optimizer.params.torch_adam=true`

## Stage1 Benchmarking (Mainline)

Mainline benchmark entry:
- `scripts/run_stage1_ab_bench.sh`
- `scripts/summarize_stage1_bench.py`

See the operational SOP:
- `docs/stage1_distributed_sop.md`

Key policy (frozen mainline):
- `zero_stage=0`
- `PRECISION_MODE=auto`
- `ATTN_IMPL=auto` (effective `sdpa` on 4090/H100)
- `ENABLE_TF32=true`
- `ENABLE_TORCH_COMPILE=false`
- `ENABLE_LENGTH_FIXED_SLICE=false`
- `ENABLE_LENGTH_BUCKET=false`
- `ENABLE_ETA_LOGGING=true`
- `ETA_DISTRIBUTED_MODE=rank0` (low overhead default; use `global_max` for conservative distributed ETA)
- `ETA_MIN_SAMPLES=10`

ETA-related outputs:
- `run_manifest.tsv` includes `last_eta_step`, `last_eta_remaining_steps`, `last_eta_sec`, `last_eta_hms`, `last_eta_mode`
- `eta_points.csv` records run-level ETA trajectories

Runtime notes for benchmarking:
- if `deepspeed` is missing in current shell, run with `CONDA_ENV=csref_2` (or your env)
- on hosts with NCCL peer-to-peer instability, add `NCCL_P2P_DISABLE=1` for distributed benchmark runs

Optional diagnosis only:
- `scripts/quick_leak_diagnose.py`

## Historical Experiments

Historical Phase A/B/C commands and retired exploratory paths are archived at:
- `docs/archive_stage1_experiments.md`

## Notes

- If you run with external `tee`, pre-create the output directory first.
- Compare performance across machines only after ensuring platform parity (GPU type/topology, driver, storage health, background load).
