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
