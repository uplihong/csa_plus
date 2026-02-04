# CSRef 2.0: Unified Speech-Text Pretraining Framework

This repository is a refactored version of the CSRef+ project, providing a unified and modular framework for speech-text pretraining using Contrastive Semantic Alignment (CSA). It features **DeepSpeed** integration for efficient distributed training (single-node and multi-node).

## Project Structure

```
CSRef_2.0/
├── configs/                # Hydra configurations
├── csref/                  # Main package
│   ├── core/               # Core engine (Trainer)
│   ├── data/               # Data pipeline (Dataset, Dataloader, Transforms)
│   ├── modeling/           # Model definitions (Encoders, CSA)
│   └── utils/              # Utilities
├── train.py                # Main entry point
└── requirements.txt
```

## Installation
> conda or docker

### Conda

1. Create a virtual environment (optional but recommended):
   ```bash
   conda create -n csref_2 python=3.10
   conda activate csref_2
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. if using Conda environment, please also install:
   ```
   conda install -y openmpi mpi4py
   ```

### docker

```
docker build -t csa_plus:cuda12.8 .
```

```
docker run --rm -it --gpus all --shm-size=8g \
  -v $PWD:/code \
  -v /data/huanglh/dataset/LibriSpeech/:/data/huanglh/dataset/LibriSpeech/ \
  -v /data/huanglh/huggingface/git/bert-base-uncased/:/data/huanglh/huggingface/git/bert-base-uncased/ \
  -v /data/huanglh/huggingface/git/wav2vec2-base/:/data/huanglh/huggingface/git/wav2vec2-base/ \
  -v /data/huanglh/code/ContrastiveAT/output/wav2vec2base_extractedtextfeature_freezefeatureencoder_32batchsize_4gpu_amp/ckpt_epoch_8.pth:/data/huanglh/code/ContrastiveAT/output/wav2vec2base_extractedtextfeature_freezefeatureencoder_32batchsize_4gpu_amp/ckpt_epoch_8.pth \
  -w /code \
  csa_plus:cuda12.8 bash
```

## Dataset

This project uses the LibriSpeech dataset. Ensure you have the dataset downloaded and extracted.
Update `configs/dataset/librispeech.yaml` with the correct `root_dir`:
```yaml
root_dir: "/path/to/LibriSpeech"
```

## Optional: Preprocess Audio (16k FLAC cache)
If CPU resampling becomes a bottleneck, you can pre-resample LibriSpeech to 16k FLAC and point `dataset.root_dir`
to the new folder:

```bash
python scripts/preprocess_librispeech_16k.py \
  --input-root /path/to/LibriSpeech \
  --output-root /path/to/LibriSpeech_16k \
  --workers 8
```

## Weights

This project uses the `bert-base-uncased` and `wav2vec2-base` pretrained weights.

```
git clone https://huggingface.co/google-bert/bert-base-uncased
git clone https://huggingface.co/facebook/wav2vec2-base
```

Update `configs/model/text_encoder/bert.yaml` and `configs/model/speech_encoder/wav2vec2.yaml` with the correct `pretrained_path`

## Verification

```
python train.py ++train.max_step_iterations=1000 ++train.log_every_steps=1 ++train.checkpoint_every_steps=100 ++train.validation_every_steps=100 +experiment=limit_longest_1-5_stage2

deepspeed --num_gpus 1 train.py ++train.max_step_iterations=10 +experiment=limit_longest_1-5_stage2

NCCL_P2P_DISABLE=1 deepspeed \
  --include 'localhost:0,4' train.py \
  '++train.max_step_iterations=100000' \
  '++train.log_every_steps=1' \
  '++train.validation_every_steps=500' \
  '++train.checkpoint_every_steps=500' \
  'deepspeed_config_yaml.train_micro_batch_size_per_gpu=128' \
  '+experiment=limit_longest_1-5_stage2'

NCCL_P2P_DISABLE=1 deepspeed   --include 'localhost:0,4' train.py   '++train.max_step_iterations=100'   '++train.log_every_steps=1'   '++train.validation_every_steps=10'   '++train.checkpoint_every_steps=10'   'deepspeed_config_yaml.train_micro_batch_size_per_gpu=4'   '+experiment=limit_longest_1-5_stage2'   '++train.evaluation.eval_batch_size=4'
```

## Training

The framework uses **Hydra** for configuration and **DeepSpeed** for distributed training.

### 1. Single GPU Training (Debug/Dev)
```bash
python train.py train.batch_size=8 dataset.train_split=train
```

### 2. Single-Node Multi-GPU (DeepSpeed)
To run on a single machine with multiple GPUs (e.g., 2 GPUs):

```bash
deepspeed --num_gpus 2 train.py
```

### 3. Multi-Node Distributed Training
Create a `hostfile` listing your nodes and slot counts (see DeepSpeed docs).

```bash
deepspeed --hostfile=/etc/deepspeed/hostfile \
  train.py +experiment=limit_longest_1-5_stage2 \
  dataset.root_dir=/mnt/fs/LibriSpeech/LibriSpeech \
  train.pretrained_model_checkpoint=
```

### Configuration
Key configurations can be overridden via command line:
- `dataset.transform` parameters (controlled via code or extended config)
- `train.epochs`: Number of epochs (or `max_step_iterations` for infinite loop)
- `deepspeed_config_yaml`: Path/Name of DeepSpeed config (e.g., `ds_config_stage2`)

## Features

- **Modular Design**: Separated Modeling, Data, and Core logic.
- **Robust Training Loop**: `Trainer` class handles DeepSpeed initialization, checkpointing, and logging.
- **Mixed Precision**: Supported via DeepSpeed config (FP16/BF16).
- **Infinite Data Loading**: Support for infinite iterator training loops common in pretraining.
