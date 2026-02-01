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

1. Create a virtual environment (optional but recommended):
   ```bash
   conda create -n csref_2 python=3.10
   conda activate csref_2
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset

This project uses the LibriSpeech dataset. Ensure you have the dataset downloaded and extracted.
Update `configs/dataset/librispeech.yaml` with the correct `root_dir`:
```yaml
root_dir: "/path/to/LibriSpeech"
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
deepspeed --hostfile /path/to/hostfile train.py
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
