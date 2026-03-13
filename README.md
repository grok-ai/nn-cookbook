# nn-cookbook

A Claude Code plugin for composable PyTorch training. Generate complete training projects from modular ingredients — no framework, no Trainer class, just plain PyTorch + Hydra config.

## Installation

Inside Claude Code, add the marketplace and install:

```
/plugin marketplace add grok-ai/nn-cookbook
/plugin install nn-cookbook@nn-cookbook
```

Or clone and use locally without marketplace setup:

```bash
git clone https://github.com/grok-ai/nn-cookbook.git
claude --plugin-dir ./nn-cookbook
```

## Usage

```
/training                                    # Interactive: pick ingredients + dataset
/training mixed_precision,checkpointing      # Direct: specify ingredients by slug
/training default                            # Default preset (see below)
```

### Default preset

`/training default` selects a recommended set of ingredients for most training jobs:

mixed_precision, checkpointing, lr_scheduler, logging_wandb, dataloader_workers, gradient_clipping, ema

You can also type `default` or `d` when prompted during interactive selection.

## How it works

1. You pick ingredients (gradient accumulation, mixed precision, checkpointing, etc.)
2. You name your project and describe your task (dataset, model, classification/regression/etc.)
3. Claude composes a working training project with all ingredients correctly integrated
4. Tests are auto-generated and verified before delivery

## Available ingredients

| Ingredient | Description |
|---|---|
| `base_training_loop` | Core train/val loop with step-level logging and inline validation (always included) |
| `gradient_accumulation` | Accumulate gradients over N micro-batches |
| `mixed_precision` | AMP with bf16 (preferred) or fp16 + GradScaler |
| `checkpointing` | Save/load training state (per-epoch and every N steps) |
| `lr_scheduler` | Learning rate scheduling (cosine, cosine with warmup, step) |
| `logging_wandb` | Weights & Biases logging |
| `dataloader_workers` | Multi-worker data loading with prefetch |
| `gradient_clipping` | Gradient norm clipping |
| `reproducibility` | Deterministic seeds and settings |
| `ema` | Exponential moving average of model weights |
| `early_stopping` | Stop training when validation metric plateaus |
| `ddp` | DistributedDataParallel multi-GPU training |
| `multi_loss` | Weighted combination of multiple losses |
| `activation_checkpointing` | Trade compute for memory by recomputing activations during backward |

## Generated project structure

```
<project_name>/
├── train.py           # Main loop with selected ingredients
├── data.py            # Dataset + DataLoader
├── model.py           # Model definition
├── config/
│   └── train.yaml     # Hydra config
├── tests/
│   └── test_training.py
└── pyproject.toml
```

## Design principles

- **Plain code.** Functions and loops, no Trainer class, no callbacks, no dataclasses.
- **Hydra config.** YAML files, `@hydra.main`, `cfg.lr`.
- **Type-hinted.** All function signatures are annotated.
- **Tested out of the box.** Each ingredient defines test cases; generated projects pass tests before delivery.
