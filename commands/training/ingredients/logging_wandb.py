"""
---
name: Weights & Biases Logging
slug: logging_wandb
description: Log training metrics, hyperparameters, and system stats to W&B.
dependencies: []
conflicts: []
touches: [train.py, config/train.yaml]
tags: [logging]
test_instructions: |
  - Test that wandb.init() is called with correct project name and config.
  - Test that wandb.log() is called each epoch with train_loss, val_loss, val_acc.
  - Test that wandb.finish() is called at the end of training.
  - Test that training works with wandb disabled (use_wandb=false).
  - Test with a mock wandb module to verify correct call sequence.
---
"""

# Adds W&B initialization, logging, and finish.
#
# Changes to train.py:
#   - wandb.init() at start of main() if cfg.use_wandb
#   - wandb.log() after each epoch
#   - wandb.finish() at end
#
# Changes to config/train.yaml:
#   + use_wandb: false
#   + wandb_project: my-training
#   + wandb_run_name: null

import wandb
from omegaconf import DictConfig


def init_wandb(cfg: DictConfig) -> None:
    if not cfg.use_wandb:
        return
    wandb.init(
        project=cfg.wandb_project,
        name=cfg.wandb_run_name,
        config=dict(cfg),
    )


def log_metrics(metrics: dict[str, float], step: int, cfg: DictConfig) -> None:
    if not cfg.use_wandb:
        return
    wandb.log(metrics, step=step)


def finish_wandb(cfg: DictConfig) -> None:
    if not cfg.use_wandb:
        return
    wandb.finish()


# In main():
#
# init_wandb(cfg)
# for epoch in range(cfg.epochs):
#     train_loss = train_epoch(...)
#     val_loss, val_acc = validate(...)
#     log_metrics({"train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc}, epoch, cfg)
# finish_wandb(cfg)
