"""
---
name: Learning Rate Scheduler
slug: lr_scheduler
description: Configurable learning rate scheduling (cosine, cosine with warmup, step).
dependencies: []
conflicts: []
touches: [train.py, config/train.yaml]
tags: [optimization]
test_instructions: |
  - Test that learning rate changes after scheduler.step().
  - Test that cosine annealing schedule reaches lr_min at the end of training.
  - Test that scheduler state is consistent after save/load (if checkpointing is used).
  - Test that initial learning rate matches cfg.lr.
  - Test with step_lr schedule that lr drops by gamma every step_size epochs.
  - Test that cosine_warmup linearly increases lr from 0 to cfg.lr over warmup_steps, then decays via cosine.
---
"""

# Adds LR scheduler creation and stepping.
#
# Changes to train.py:
#   - Create scheduler after optimizer in main()
#   - Call scheduler.step() after each epoch (or each step for per-step schedulers)
#   - For cosine_warmup: call scheduler.step() every training step, not per-epoch
#
# Changes to config/train.yaml:
#   + scheduler: cosine       # cosine | cosine_warmup | step | none
#   + lr_min: 1e-6            # for cosine / cosine_warmup
#   + warmup_steps: 100       # for cosine_warmup (linear warmup phase)
#   + step_size: 30           # for step
#   + gamma: 0.1              # for step

import torch
from omegaconf import DictConfig


def get_cosine_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    lr_min: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup for warmup_steps, then cosine decay to lr_min."""
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        import math
        return max(lr_min / optimizer.defaults["lr"], 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: DictConfig,
    steps_per_epoch: int = 0,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    if cfg.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.epochs, eta_min=cfg.lr_min,
        )
    elif cfg.scheduler == "cosine_warmup":
        total_steps = cfg.epochs * steps_per_epoch
        return get_cosine_warmup_scheduler(
            optimizer, cfg.warmup_steps, total_steps, cfg.lr_min,
        )
    elif cfg.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg.step_size, gamma=cfg.gamma,
        )
    elif cfg.scheduler == "none":
        return None
    else:
        raise ValueError(f"Unknown scheduler: {cfg.scheduler}")


# In main():
#
# scheduler = get_scheduler(optimizer, cfg, steps_per_epoch=len(train_loader))
# for epoch in range(cfg.epochs):
#     train_loss = train_epoch(...)
#     val_loss, val_acc = validate(...)
#     if scheduler is not None:
#         if cfg.scheduler == "cosine_warmup":
#             pass  # stepped per-batch inside train_epoch
#         else:
#             scheduler.step()
#     print(f"... lr: {optimizer.param_groups[0]['lr']:.6f}")
#
# Inside train_epoch (when cosine_warmup):
#     if scheduler is not None:
#         scheduler.step()
