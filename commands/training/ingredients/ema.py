"""
---
name: Exponential Moving Average
slug: ema
description: Maintain an EMA copy of model weights for better generalization at inference.
dependencies: []
conflicts: []
touches: [train.py, config/train.yaml]
tags: [optimization, generalization]
test_instructions: |
  - Test that EMA weights differ from model weights after training steps.
  - Test that EMA weights are a weighted average (closer to recent weights with high decay).
  - Test that EMA with decay=0.0 makes EMA weights equal to model weights.
  - Test that EMA with decay=1.0 keeps EMA weights at their initial values.
  - Test that EMA update does not affect the original model's parameters.
  - Test that EMA model produces valid outputs (forward pass works).
---
"""

# Maintains an EMA copy of model weights.
#
# Changes to train.py:
#   - Create EMA model in main(): ema_model = create_ema(model, cfg.ema_decay)
#   - Update EMA after each optimizer step: update_ema(ema_model, model, cfg.ema_decay)
#   - Optionally validate with EMA model
#
# Changes to config/train.yaml:
#   + ema_decay: 0.999
#   + use_ema: true

import copy
import torch


def create_ema(model: torch.nn.Module, decay: float) -> torch.nn.Module:
    ema_model = copy.deepcopy(model)
    for p in ema_model.parameters():
        p.requires_grad_(False)
    return ema_model


@torch.no_grad()
def update_ema(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    for ema_p, model_p in zip(ema_model.parameters(), model.parameters()):
        ema_p.data.mul_(decay).add_(model_p.data, alpha=1.0 - decay)


# In main():
#
# if cfg.use_ema:
#     ema_model = create_ema(model, cfg.ema_decay)
#
# In train_epoch() or after each optimizer.step():
#
# if cfg.use_ema:
#     update_ema(ema_model, model, cfg.ema_decay)
#
# For validation with EMA:
#
# eval_model = ema_model if cfg.use_ema else model
# val_loss, val_acc = validate(eval_model, val_loader, device)
