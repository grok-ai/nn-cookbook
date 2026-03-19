"""
---
name: Multi-Loss
slug: multi_loss
description: Weighted combination of multiple loss functions.
dependencies: []
conflicts: []
touches: [train.py, config/train.yaml]
tags: [optimization]
test_instructions: |
  - Test that total loss equals the weighted sum of individual losses.
  - Test that each individual loss is computed correctly.
  - Test with a single loss (weight=1.0) matches standard single-loss training.
  - Test that zero weight on a loss excludes it from total (gradient doesn't flow through it).
  - Test that all individual losses are returned for logging.
  - Test that loss weights are read from config correctly.
---
"""

# Computes a weighted combination of multiple losses.
#
# Changes to train.py:
#   - Replace single loss computation with compute_losses() call
#   - Total loss = sum(weight_i * loss_i)
#   - Log individual losses if wandb is enabled
#
# Changes to config/train.yaml:
#   + losses:
#   +   cross_entropy:
#   +     weight: 1.0
#   +   label_smoothing_ce:
#   +     weight: 0.5
#   +     label_smoothing: 0.1

import torch
import torch.nn.functional as F


LOSS_REGISTRY = {
    "cross_entropy": lambda output, target, **kw: F.cross_entropy(output, target),
    "label_smoothing_ce": lambda output, target, **kw: F.cross_entropy(
        output, target, label_smoothing=kw.get("label_smoothing", 0.1)
    ),
    "mse": lambda output, target, **kw: F.mse_loss(output, target.float()),
    "l1": lambda output, target, **kw: F.l1_loss(output, target.float()),
}


def compute_losses(
    output: torch.Tensor,
    target: torch.Tensor,
    loss_configs: dict,
) -> tuple[torch.Tensor, dict[str, float]]:
    individual = {}
    total = torch.tensor(0.0, device=output.device)
    for name, lcfg in loss_configs.items():
        loss_fn = LOSS_REGISTRY[name]
        kwargs = {k: v for k, v in lcfg.items() if k != "weight"}
        loss_val = loss_fn(output, target, **kwargs)
        individual[name] = loss_val.item()
        total = total + lcfg.weight * loss_val
    individual["total"] = total.item()
    return total, individual


# In train_epoch():
#
# loss, individual_losses = compute_losses(output, y, cfg.losses)
# loss.backward()
