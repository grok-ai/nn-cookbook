"""
---
name: Gradient Clipping
slug: gradient_clipping
description: Clip gradient norms to prevent exploding gradients.
dependencies: []
conflicts: []
touches: [train.py, config/train.yaml]
tags: [optimization, stability]
test_instructions: |
  - Test that gradient norms are clipped to max_grad_norm after clipping.
  - Test that clipping with a very large max_grad_norm does not change gradients.
  - Test that clipping with max_grad_norm=0 effectively zeros gradients.
  - Test that clip_grad_norm_ returns the original (unclipped) total norm.
  - Test that model still trains correctly with gradient clipping enabled.
---
"""

# Adds gradient clipping before optimizer.step().
#
# Changes to train.py:
#   - torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
#     inserted between loss.backward() and optimizer.step()
#
# Changes to config/train.yaml:
#   + max_grad_norm: 1.0

import torch


def clip_gradients(model: torch.nn.Module, max_grad_norm: float) -> torch.Tensor:
    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)


# In train_epoch(), between backward() and optimizer.step():
#
# grad_norm = clip_gradients(model, cfg.max_grad_norm)
