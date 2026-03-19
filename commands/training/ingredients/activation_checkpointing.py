"""
---
name: Activation Checkpointing
slug: activation_checkpointing
description: Trade compute for memory by recomputing activations during backward pass instead of storing them.
dependencies: []
conflicts: []
touches: [train.py, config/train.yaml]
tags: [memory, performance]
test_instructions: |
  - Test that activation checkpointing reduces peak memory usage compared to without it.
  - Test that model produces the same gradients with and without activation checkpointing (numerical correctness).
  - Test that training completes successfully with activation checkpointing enabled.
  - Test that apply_activation_checkpointing correctly wraps the specified module types.
---
"""

# Wraps selected model layers with torch.utils.checkpoint so activations
# are recomputed during backward instead of cached. Significantly reduces
# memory at the cost of ~30% extra compute.
#
# Changes to train.py:
#   - After model creation, call apply_activation_checkpointing(model, cfg)
#   - Forward pass must use use_reentrant=False (torch >= 2.1 default)
#
# Changes to config/train.yaml:
#   + activation_checkpointing: true
#   + ac_layer_types: null     # null = all layers, or list like ["TransformerBlock"]
#
# Composition note (activation_checkpointing + DDP):
#   Apply activation checkpointing BEFORE wrapping with DDP.

from functools import partial
import torch
from torch.utils.checkpoint import checkpoint
from omegaconf import DictConfig


def apply_activation_checkpointing(model: torch.nn.Module, cfg: DictConfig) -> None:
    """Wrap forward() of eligible layers to use gradient checkpointing."""
    import torch.nn as nn

    layer_types = getattr(cfg, "ac_layer_types", None)

    def _should_wrap(module: nn.Module) -> bool:
        if layer_types is None:
            return len(list(module.children())) > 0 and not isinstance(module, type(model))
        return type(module).__name__ in layer_types

    def _checkpointed_forward(original_forward, *args, **kwargs):
        return checkpoint(original_forward, *args, use_reentrant=False, **kwargs)

    for module in model.modules():
        if _should_wrap(module):
            original_forward = module.forward
            module.forward = partial(_checkpointed_forward, original_forward)


# In main():
#
# model = get_model(cfg).to(device)
# if cfg.activation_checkpointing:
#     apply_activation_checkpointing(model, cfg)
# # ... then wrap with DDP if using DDP
