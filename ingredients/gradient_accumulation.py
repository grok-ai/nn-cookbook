"""
---
name: Gradient Accumulation
slug: gradient_accumulation
description: Accumulates gradients over N micro-batches before stepping the optimizer.
dependencies: []
conflicts: []
touches: [train.py, config/train.yaml]
tags: [optimization]
test_instructions: |
  - Test that optimizer.step() is called every accum_steps batches, not every batch.
  - Test with accum_steps=1 produces identical results to no accumulation.
  - Test that loss is divided by accum_steps before backward.
  - Test gradient values after N micro-batches match a single large batch (within tolerance).
---
"""

# Modifies train_epoch to accumulate gradients.
#
# Changes to train.py:
#   - train_epoch accepts accum_steps parameter
#   - loss is divided by accum_steps before backward
#   - optimizer.zero_grad() at start and after each accumulation boundary
#   - optimizer.step() only at accumulation boundary
#
# Changes to config/train.yaml:
#   + accum_steps: 1

import torch


def train_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    accum_steps: int = 1,
) -> float:
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        output = model(x)
        loss = torch.nn.functional.cross_entropy(output, y)
        loss = loss / accum_steps
        loss.backward()
        if (batch_idx + 1) % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        total_loss += loss.item() * accum_steps
    # Handle leftover batches
    if (batch_idx + 1) % accum_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
    return total_loss / len(loader)
