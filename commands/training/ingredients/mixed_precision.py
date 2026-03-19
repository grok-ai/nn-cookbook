"""
---
name: Mixed Precision (AMP)
slug: mixed_precision
description: Automatic mixed precision training with torch.amp. Supports bf16 (preferred) and fp16 with GradScaler.
dependencies: []
conflicts: []
touches: [train.py, config/train.yaml]
tags: [performance]
test_instructions: |
  - Test that forward pass runs inside autocast context (output dtype is float16 or bfloat16 on CUDA).
  - Test that GradScaler is only created when amp_dtype is fp16.
  - Test that bf16 path skips GradScaler and uses plain backward/step.
  - Test that validation also uses autocast for inference.
  - Test that training completes without NaN losses on a small synthetic dataset.
---
"""

# Wraps forward pass in autocast. GradScaler is only needed for fp16;
# bf16 has sufficient dynamic range and doesn't require loss scaling.
#
# Changes to train.py:
#   - Create GradScaler in main() only when amp_dtype == "fp16"
#   - train_epoch: wrap forward in autocast(dtype=amp_dtype),
#     use scaler path for fp16, plain backward/step for bf16
#   - validate: wrap forward in autocast
#
# Changes to config/train.yaml:
#   + amp: true
#   + amp_dtype: bf16          # bf16 (preferred, no GradScaler) | fp16

import torch

AMP_DTYPE_MAP = {"bf16": torch.bfloat16, "fp16": torch.float16}


def train_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    amp_dtype: str,
    scaler: torch.amp.GradScaler | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    dtype = AMP_DTYPE_MAP[amp_dtype]
    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device.type, dtype=dtype):
            output = model(x)
            loss = torch.nn.functional.cross_entropy(output, y)
        if scaler is not None:  # fp16 path
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:  # bf16 path — no scaling needed
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def validate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    amp_dtype: str,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    dtype = AMP_DTYPE_MAP[amp_dtype]
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast(device_type=device.type, dtype=dtype):
                output = model(x)
                total_loss += torch.nn.functional.cross_entropy(output, y).item()
            correct += (output.argmax(1) == y).sum().item()
            total += y.size(0)
    return total_loss / len(loader), correct / total


# In main():
#
# amp_dtype = cfg.amp_dtype  # "bf16" or "fp16"
# scaler = torch.amp.GradScaler() if amp_dtype == "fp16" else None
# ...
# train_loss = train_epoch(model, loader, optimizer, device, amp_dtype, scaler)
# val_loss, val_acc = validate(model, loader, device, amp_dtype)
