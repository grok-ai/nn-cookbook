"""
---
name: DistributedDataParallel
slug: ddp
description: Multi-GPU training with PyTorch DDP via torchrun.
dependencies: []
conflicts: []
touches: [train.py, data.py, config/train.yaml]
tags: [distributed, performance]
test_instructions: |
  - Test that model is wrapped in DDP when world_size > 1.
  - Test that DistributedSampler is used for training data.
  - Test that sampler.set_epoch() is called each epoch for proper shuffling.
  - Test that only rank 0 prints and saves checkpoints.
  - Test that dist.barrier() is called before checkpoint loading.
  - Test that cleanup (dist.destroy_process_group) is called at the end.
  - Test that the training function works correctly in single-GPU mode (no DDP).
---
"""

# Adds DDP setup/cleanup and wraps model.
# Launch with: torchrun --nproc_per_node=N train.py
#
# Changes to train.py:
#   - setup_ddp() / cleanup_ddp() functions
#   - Wrap model in DDP
#   - Use DistributedSampler
#   - Guard prints/saves/wandb on rank 0
#
# Changes to data.py:
#   - Accept sampler parameter in get_dataloaders
#
# Changes to config/train.yaml:
#   + ddp: false
#   + backend: nccl

import os
import torch
import torch.distributed as dist


def setup_ddp(backend: str = "nccl") -> int:
    dist.init_process_group(backend=backend)
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp() -> None:
    dist.destroy_process_group()


def get_rank() -> int:
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def is_main_process() -> bool:
    return get_rank() == 0


def broadcast_early_stop(should_stop: bool, device: torch.device) -> bool:
    stop_tensor = torch.tensor([1 if should_stop else 0], device=device)
    dist.broadcast(stop_tensor, src=0)
    return stop_tensor.item() == 1


# In main():
#
# if cfg.ddp:
#     local_rank = setup_ddp(cfg.backend)
#     device = torch.device(f"cuda:{local_rank}")
#
# model = get_model(cfg).to(device)
# if cfg.ddp:
#     model = DDP(model, device_ids=[local_rank])
#
# # For data loading:
# train_sampler = DistributedSampler(train_set) if cfg.ddp else None
# train_loader = DataLoader(train_set, batch_size=cfg.batch_size,
#                           shuffle=(train_sampler is None), sampler=train_sampler)
#
# for epoch in range(cfg.epochs):
#     if train_sampler is not None:
#         train_sampler.set_epoch(epoch)
#     ...
#     if is_main_process():
#         print(...)
#         # save checkpoint, log to wandb, etc.
#
# if cfg.ddp:
#     cleanup_ddp()
