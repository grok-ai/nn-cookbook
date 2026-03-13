"""
---
name: Checkpointing
slug: checkpointing
description: Save and load full training state. Supports per-epoch and mid-epoch (every N steps) saves.
dependencies: []
conflicts: []
touches: [train.py, config/train.yaml]
tags: [reliability]
test_instructions: |
  - Test that save_checkpoint creates a file with model, optimizer, and epoch state.
  - Test that load_checkpoint restores model parameters exactly.
  - Test that load_checkpoint restores optimizer state (e.g., momentum buffers).
  - Test that training resumes from correct epoch after loading checkpoint.
  - Test that best_val_loss is preserved across save/load.
  - Test that checkpoint file is not created when save_checkpoint is not called.
  - Test that save_every_n_steps produces mid-epoch checkpoint files at correct intervals.
---
"""

# Adds save/load checkpoint functions.
#
# Changes to train.py:
#   + save_checkpoint(path, model, optimizer, epoch, best_val_loss, **extras)
#   + load_checkpoint(path, model, optimizer)  -> returns epoch, best_val_loss
#   - In main(): save after each epoch, optionally resume from checkpoint
#   - In train_epoch: if save_every_n_steps > 0, save mid-epoch checkpoints
#
# Changes to config/train.yaml:
#   + checkpoint_dir: checkpoints
#   + resume: null              # path to checkpoint to resume from
#   + save_every_n_steps: 0     # 0 = off, N = save checkpoint every N training steps

import os
import torch


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_loss: float,
    **extras,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_loss": best_val_loss,
    }
    state.update(extras)
    torch.save(state, path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[int, float]:
    checkpoint = torch.load(path, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint.get("epoch", 0), checkpoint.get("best_val_loss", float("inf"))


# In main():
#
# best_val_loss = float("inf")
# start_epoch = 0
# if cfg.resume:
#     start_epoch, best_val_loss = load_checkpoint(cfg.resume, model, optimizer)
#
# for epoch in range(start_epoch, cfg.epochs):
#     train_loss = train_epoch(...)
#     val_loss, val_acc = validate(...)
#     if val_loss < best_val_loss:
#         best_val_loss = val_loss
#         save_checkpoint(
#             os.path.join(cfg.checkpoint_dir, "best.pt"),
#             model, optimizer, epoch, best_val_loss,
#         )
#     save_checkpoint(
#         os.path.join(cfg.checkpoint_dir, "last.pt"),
#         model, optimizer, epoch, best_val_loss,
#     )
#
# Inside train_epoch (when save_every_n_steps > 0):
#
#     save_every = getattr(cfg, "save_every_n_steps", 0)
#     if save_every and (batch_idx + 1) % save_every == 0:
#         global_step = epoch * len(loader) + batch_idx + 1
#         save_checkpoint(
#             os.path.join(cfg.checkpoint_dir, f"step_{global_step}.pt"),
#             model, optimizer, epoch, best_val_loss,
#         )
