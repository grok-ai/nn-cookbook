"""
---
name: Base Training Loop
slug: base_training_loop
description: Core train/val loop with Hydra config. Supports step-level logging and inline validation.
dependencies: []
conflicts: []
touches: [train.py, data.py, model.py, config/train.yaml]
tags: [core]
test_instructions: |
  - Test that training loss decreases over multiple epochs on a small synthetic dataset.
  - Test that model parameters change after one training step.
  - Test that validation loop does not update model parameters (compare params before and after).
  - Test that the training function runs to completion with a minimal config (1 epoch, tiny dataset).
  - Test that Hydra config values are correctly passed through (e.g., lr, batch_size).
  - Test that log_every_n_steps produces step-level output at the correct frequency.
  - Test that val_every_n_steps triggers inline validation during training.
---
"""

# === train.py ===

import torch
import hydra
from omegaconf import DictConfig

from data import get_dataloaders
from model import get_model


def get_device(cfg_device: str = "auto") -> torch.device:
    """Resolve device string to torch.device with fallback: cuda > mps > cpu."""
    if cfg_device != "auto":
        return torch.device(cfg_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    cfg: DictConfig,
    val_loader: torch.utils.data.DataLoader | None = None,
    validate_fn=None,
) -> float:
    model.train()
    total_loss = 0.0
    log_every = getattr(cfg, "log_every_n_steps", 0)
    val_every = getattr(cfg, "val_every_n_steps", 0)
    global_step_offset = epoch * len(loader)

    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = torch.nn.functional.cross_entropy(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        step = batch_idx + 1
        global_step = global_step_offset + step

        if log_every and step % log_every == 0:
            avg = total_loss / step
            print(f"  [step {global_step}] loss: {loss.item():.4f} (avg: {avg:.4f})")

        if val_every and val_loader is not None and step % val_every == 0:
            vl, va = validate_fn(model, val_loader, device)
            print(f"  [step {global_step}] val_loss: {vl:.4f} val_acc: {va:.4f}")
            model.train()

    return total_loss / len(loader)


def validate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            total_loss += torch.nn.functional.cross_entropy(output, y).item()
            correct += (output.argmax(1) == y).sum().item()
            total += y.size(0)
    return total_loss / len(loader), correct / total


@hydra.main(config_path="config", config_name="train", version_base=None)
def main(cfg: DictConfig):
    device = get_device(cfg.device)

    train_loader, val_loader = get_dataloaders(cfg)
    model = get_model(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    for epoch in range(cfg.epochs):
        train_loss = train_epoch(
            model, train_loader, optimizer, device, epoch, cfg,
            val_loader=val_loader, validate_fn=validate,
        )
        val_loss, val_acc = validate(model, val_loader, device)
        print(f"Epoch {epoch+1}/{cfg.epochs} - train_loss: {train_loss:.4f} - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}")


if __name__ == "__main__":
    main()


# === config/train.yaml ===
#
# device: auto              # auto | cuda | mps | cpu
# epochs: 10
# lr: 1e-3
# batch_size: 64
# num_workers: 0
# log_every_n_steps: 0      # 0 = off, N = print loss every N steps
# val_every_n_steps: 0       # 0 = off (validate per-epoch only), N = also validate every N steps
