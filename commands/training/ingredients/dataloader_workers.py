"""
---
name: DataLoader Workers
slug: dataloader_workers
description: Multi-worker data loading with prefetch factor and persistent workers.
dependencies: []
conflicts: []
touches: [data.py, config/train.yaml]
tags: [performance]
test_instructions: |
  - Test that DataLoader is created with num_workers from config.
  - Test that prefetch_factor is set when num_workers > 0.
  - Test that persistent_workers is True when num_workers > 0.
  - Test that pin_memory is True when device is cuda.
  - Test that num_workers=0 still works (single-process loading).
---
"""

# Configures DataLoader for multi-worker loading.
#
# Changes to data.py:
#   - Pass num_workers, prefetch_factor, persistent_workers, pin_memory to DataLoader
#
# Changes to config/train.yaml:
#   + num_workers: 4
#   + prefetch_factor: 2
#   + pin_memory: true
#   + persistent_workers: true

from omegaconf import DictConfig


def get_dataloader_kwargs(cfg: DictConfig) -> dict[str, object]:
    kwargs = {
        "num_workers": cfg.num_workers,
        "pin_memory": cfg.pin_memory,
    }
    if cfg.num_workers > 0:
        kwargs["prefetch_factor"] = cfg.prefetch_factor
        kwargs["persistent_workers"] = cfg.persistent_workers
    return kwargs


# In get_dataloaders():
#
# dl_kwargs = get_dataloader_kwargs(cfg)
# train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, **dl_kwargs)
# val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False, **dl_kwargs)
