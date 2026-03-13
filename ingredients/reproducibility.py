"""
---
name: Reproducibility
slug: reproducibility
description: Set random seeds and deterministic flags for reproducible training.
dependencies: []
conflicts: []
touches: [train.py, config/train.yaml]
tags: [debugging]
test_instructions: |
  - Test that two runs with the same seed produce identical losses after one epoch.
  - Test that two runs with different seeds produce different losses.
  - Test that torch.manual_seed, numpy seed, and random seed are all set.
  - Test that torch.backends.cudnn.deterministic is True when deterministic mode is on.
  - Test that DataLoader worker seeds are set (worker_init_fn).
---
"""

# Sets all random seeds for reproducibility.
#
# Changes to train.py:
#   - Call set_seed(cfg.seed) at the start of main()
#
# Changes to config/train.yaml:
#   + seed: 42
#   + deterministic: true

import os
import random
import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def worker_init_fn(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


# In main():
#
# set_seed(cfg.seed, cfg.deterministic)
#
# In DataLoader:
#
# DataLoader(..., worker_init_fn=worker_init_fn)
