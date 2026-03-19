"""
---
name: Early Stopping
slug: early_stopping
description: Stop training when validation metric stops improving.
dependencies: []
conflicts: []
touches: [train.py, config/train.yaml]
tags: [optimization, regularization]
test_instructions: |
  - Test that training stops after patience epochs without improvement.
  - Test that training continues when validation metric improves within patience window.
  - Test that counter resets to zero when a new best metric is found.
  - Test with patience=0 stops after the first epoch with no improvement.
  - Test that the best metric value is tracked correctly across epochs.
  - Test that early stopping works for both minimize (loss) and maximize (accuracy) modes.
---
"""

# Implements early stopping based on validation metric.
#
# Changes to train.py:
#   - Track best metric and patience counter in main loop
#   - Break from training loop when patience is exhausted
#
# Changes to config/train.yaml:
#   + early_stopping: true
#   + patience: 5
#   + es_mode: min        # min (loss) or max (accuracy)


def check_early_stopping(
    current_metric: float,
    best_metric: float,
    counter: int,
    patience: int,
    mode: str = "min",
) -> tuple[float, int, bool]:
    improved = False
    if mode == "min":
        improved = current_metric < best_metric
    else:
        improved = current_metric > best_metric

    if improved:
        return current_metric, 0, False  # new_best, reset counter, don't stop
    else:
        counter += 1
        stop = counter >= patience
        return best_metric, counter, stop


# In main():
#
# best_metric = float("inf") if cfg.es_mode == "min" else -float("inf")
# es_counter = 0
#
# for epoch in range(cfg.epochs):
#     train_loss = train_epoch(...)
#     val_loss, val_acc = validate(...)
#     metric = val_loss if cfg.es_mode == "min" else val_acc
#     if cfg.early_stopping:
#         best_metric, es_counter, should_stop = check_early_stopping(
#             metric, best_metric, es_counter, cfg.patience, cfg.es_mode
#         )
#         if should_stop:
#             print(f"Early stopping at epoch {epoch+1}")
#             break
