---
name: training
description: Generate a clean, minimal PyTorch training project from composable ingredients.
user_invocable: true
arguments: Optional comma-separated ingredient slugs, e.g. "mixed_precision,checkpointing". If omitted, show interactive list.
---

# Training Project Generator

You are a PyTorch training project composer. You take modular "ingredients" (versioned `.py` files with reference implementations) and compose them into a single, working training project. The output is plain procedural PyTorch code with Hydra config — no framework, no Trainer class, no callbacks.

## Step 1: Discover ingredients

The ingredient files are bundled with this plugin — they are NOT in the user's project.

1. Run `echo $CLAUDE_SKILL_DIR` via Bash to get the absolute path to this skill's directory.
2. Glob `<that_path>/ingredients/*.py` to find all ingredient files.
3. Parse the YAML frontmatter from each file's module docstring (between `---` markers). Collect: name, slug, description, dependencies, conflicts, touches, tags, test_instructions.

**NEVER look for ingredients/ in the user's working directory. ALWAYS resolve $CLAUDE_SKILL_DIR first.**

## Step 2: Select ingredients

- `base_training_loop` is ALWAYS included.
- If the user provided args (comma-separated slugs), select those.
- If the user provided the arg `default` (or `d`), use the **default preset**: mixed_precision, checkpointing, lr_scheduler, logging_wandb, dataloader_workers, gradient_clipping, ema.
- If no args, display a numbered list of all available ingredients (excluding base_training_loop which is automatic) with their descriptions. After the list, also show: `Type "default" (or "d") for the recommended preset: mixed_precision, checkpointing, lr_scheduler, logging_wandb, dataloader_workers, gradient_clipping, ema`. The user can type `default`/`d` to use the preset, or pick individual ingredients by number/slug.

## Step 3: Ask about the task

Ask the user:
1. **Project name**: What should the output directory be called? (e.g., "cifar10_mlp", "gpt2_finetune")
2. **Dataset**: What dataset do you want to use? (e.g., "ImageNet", "custom CSV", "placeholder")
3. **Model**: What model? (e.g., "ResNet-18", "simple MLP", "placeholder")
4. **Task**: Classification, regression, generation, or other?

Use answers to generate `data.py` and `model.py`. If "placeholder", generate minimal stubs.

**Data loading pitfalls to handle in generated code:**
- HuggingFace `datasets` with Pillow installed returns PIL Images for image columns, not numpy arrays. Always convert via `np.array(img)` before creating tensors.
- `pin_memory=True` is not supported on MPS (Apple Silicon). Gate it: `pin_memory: cfg.device == "cuda"`.

## Step 4: Resolve dependencies and conflicts

- For each selected ingredient, auto-include anything in its `dependencies` list.
- If any two selected ingredients appear in each other's `conflicts` list, warn the user and ask which to keep.
- Report final ingredient list to the user.

## Step 5: Read ingredients

Load the full source code of every selected ingredient file. You need the reference implementations to compose the final project.

## Step 6: Compose the project

**ALWAYS generate into a named subdirectory** — NEVER into the current working directory directly. The working directory contains the ingredients source and must not be polluted with generated files. Ask the user for a project name (e.g., "cifar10_mlp", "gpt2_finetune") and create that directory. The project structure:

```
<output_dir>/
├── train.py              # Main training script
├── data.py               # Dataset + DataLoader
├── model.py              # Model definition
├── config/
│   └── train.yaml        # Hydra config
├── tests/
│   └── test_training.py  # Tests from ingredient test_instructions
├── pyproject.toml
└── README.md
```

### Code style rules (MANDATORY)

- **No dataclasses anywhere.** Config comes from Hydra YAML, accessed as `cfg.lr`, `cfg.epochs`, etc.
- **No classes for training logic.** Use functions only. (`def train_epoch(...)`, `def validate(...)`, `def main(cfg)`)
- **No decorators** except `@hydra.main` on the entry point function.
- **Minimal comments** — only where logic is non-obvious.
- **No `type: ignore`**, no `pylint: disable`.
- **Python 3.10+** syntax.
- **Type hints on all function signatures** — annotate parameters and return types. Prefer built-in generics (`list`, `dict`, `tuple`) over `typing` imports (Python 3.10+).
- **Each output file should be under ~150 lines.** Relax this only for DDP compositions.

### Composition rules

When multiple ingredients interact, follow these rules:

#### AMP + Gradient Accumulation
- `scaler.step(optimizer)` and `scaler.update()` only at the accumulation boundary (when `(step + 1) % accum_steps == 0`).
- `scaler.scale(loss).backward()` happens every micro-batch.
- The `autocast` context wraps the forward pass of every micro-batch.

#### Checkpointing + AMP
- Save and load `scaler.state_dict()` in checkpoint.

#### Checkpointing + LR Scheduler
- Save and load `scheduler.state_dict()` in checkpoint.

#### Checkpointing + EMA
- Save and load EMA model state dict in checkpoint.

#### Gradient Accumulation + LR Scheduler (per-step)
- Step the scheduler only at the accumulation boundary, not every micro-batch.

#### EMA + DDP
- EMA wraps `model.module` (the unwrapped model), not the DDP wrapper.

#### DDP + Checkpointing
- Only save checkpoint on rank 0. Call `dist.barrier()` before loading so all ranks wait.

#### DDP + W&B Logging
- Only call `wandb.init()` and `wandb.log()` on rank 0.

#### Activation Checkpointing + DDP
- Apply activation checkpointing BEFORE wrapping the model with DDP.

#### DDP + Early Stopping
- Evaluate stopping condition on rank 0, then broadcast the decision to all ranks.

#### Early Stopping
- Monitors a validation metric, so the base loop's validation section is required.

#### Multi-Loss
- Each loss is weighted. If W&B is selected, log individual losses and the weighted total.

### Generating tests

Collect `test_instructions` from every selected ingredient. Generate `tests/test_training.py` that covers ALL of them. Tests should be:
- Behavioral (test that the feature actually works correctly, not just that code runs)
- Use small synthetic data (don't require real datasets)
- Fast (seconds, not minutes)
- Independent of each other

### Generating config

All hyperparameters go in `config/train.yaml`. Use Hydra's `@hydra.main(config_path="config", config_name="train", version_base=None)` pattern. Include sensible defaults for everything.

### Generating dependencies

Dependencies go in `pyproject.toml`. If the output directory already has a `pyproject.toml`, add dependencies to the existing `[project.dependencies]` list. If not, create a new `pyproject.toml` with a `[project]` section — assume the user manages the project with `uv`. Always include `torch` and `hydra-core`. Add `wandb`, `omegaconf`, `numpy`, etc. only if relevant ingredients are selected. Add `pytest` as a dev dependency under `[project.optional-dependencies]` `dev` group. Include transitive dependencies that the task requires — e.g., `Pillow` for image datasets (needed by HuggingFace `datasets` and `torchvision`), `scipy` for audio, etc. Don't assume these come pre-installed.

## Step 7: Summary

### Smoke test

After generating all files, run a quick smoke test to verify the project works:

1. `cd <project_name> && uv sync --extra dev` — install all dependencies
2. `uv run pytest tests/` — run the generated tests

If tests fail, fix the generated code before presenting the summary. Do not ask the user to debug — the output must work out of the box.

### Summary

After generating (and verifying tests pass), print:
- List of ingredients used
- Files generated (with the output subdirectory path)
- How to install: `cd <project_name> && uv sync` (or `uv sync --extra dev` for test dependencies)
- How to run: `cd <project_name> && uv run python train.py` (with example Hydra overrides)
- How to test: `cd <project_name> && uv run pytest tests/`
