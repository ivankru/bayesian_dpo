---
name: repo-temporary-scripts
description: >-
  Places ad-hoc helper scripts used for calculations, verification, log parsing,
  or data checks under the repository `scripts/` directory. Use when the user
  asks to count, verify, aggregate, probe, or automate checks and the agent would
  otherwise write a throwaway script in an arbitrary location.
---

# Temporary and helper scripts in `scripts/`

## Rule

When you need a **small script** to satisfy a request (Python, shell, or similar)—for example to compute metrics, verify files, parse logs, sample datasets, or batch-check benchmarks—**write the file under** `scripts/` **at the project root** (`bayesian_dpo/scripts/`), not under `/tmp`, the home directory, or random paths inside unrelated packages.

## How to apply

1. **Path**: `scripts/<descriptive_name>.py` (or `.sh`, etc.). Use a clear, task-specific name (e.g. `summarize_eval_logs.py`, `check_checkpoint_sizes.sh`).
2. **Missing folder**: If `scripts/` does not exist, create it before writing the file.
3. **Scope**: One-off or “temporary” scripts still go here so they stay in the repo, are discoverable, and can be rerun or deleted in one place.
4. **Cleanup**: If the user clearly wants a single-use throwaway and no file on disk, you may run inline commands instead; if you do create a script file for the task, default to `scripts/`.

## Do not

- Put helper or one-off scripts under `utils/` (including `utils/scripts/`). They belong only in the repository root `scripts/`.
- Drop helper scripts next to unrelated modules without reason.
- Rely only on chat-fenced code when a saved script in `scripts/` would make the same work repeatable for the user.
