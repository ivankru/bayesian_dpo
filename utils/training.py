# -*- coding: utf-8 -*-
"""
Shared one-epoch DPO training loop and universal train_dpo (hard / soft / bayes modes).
"""
import atexit
import fcntl
import json
import math
import os
import re
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import mlflow
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import Dataset
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from config.base_config import (
    CAPABILITY_EVAL_BATCH_SIZE,
    CAPABILITY_EVAL_LIMIT,
    CAPABILITY_EVAL_MAX_NEW_TOKENS,
    CAPABILITY_EVAL_MAX_PROMPT_TOKENS,
    LOG_INTERVAL,
    LR_ALIGN_LOG_INTERVAL,
    MAX_FULL_LEN,
    MAX_PROMPT_LEN,
    P_PRED_TARGET_TEMPERATURE,
    USE_CHAT_TEMPLATE,
    VAL_ENTROPY_FORWARD_CHUNK_SIZE,
    VAL_ENTROPY_MAX_NEW_TOKENS,
    VAL_ENTROPY_MAX_PROMPTS,
    VAL_ENTROPY_NUM_SAMPLES,
    VAL_ENTROPY_PROMPT_BATCH_SIZE,
    VAL_KL_MC_NUM_SAMPLES,
    VAL_KL_MC_PROMPT_BATCH_SIZE,
)
from utils.datasets import precompute_p_pred_cached, precompute_p_pred_teacher
from utils.loss import get_loss, hard_dpo_loss
from utils.metrics import (
    EvalRow,
    aggregate_anchor_alignment_window,
    build_ref_cache_metadata,
    estimate_val_response_entropy,
    estimate_val_kl_mc,
    eval_pairwise_accuracy,
    eval_pairwise_nll,
    format_anchor_alignment_log,
    format_capability_retention_log_lines,
    load_eval_rows,
    load_ref_texts_cache_if_compatible,
    log_mlflow_capability_metrics,
    run_retention_eval_pair,
    save_ref_texts_cache,
)
from utils.probe_margins import (
    StepTracker,
    build_probe_loader,
    cache_ref_logps,
    load_or_create_probe_indices,
    snapshot_probe,
)
from utils.val_distributions import (
    compute_val_delta_distributions,
    log_train_diff_stats,
    log_val_diff_from_loader,
    summarize_margin,
)

DPO_MODE_CHOICES = ("hard", "soft", "bayes")
OPTIMIZER_CHOICES = ("adamw", "sgd")

# MC forward KL(π‖ref) from π_θ samples; min(N, len(val)) prompts. 0 in val_kl_mc_max_prompts disables.
DEFAULT_VAL_KL_MC_MAX_PROMPTS = 256

# Soft/Bayes-ADPO: half-epoch validation and lambda / p_pred_cached update.
# Large preference sets only (not orca_dpo / helpsteer3).
ULTRAFB_MID_EPOCH_DATASETS = frozenset(
    {"openbmb", "ultrafeedback_binarized", "ultrafeedback_soft", "hh_rlhf"}
)


def infer_run_root_from_checkpoint_dir(resume_checkpoint_dir: str) -> Optional[Path]:
    """
    Run root (parent dir with train.log): .../best, .../epochs/epoch_XXX, else parent.
    """
    p = Path(resume_checkpoint_dir).expanduser()
    try:
        p = p.resolve()
    except OSError:
        p = Path(resume_checkpoint_dir).expanduser()
    if not p.exists():
        return None
    name = p.name
    if name == "best":
        return p.parent
    if p.parent.name == "epochs" and re.match(r"^epoch_\d{3}$", name):
        return p.parent.parent
    # Run root (output_dir): adapter under best/, train.log alongside
    if (p / "best" / "adapter_config.json").is_file():
        return p
    return p.parent


def slice_train_log_lines_before_resume_start_epoch(
    lines: List[str],
    mode: str,
    resume_start_epoch_1based: int,
) -> List[str]:
    """
    train.log lines before epoch resume_start_epoch_1based (1-based).
    Soft/bayes: cut before line «=== Epoch S ===» or «=== Epoch S/E ===» (new format).
    Hard: cut before first line «[epoch S]» (training epoch S; format unchanged).
    """
    s = int(resume_start_epoch_1based)
    if s <= 1:
        return []
    if mode in ("soft", "bayes"):
        # Accept both legacy "=== Epoch S ===" and new "=== Epoch S/E ===".
        boundary = re.compile(rf"^\s*=== Epoch {s}(?:/\d+)? ===\s*$")
        for i, line in enumerate(lines):
            if boundary.match(line):
                return lines[:i]
        return []
    if mode == "hard":
        boundary = re.compile(rf"^\[epoch {int(s)}\]")
        for i, line in enumerate(lines):
            if boundary.match(line):
                return lines[:i]
        return []
    return []


def _lambda_label_at_progress(
    progress: float, lambda_min: float, lambda_schedule: str
) -> float:
    """progress in [0, 1]: same as main epoch loop (linear / cosine)."""
    progress = max(0.0, min(1.0, float(progress)))
    if lambda_schedule == "linear":
        return 1.0 - (1.0 - lambda_min) * progress
    return lambda_min + (1.0 - lambda_min) * (1.0 + math.cos(math.pi * progress)) / 2.0


def _lambda_schedule_progress(
    epoch_idx_0: int,
    epochs: int,
    lambda_full_epochs: int,
    mid_frac: float,
) -> float:
    """
    Fraction in [0, 1] passed to _lambda_label_at_progress. The map is symmetric
    epoch-to-epoch between k=0 and k>0: for the same (epochs, epoch_idx_0),
    epoch starts share the same progress in both modes if epoch 1 at k=0 is
    treated as implicit label-warmup and at k>0 as explicit k-epoch warmup. This
    enables:
      (a) fair k=0 vs k>0 comparisons for the same total epochs;
      (b) resume right after epoch k (resume_start_epoch_1based == k+1) to start
          with λ<1 on the first step instead of spending another epoch at λ=1.

    lambda_full_epochs == 0 (no warmup anchor):
        progress = (epoch_idx_0 + mid_frac) / (epochs - 1).
        First epoch (idx=0, mid=0): progress=0 → λ=1 (implicit label warmup);
        last (idx=epochs-1, mid=0): progress=1 → λ=lambda_min.

    lambda_full_epochs == k > 0 (warmup anchor: epochs 1..k labels only; end of
    epoch k fixes p_pred_teacher):
        warmup 1..k → progress = 0 → λ = 1;
        tail k+1..epochs (decay_epochs = epochs - k epochs):
            rel = (epoch_idx_0 - k) + mid_frac;
            progress = (rel + 1) / decay_epochs.
        Start of first tail epoch (rel=0): progress = 1/decay → λ<1 immediately.
        End of last (rel=decay-1, mid=0 on that epoch): progress = 1 → λ=lambda_min.

    Symmetry check for epochs=5:
        k=0: epoch starts → [0, 0.25, 0.5, 0.75, 1.0]
        k=1: epoch starts → [warmup=0, 0.25, 0.5, 0.75, 1.0]
        k=2: epoch starts → [warmup=0, warmup=0, 1/3, 2/3, 1.0]
    The tail schedule spans decay_epochs epochs so the first tail step is active
    (not duplicating warmup).

    Edge cases:
      - decay_epochs <= 0 (k >= epochs): no tail, all warmup → progress=0.
      - decay_epochs == 1 (single tail epoch): progress = 1 immediately
        (λ=lambda_min for that whole epoch).
      - epochs <= 1: trivial, 0.5 / 1.0 by mid_frac.

    Determined by (epoch_idx_0, epochs, k, mid_frac) — resume from any
    resume_start_epoch_1based yields the same λ as a continuous run with the same
    hyperparameters. Resuming exactly on the first tail epoch
    (resume_start_epoch_1based == k+1) recomputes p_pred_teacher from LOADED
    weights (see train_dpo) — equivalent to end of epoch k in a continuous run,
    and λ<1 applies from the first step after load.

    mid_frac: 0 — epoch start; 0.5 — mid-epoch (mid-epoch validation).
    """
    if epochs <= 1:
        return 0.5 if mid_frac > 0 else 1.0
    if lambda_full_epochs <= 0:
        return min(1.0, (epoch_idx_0 + mid_frac) / (epochs - 1))
    f = int(lambda_full_epochs)
    if epoch_idx_0 < f:
        return 0.0
    decay_epochs = epochs - f
    if decay_epochs <= 0:
        return 0.0
    if decay_epochs == 1:
        return 1.0
    rel = (epoch_idx_0 - f) + mid_frac
    return min(1.0, max(0.0, (rel + 1) / decay_epochs))


def _val_resp_entropy_vocab_nats_max(tokenizer, policy_model) -> Tuple[int, float]:
    """
    V and log(V) in nats: upper bound on one-step entropy under uniform softmax
    over the full vocabulary (as in estimate_val_response_entropy).
    """
    cfg = getattr(policy_model, "config", None)
    v = getattr(cfg, "vocab_size", None) if cfg is not None else None
    if not isinstance(v, int) or v <= 0:
        v = getattr(tokenizer, "vocab_size", None)
    if not isinstance(v, int) or v <= 0:
        v = len(tokenizer)
    v = int(v)
    log_v = math.log(float(v)) if v > 1 else float("nan")
    return v, log_v


def _log_val_response_entropy_two_lines(
    log_msg: Callable[..., None],
    ent_stats: Dict[str, float],
    tokenizer,
    policy_model,
    *,
    l_tokens: int,
    n_prompts: int,
    num_samples: int,
) -> None:
    """Two log lines: absolute nats (max = log V) and the same stats as % of max."""
    v, log_v = _val_resp_entropy_vocab_nats_max(tokenizer, policy_model)
    hdr = (
        "validation response entropy "
        f"(L={l_tokens}, {n_prompts} prompts × {num_samples})"
    )
    m = float(ent_stats["mean"])
    med = float(ent_stats["median"])
    p10 = float(ent_stats["p10"])
    p90 = float(ent_stats["p90"])
    if math.isfinite(log_v) and log_v > 0:
        inv_pct = 100.0 / log_v
        log_msg(
            f"{hdr} — abs (nats): mean={m:.4f} median={med:.4f} p10={p10:.4f} p90={p90:.4f} "
            f"(max uniform = log V = {log_v:.4f}, V={v})"
        )
        log_msg(
            f"{hdr} — % of max: mean={m * inv_pct:.2f}% median={med * inv_pct:.2f}% "
            f"p10={p10 * inv_pct:.2f}% p90={p90 * inv_pct:.2f}%"
        )
    else:
        log_msg(
            f"{hdr} : mean={m:.4f} median={med:.4f} p10={p10:.4f} p90={p90:.4f}"
        )


def _fmt_seconds(seconds: float) -> str:
    s = max(0.0, float(seconds))
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = s - h * 3600 - m * 60
    return f"{h}h {m:02d}m {sec:05.2f}s"


def _make_shuffled_train_loader(
    ds,
    collate_fn,
    batch_size: int,
    generator: torch.Generator,
) -> DataLoader:
    """Training DataLoader with shuffle via the given torch.Generator.

    Single construction site: all restarts after precompute_p_pred_* share the
    same `generator` (continues RNG from seed), num_workers=0 for deterministic
    batch order.
    """
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        generator=generator,
    )


def _make_ordered_loader(
    ds,
    collate_fn,
    batch_size: int,
) -> DataLoader:
    """DataLoader without shuffle for validation and fixed epoch splits."""
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )


def _build_loss_spec(
    mode: str,
    beta: float,
    use_chat_template: bool,
    p_pred_target_temperature: float,
    soft_loss_type: str,
) -> Tuple[Callable, Callable, Dict[str, Any], str]:
    """Return (train_collate, train_loss_fn, loss_kwargs, mode_label) for the mode.

    Pure function; mode must already be validated against DPO_MODE_CHOICES.
    """
    if mode == "hard":
        return (
            collate_fn_hard,
            hard_dpo_loss,
            {"beta": beta, "use_chat_template": use_chat_template},
            "Hard DPO",
        )
    use_bayes = mode == "bayes"
    soft_loss_by_type = {
        "classic": "soft_dpo_classic_loss",
        "approximation": "soft_dpo_approximation_loss",
        "centered_softplus": "soft_dpo_centered_softplus_loss",
    }
    try:
        soft_loss_name = soft_loss_by_type[soft_loss_type]
    except KeyError as exc:
        available = ", ".join(sorted(soft_loss_by_type))
        raise ValueError(
            f"Unknown soft_loss_type {soft_loss_type!r}. Available: {available}"
        ) from exc
    soft_loss_fn = get_loss(soft_loss_name)
    soft_mode_label = "Bayes DPO" if use_bayes else "Soft DPO"
    return (
        collate_fn_soft,
        soft_loss_fn,
        {
            "beta": beta,
            "use_bayes": use_bayes,
            "use_chat_template": use_chat_template,
            "p_pred_target_temperature": p_pred_target_temperature,
        },
        f"{soft_mode_label} ({soft_loss_type})",
    )


def _epoch_lambda_and_loss_kw(
    g0: int,
    epochs: int,
    lambda_full_epochs: int,
    lambda_min: float,
    lambda_schedule: str,
    has_teacher_column: bool,
    base_loss_kwargs: Dict[str, Any],
) -> Tuple[Dict[str, Any], float, bool, float]:
    """Build loss_kwargs at epoch start (g0, 1-based = g0+1) for soft/bayes.

    Returns (epoch_loss_kw, lambda_label_epoch, has_teacher_anchor, teacher_blend_w).
    Used outside the hard branch.
    """
    progress_epoch = _lambda_schedule_progress(
        g0, epochs, lambda_full_epochs, 0.0
    )
    lambda_label_epoch = _lambda_label_at_progress(
        progress_epoch, lambda_min, lambda_schedule
    )
    has_teacher_anchor = lambda_full_epochs > 0 and has_teacher_column
    teacher_blend_w = 0.5 if has_teacher_anchor else 0.0
    epoch_loss_kw = {
        **base_loss_kwargs,
        "lambda_label": lambda_label_epoch,
        "p_pred_teacher_blend": teacher_blend_w,
    }
    return epoch_loss_kw, lambda_label_epoch, has_teacher_anchor, teacher_blend_w


def _validate_train_dpo_args(
    mode: str,
    epochs: int,
    lambda_min: float,
    lambda_schedule: str,
    lambda_full_epochs: int,
    p_pred_target_temperature: float,
    resume_start_epoch_1based: int,
    resume_rewarmup_steps: int,
    resume_rewarmup_lr_floor: float,
    grad_clip_norm: float,
    optimizer_name: str,
) -> None:
    """Validate train_dpo arguments. Single failure point with clear errors."""
    if mode not in DPO_MODE_CHOICES:
        raise ValueError(f"mode must be one of {DPO_MODE_CHOICES}, got: {mode!r}")
    if not 0.0 <= lambda_min <= 1.0:
        raise ValueError(f"lambda_min must be in [0, 1], got {lambda_min!r}")
    if lambda_schedule not in ("linear", "cosine"):
        raise ValueError(
            f"lambda_schedule must be one of ('linear', 'cosine'), got {lambda_schedule!r}"
        )
    if lambda_full_epochs < 0:
        raise ValueError(f"lambda_full_epochs must be >= 0, got {lambda_full_epochs!r}")
    if p_pred_target_temperature <= 0:
        raise ValueError(
            f"p_pred_target_temperature must be > 0, got {p_pred_target_temperature!r}"
        )
    if resume_start_epoch_1based < 1:
        raise ValueError(
            f"resume_start_epoch_1based must be >= 1, got {resume_start_epoch_1based!r}"
        )
    if resume_rewarmup_steps < 0:
        raise ValueError(
            f"resume_rewarmup_steps must be >= 0, got {resume_rewarmup_steps!r}"
        )
    if not 0.0 <= resume_rewarmup_lr_floor <= 1.0:
        raise ValueError(
            f"resume_rewarmup_lr_floor must be in [0, 1], got {resume_rewarmup_lr_floor!r}"
        )
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs!r}")
    if resume_start_epoch_1based > epochs:
        raise ValueError(
            f"resume_start_epoch_1based={resume_start_epoch_1based} must be <= epochs={epochs}"
        )
    if grad_clip_norm < 0:
        raise ValueError(f"grad_clip_norm must be >= 0, got {grad_clip_norm!r}")
    if optimizer_name.lower() not in OPTIMIZER_CHOICES:
        raise ValueError(
            f"optimizer_name must be one of {OPTIMIZER_CHOICES}, got {optimizer_name!r}"
        )


def _gpu_peak_memory_gb(device: torch.device) -> Optional[float]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    try:
        idx = device.index if device.index is not None else torch.cuda.current_device()
        bytes_peak = torch.cuda.max_memory_allocated(idx)
    except Exception:
        return None
    return float(bytes_peak) / (1024.0**3)


def _reset_cuda_peak_memory_stats(device: torch.device) -> None:
    if device.type != "cuda" or not torch.cuda.is_available():
        return
    try:
        idx = device.index if device.index is not None else torch.cuda.current_device()
        torch.cuda.reset_peak_memory_stats(idx)
    except Exception:
        pass


def _cuda_empty_cache(device: torch.device) -> None:
    if device.type != "cuda" or not torch.cuda.is_available():
        return
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


def _fmt_mem_gb(mem_gb: Optional[float]) -> str:
    if mem_gb is None:
        return "n/a"
    return f"{mem_gb:.2f} GB"


@contextmanager
def _mlflow_training_context(
    enabled: bool,
    experiment: str,
    run_name: Optional[str],
    tracking_uri: Optional[str],
    params: Dict[str, Any],
    log_path: str,
) -> Iterator[None]:
    if not enabled:
        yield
        return
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    mlflow.start_run(run_name=run_name)
    try:
        to_log = {k: str(v) for k, v in params.items() if v is not None}
        mlflow.log_params(to_log)
        yield
    finally:
        if os.path.isfile(log_path):
            try:
                mlflow.log_artifact(log_path)
            except OSError:
                pass
        mlflow.end_run()


def collate_fn_hard(examples: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    return {
        "prompt": [e["prompt"] for e in examples],
        "chosen": [e["chosen"] for e in examples],
        "rejected": [e["rejected"] for e in examples],
    }


def collate_fn_soft(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "prompt": [e["prompt"] for e in examples],
        "resp1": [e["resp1"] for e in examples],
        "resp2": [e["resp2"] for e in examples],
        "p": [e["p"] for e in examples],
        "p_bayes": [e["p_bayes"] for e in examples],
    }
    if examples and "p_pred_cached" in examples[0]:
        out["p_pred_cached"] = [e["p_pred_cached"] for e in examples]
    if examples and "p_pred_teacher" in examples[0]:
        out["p_pred_teacher"] = [e["p_pred_teacher"] for e in examples]
    return out


def train_one_epoch_dpo(
    train_loader_box: List[DataLoader],
    tokenizer,
    policy_model,
    ref_model,
    device: str,
    loss_fn: Callable[..., Any],
    optimizer,
    scheduler,
    epoch_1based: int,
    global_step: int,
    loss_kw: Dict[str, Any],
    grad_clip_norm: float = 0.0,
    log=print,
    use_mlflow: bool = False,
    mid_epoch_hook: Optional[Callable[[int], None]] = None,
    step_tracker: Optional[StepTracker] = None,
    after_step_hook: Optional[Callable[[int, int], None]] = None,
) -> int:
    """
    One DPO epoch. loss_fn(..., **loss_kw) returns (loss, kl_approx).
    loss_kw may be mutated in-place (e.g. lambda_label after mid_epoch_hook).

    train_loader_box:
      - length 1 — single DataLoader for the whole epoch (mid_epoch_hook ignored);
      - length 2 — [first_half_loader, second_half_placeholder]: iterate the first
        loader, call mid_epoch_hook(global_step) once (it must place the second
        DataLoader in train_loader_box[1]), then iterate the second.
        This gives 100% epoch coverage with disjoint samples.

    epoch_1based: 1-based epoch index for train log lines only.
    """
    policy_model.train()
    running_loss = 0.0
    running_kl = 0.0
    running_grad_abs_mean = 0.0
    running_grad_norm = 0.0
    log_interval = int(LOG_INTERVAL)
    # LR line and aggregated align metrics at the same cadence (accumulated over the interval).
    lr_align_log_interval = int(LR_ALIGN_LOG_INTERVAL)
    align_gap_parts: List[np.ndarray] = []
    align_ts_parts: List[np.ndarray] = []
    train_diff_parts: List[np.ndarray] = []
    running_diff_parts: List[np.ndarray] = []

    def flush_align_log() -> None:
        if not align_ts_parts:
            return
        align_m = aggregate_anchor_alignment_window(align_gap_parts, align_ts_parts)
        log(format_anchor_alignment_log(align_m))
        if use_mlflow:
            for k, v in align_m.items():
                if v == v:  # not NaN
                    mlflow.log_metric(f"train_{k}", v, step=global_step)
        align_gap_parts.clear()
        align_ts_parts.clear()

    def process_batch(batch) -> None:
        nonlocal global_step, running_loss, running_kl, running_grad_abs_mean, running_grad_norm
        optimizer.zero_grad(set_to_none=True)
        out = loss_fn(batch, tokenizer, policy_model, ref_model, device, **loss_kw)
        if len(out) == 3:
            loss, kl_batch, soft_diag = out
            if isinstance(soft_diag, dict):
                ts = soft_diag.get("target_shift")
                if ts is not None and ts.size:
                    align_ts_parts.append(ts)
                ga = soft_diag.get("gap_abs")
                if ga is not None and ga.size:
                    align_gap_parts.append(ga)
                diff_arr = soft_diag.get("diff")
                if diff_arr is not None and np.asarray(diff_arr).size:
                    arr = np.asarray(diff_arr, dtype=np.float64)
                    train_diff_parts.append(arr)
                    running_diff_parts.append(arr)
        else:
            loss, kl_batch = out
        loss.backward()
        # Trainable grads only (frozen base has grad=None). Same L2 as TRL train/grad_norm:
        # ||∇_θ L||_2 = sqrt(Σ_j a_j^2), a = ∇_θ L. Measured before clip_grad_norm_.
        grad_abs_sum = 0.0
        grad_sq_sum = 0.0
        grad_numel = 0
        for p in policy_model.parameters():
            if p.grad is None:
                continue
            g = p.grad.detach()
            grad_abs_sum += g.abs().sum().item()
            grad_sq_sum += g.pow(2).sum().item()
            grad_numel += g.numel()
        grad_abs_mean = grad_abs_sum / max(1, grad_numel)
        grad_norm = math.sqrt(grad_sq_sum)
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                policy_model.parameters(), grad_clip_norm
            )
        # η used by this update is the current group lr; scheduler.step() after
        # optimizer.step() advances lr for the *next* batch.
        lr_used = float(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
        if step_tracker is not None:
            step_tracker.H_cum += lr_used
            step_tracker.last_lr = lr_used
            step_tracker.last_grad_norm = float(grad_norm)

        running_loss += loss.item()
        running_kl += kl_batch
        running_grad_abs_mean += grad_abs_mean
        running_grad_norm += grad_norm
        global_step += 1

        if global_step % lr_align_log_interval == 0:
            lr_cur = optimizer.param_groups[0]["lr"]
            log(f"[epoch {epoch_1based} step {global_step}] lr={lr_cur:.2e}")
            if use_mlflow:
                mlflow.log_metric("lr", lr_cur, step=global_step)
            flush_align_log()
        if global_step % log_interval == 0:
            n = log_interval
            if running_diff_parts:
                window_diff = np.concatenate(running_diff_parts)
                train_diff_mean = float(np.mean(window_diff))
                train_diff_std = float(np.std(window_diff))
                # Per-pair σ(−β Δ_i); mean/std over pairs (not σ of the mean).
                beta_w = float(loss_kw.get("beta", 0.0))
                neg_beta_delta = torch.as_tensor(
                    window_diff, dtype=torch.float32
                ).mul_(-beta_w)
                s_i = torch.sigmoid(neg_beta_delta).numpy()
                sigmoid_mean = float(np.mean(s_i))
                sigmoid_std = float(np.std(s_i))
            else:
                train_diff_mean = float("nan")
                train_diff_std = float("nan")
                sigmoid_mean = float("nan")
                sigmoid_std = float("nan")
            log(
                f"[epoch {epoch_1based} step {global_step}] "
                f"loss={running_loss / n:.4f} "
                f"logp_gap_mean={running_kl / n:.4f} "
                f"train_diff_mean={train_diff_mean:.4f} "
                f"train_diff_std={train_diff_std:.4f} "
                f"sigmoid_mean={sigmoid_mean:.4f} "
                f"sigmoid_std={sigmoid_std:.4f} "
                f"grad_abs_mean={running_grad_abs_mean / n:.6e} "
                f"grad_norm={running_grad_norm / n:.6e}"
            )
            if use_mlflow:
                mlflow.log_metric("loss", running_loss / n, step=global_step)
                mlflow.log_metric("logp_gap_mean", running_kl / n, step=global_step)
                mlflow.log_metric("train_diff_mean", train_diff_mean, step=global_step)
                mlflow.log_metric("train_diff_std", train_diff_std, step=global_step)
                mlflow.log_metric("sigmoid_mean", sigmoid_mean, step=global_step)
                mlflow.log_metric("sigmoid_std", sigmoid_std, step=global_step)
                mlflow.log_metric(
                    "grad_abs_mean", running_grad_abs_mean / n, step=global_step
                )
                mlflow.log_metric(
                    "grad_norm", running_grad_norm / n, step=global_step
                )
            running_loss = 0.0
            running_kl = 0.0
            running_grad_abs_mean = 0.0
            running_grad_norm = 0.0
            running_diff_parts.clear()

        if after_step_hook is not None:
            after_step_hook(global_step, epoch_1based)

    split_mid = mid_epoch_hook is not None and len(train_loader_box) == 2

    if not split_mid:
        loader = train_loader_box[0]
        for batch in loader:
            process_batch(batch)
        flush_align_log()
    else:
        first_loader = train_loader_box[0]
        for batch in first_loader:
            process_batch(batch)

        mid_epoch_hook(global_step)

        second_loader = train_loader_box[1]
        if second_loader is None:
            raise RuntimeError(
                "mid_epoch_hook must place the second DataLoader in train_loader_box[1]"
            )
        for batch in second_loader:
            process_batch(batch)

        flush_align_log()

    if train_diff_parts:
        train_diff_stats = summarize_margin(np.concatenate(train_diff_parts))
        log_train_diff_stats(train_diff_stats, log, epoch_1based)
        if use_mlflow:
            mlflow.log_metric("train_diff_mean", train_diff_stats["mean"], step=global_step)
            mlflow.log_metric("train_diff_std", train_diff_stats["std"], step=global_step)
            mlflow.log_metric(
                "train_diff_median", train_diff_stats["median"], step=global_step
            )
    else:
        log(f"[epoch {epoch_1based}] train_diff (margin): (no samples)")

    return global_step


def _read_train_lock_pid(lock_path: str) -> Optional[int]:
    try:
        with open(lock_path, "r", encoding="utf-8") as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def acquire_output_dir_lock(output_dir: str) -> int:
    """
    Exclusive flock on ``{output_dir}/.train.lock``.

    Two jobs with the same RUN_NAME share this directory; without a lock they
    interleave train.log / best/ and look like a crash. Raises RuntimeError if
    another process already holds the lock.
    """
    lock_path = os.path.join(output_dir, ".train.lock")
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        other = _read_train_lock_pid(lock_path)
        extra = f" pid={other}" if other is not None else ""
        raise RuntimeError(
            f"Refusing to start: another training process{extra} already uses "
            f"{output_dir} (lock {lock_path}). Do not launch two jobs into the "
            "same RUN_NAME / --output-dir."
        ) from None
    os.ftruncate(fd, 0)
    os.write(fd, f"{os.getpid()}\n".encode())
    os.fsync(fd)
    return fd


def release_output_dir_lock(fd: Optional[int], output_dir: str) -> None:
    if fd is None:
        return
    lock_path = os.path.join(output_dir, ".train.lock")
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
    except OSError:
        pass
    try:
        os.unlink(lock_path)
    except OSError:
        pass


# output_dir -> flock fd; so train_dpo is a no-op if the CLI already locked.
_held_output_dir_locks: Dict[str, int] = {}


def ensure_output_dir_lock(output_dir: str) -> None:
    """
    Take ``.train.lock`` once per process, before loading the model.

    If the parent ``universal.sh`` already holds the lock
    (``SOFT_DPO_TRAIN_LOCK_HELD=1``), do nothing — a second flock on the same
    file would fail even though this is the same job.
    """
    if os.environ.get("SOFT_DPO_TRAIN_LOCK_HELD") == "1":
        return
    key = os.path.abspath(output_dir)
    if key in _held_output_dir_locks:
        return
    os.makedirs(output_dir, exist_ok=True)
    fd = acquire_output_dir_lock(output_dir)
    _held_output_dir_locks[key] = fd
    atexit.register(release_output_dir_lock, fd, output_dir)


def train_dpo(
    train_ds: Dataset,
    val_ds: Dataset,
    tokenizer,
    policy_model,
    ref_model,
    device: str | torch.device,
    mode: str = "hard",
    epochs: int = 1,
    batch_size: int = 8,
    lr: float = 5e-6,
    beta: float = 0.2,
    alpha: float = 1.0,
    output_dir: str = "checkpoints/dpo",
    num_training_steps_override: Optional[int] = None,
    dataset_name: Optional[str] = None,
    model_name: Optional[str] = None,
    lambda_min: float = 1.0,
    lambda_schedule: str = "linear",
    lambda_full_epochs: int = 0,
    p_pred_target_temperature: float = P_PRED_TARGET_TEMPERATURE,
    soft_loss_type: str = "classic",
    seed: int = 42,
    label_noise_prob: Optional[float] = None,
    use_chat_template: bool = USE_CHAT_TEMPLATE,
    log=print,
    use_mlflow: bool = False,
    mlflow_experiment: str = "bayesian_dpo",
    mlflow_run_name: Optional[str] = None,
    mlflow_tracking_uri: Optional[str] = None,
    val_kl_mc_max_prompts: int = DEFAULT_VAL_KL_MC_MAX_PROMPTS,
    val_kl_mc_num_samples: int = VAL_KL_MC_NUM_SAMPLES,
    val_kl_mc_max_new_tokens: int = 128,
    val_kl_mc_prompt_batch_size: int = VAL_KL_MC_PROMPT_BATCH_SIZE,
    val_entropy_max_prompts: int = VAL_ENTROPY_MAX_PROMPTS,
    val_entropy_num_samples: int = VAL_ENTROPY_NUM_SAMPLES,
    val_entropy_max_new_tokens: int = VAL_ENTROPY_MAX_NEW_TOKENS,
    # Conservative defaults for one A100 80GB with ref + KL_MC: larger values risk OOM on full-length forward.
    val_entropy_prompt_batch_size: int = VAL_ENTROPY_PROMPT_BATCH_SIZE,
    val_entropy_forward_chunk_size: int = VAL_ENTROPY_FORWARD_CHUNK_SIZE,
    val_distributions_max_batches: Optional[int] = None,
    capability_eval_dir: Optional[str] = None,
    capability_eval_limit: Optional[int] = CAPABILITY_EVAL_LIMIT,
    capability_eval_max_new_tokens: int = CAPABILITY_EVAL_MAX_NEW_TOKENS,
    capability_eval_batch_size: int = CAPABILITY_EVAL_BATCH_SIZE,
    capability_eval_max_prompt_tokens: int = CAPABILITY_EVAL_MAX_PROMPT_TOKENS,
    capability_ref_cache_path: Optional[str] = None,
    resume_start_epoch_1based: int = 1,
    resume_checkpoint_dir: Optional[str] = None,
    resume_rewarmup_steps: int = 50,
    resume_rewarmup_lr_floor: float = 0.0,
    grad_clip_norm: float = 0.0,
    optimizer_name: str = "AdamW",
    save_epoch_checkpoints: bool = True,
    probe_margins: bool = False,
    probe_size: int = 256,
    probe_every: int = 100,
    probe_seed: int = 0,
):
    """
    Universal DPO loop: hard, soft, or bayes.

    mode: "hard" — train/val chosen/rejected, loss = hard_dpo_loss.
          "soft" — train resp1, resp2, p, p_bayes; val chosen/rejected; train loss = soft_dpo_classic_loss(use_bayes=False).
          "bayes" — like soft, train loss = soft_dpo_classic_loss(use_bayes=True).

    epochs: planned epoch count (for λ and linear LR on scale 1..epochs). Training runs
        resume_start_epoch_1based, …, epochs (inclusive). Fresh start: resume_start_epoch_1based=1.
    val_ds is always chosen/rejected; validation uses hard DPO loss, NLL, accuracy.
    num_training_steps_override: for soft/bayes, override step count (e.g. hard train size) to align LR schedule.
    lambda_min: for soft/bayes, lower bound on lambda_label per epoch (blend with p_pred); 1.0 matches legacy behavior.
    lambda_full_epochs: for soft/bayes, k (1-based): epochs 1..k labels only (λ=1); end of epoch k fixes
        p_pred_teacher (σ(beta*diff) without T). From epoch k+1, λ<1 immediately on the tail schedule (decay=epochs-k
        epochs): first tail progress = 1/decay, last = 1. Epoch alignment matches
        lambda_full_epochs=0 (epoch n at k=0 ≡ epoch n at k>0 in progress if epoch 1 at
        k=0 counts as implicit label warmup). Useful on resume with
        resume_start_epoch_1based == k+1 (weights after epoch k): λ<1 from the first step
        after load — no extra epoch at λ=1. While train_ds has p_pred_teacher, at λ<1
        always w=0.5: 0.5*p_pred_teacher + 0.5*σ((beta*diff)/T).
        0 — no warmup anchor: λ schedule from epoch 1, p_pred_cached recomputed every step.
    p_pred_target_temperature: T>0 for σ((beta*diff)/T) in anchor mode (see utils.losses.classic.soft_dpo_classic_loss); unused if lambda_full_epochs=0.
    soft_loss_type: train loss for mode in {"soft","bayes"}:
        "classic" -> soft_dpo_classic_loss,
        "approximation" -> soft_dpo_approximation_loss,
        "centered_softplus" -> soft_dpo_centered_softplus_loss.
    seed: fixes train DataLoader shuffle (torch.Generator + num_workers=0).
    label_noise_prob: label noise when building soft train (--label-noise-prob); not used for hard (log shows N/A).
    use_chat_template: if True, get_logps uses tokenizer.apply_chat_template (Qwen-Instruct); else plain prompt\\nresponse (default: config.base_config.USE_CHAT_TEMPLATE).
    use_mlflow: log params, metrics, and train.log to MLflow (URI from mlflow_tracking_uri or env default).
    val_kl_mc_max_prompts: if >0 (default DEFAULT_VAL_KL_MC_MAX_PROMPTS), after each epoch val compute MC KL(π‖ref) from π_θ samples
          on first min(N, len(val)) val prompts (see utils.metrics.estimate_val_kl_mc); log val_kl_mc, MLflow if use_mlflow. 0 disables.
    val_kl_mc_num_samples: independent generations per prompt for MC.
    val_entropy_max_prompts: if >0, after each epoch val compute mean per-token response entropy of policy
          on first min(L, T_resp) tokens, aggregated over prompts; 0 disables.
    val_entropy_num_samples: independent generations per prompt for entropy.
    val_entropy_max_new_tokens: L, cap on first response tokens for entropy.
    val_entropy_prompt_batch_size: val prompts per generate (× num_samples parallel chains).
    val_entropy_forward_chunk_size: microbatch for full forward on generated seq (lowers VRAM peak).
    val_distributions_max_batches: if set (>0), after main val metrics compute distributions
        delta_theta, delta_ref, diff on first N val batches; log, MLflow, np.savez_compressed under output_dir.
    capability_eval_dir: if set (dir with knowledge/*.jsonl and reasoning/*.jsonl), each validation
        and at start (epoch init) run capability retention: ref vs policy on gold; log + JSON under output_dir;
        ref answers cached after first generation. MLflow: val_cap_*.
    capability_eval_limit: cap number of examples (first N in file order).
    capability_eval_max_new_tokens / capability_eval_batch_size / capability_eval_max_prompt_tokens: generation knobs.
    capability_ref_cache_path: path to JSON cache of ref answers for retention.
        If unset, use {capability_eval_dir}/ref_cache/<safe_model_name>_ref_texts.json.
    resume_rewarmup_steps: on resume (g0_start>0), first N steps after restart ramp
        lr again (extra factor on top of main schedule) linearly from
        resume_rewarmup_lr_floor to 1.0. Optimizer is recreated from scratch on each
        resume (moments=0), so full lr on step 0 can be unstable — this re-warmup smooths it. 0 disables.
    resume_rewarmup_lr_floor: minimum lr fraction at resume (0.0 — ramp from zero over N steps; 0.05 — start at 5%).
    grad_clip_norm: max L2 grad norm. 0 — no clip_grad_norm_ (default).
    optimizer_name: policy optimizer: "AdamW" (default) or "SGD".
    save_epoch_checkpoints: if True (default), write LoRA weights to epochs/epoch_XXX after
        each full epoch. If False, only ``best/`` is saved (on val NLL improvement).
        ``best/`` is written as soon as pairwise val NLL is known, before KL-MC /
        response entropy / capability-retention generate (those can take ~1h and
        are the usual kill point on the last epoch). Resume from a mid-run epoch
        then requires ``best/`` or an external copy of weights.
    probe_margins: if True, every ``probe_every`` steps evaluate Δ on a fixed val
        subset. Orca uses committed ``utils/probe_indices_orca_dpo.py``
        (256 pairs; not resampled from ``probe_seed``). Other datasets sample
        once with ``probe_size`` / ``probe_seed`` (not the training seed) and
        lock the result in the run dir. Ref logps are cached once. Writes
        ``probe_indices.json``, ``probe_ref_logps.npz``, ``probe_margins.jsonl``,
        and full-val Δ ``val_deltas/epoch_*.npz`` at each validation (same
        forward as val_diff). jsonl also stores this-step lr, H_cum=Ση, and
        endpoint grad_norm.
    probe_size / probe_every / probe_seed: cadence plus subset size/RNG when
        there is no canonical file (default 256 / 100 / 0). Orca ignores
        ``probe_size`` / ``probe_seed`` in favor of the frozen index list.
    resume_start_epoch_1based: first epoch of this run (1-based, as in logs and epochs/epoch_XXX).
        Require 1 <= resume_start_epoch_1based <= epochs. Checkpoint weights are after epoch N-1
        (e.g. after epoch_003 pass 4).         If N>1, first validation is full val as after epoch (N-1): same header/tags
        as end-of-epoch; best_val_nll initialized from that NLL (stitched with prior train.log).
    resume_checkpoint_dir: checkpoint path as in --resume (best or epochs/epoch_XXX); train.log is searched nearby.
        If resume_start_epoch_1based>1 and a non-empty prefix through epoch S exists, lines are prepended to train.log
        in output_dir (before this run's log).
        For soft/bayes with lambda_full_epochs=k>0: if resume_start_epoch_1based==k+1 (first tail step),
        precompute_p_pred_teacher runs on loaded weights before the epoch loop — same teacher fix
        as end of epoch k in a continuous run (when train_ds lacks p_pred_teacher column).
    For soft/bayes on openbmb, ultrafeedback_binarized, ultrafeedback_soft, hh_rlhf with epochs>=2:
        after first half of epoch batches — validation tagged "0.5", "1.5", …; then lambda_label
        on schedule for k.5 (with lambda_full_epochs) and optionally recompute p_pred_cached
        for the second half; in anchor mode (p_pred_teacher) cache is not recomputed.
    """
    _validate_train_dpo_args(
        mode=mode,
        epochs=epochs,
        lambda_min=lambda_min,
        lambda_schedule=lambda_schedule,
        lambda_full_epochs=lambda_full_epochs,
        p_pred_target_temperature=p_pred_target_temperature,
        resume_start_epoch_1based=resume_start_epoch_1based,
        resume_rewarmup_steps=resume_rewarmup_steps,
        resume_rewarmup_lr_floor=resume_rewarmup_lr_floor,
        grad_clip_norm=grad_clip_norm,
        optimizer_name=optimizer_name,
    )
    g0_start = resume_start_epoch_1based - 1

    if not isinstance(device, torch.device):
        device = torch.device(device)

    os.makedirs(output_dir, exist_ok=True)
    ensure_output_dir_lock(output_dir)
    log_path = os.path.join(output_dir, "train.log")

    prior_train_log_lines: List[str] = []
    prior_train_log_src: Optional[str] = None
    if (
        resume_start_epoch_1based > 1
        and resume_checkpoint_dir
        and str(resume_checkpoint_dir).strip()
    ):
        root = infer_run_root_from_checkpoint_dir(str(resume_checkpoint_dir))
        if root is not None:
            cand = root / "train.log"
            prior_train_log_src = str(cand)
            if cand.is_file():
                try:
                    with open(cand, "r", encoding="utf-8", errors="replace") as rf:
                        raw_lines = rf.readlines()
                    prior_train_log_lines = slice_train_log_lines_before_resume_start_epoch(
                        raw_lines, mode, resume_start_epoch_1based
                    )
                    if raw_lines and not prior_train_log_lines:
                        print(
                            "train.log: could not slice history up to "
                            f"epoch {resume_start_epoch_1based} "
                            f"({mode}: expected epoch boundary in log) at {prior_train_log_src}; "
                            "skipping prefix transfer.",
                            file=sys.stderr,
                        )
                except OSError as e:
                    print(
                        f"train.log: could not read {cand}: {e}",
                        file=sys.stderr,
                    )

    def log_msg(msg: str) -> None:
        log(msg)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    if prior_train_log_lines:
        sep = (
            f"\n--- train.log resumed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
            f"(epochs 1..{resume_start_epoch_1based - 1} from prior train.log: {prior_train_log_src}) ---\n"
        )
        with open(log_path, "w", encoding="utf-8") as wf:
            wf.writelines(prior_train_log_lines)
            wf.write(sep)
    else:
        # Each fresh run without resume history starts train.log empty.
        with open(log_path, "w", encoding="utf-8"):
            pass

    mlflow_param_dict: Dict[str, Any] = {
        "mode": mode,
        "beta": beta,
        "lr": lr,
        "batch_size": batch_size,
        "epochs": epochs,
        "lambda_min": lambda_min,
        "lambda_schedule": lambda_schedule,
        "lambda_full_epochs": lambda_full_epochs,
        "p_pred_target_temperature": p_pred_target_temperature,
        "soft_loss_type": soft_loss_type,
        "seed": seed,
        "dataset_name": dataset_name,
        "model_name": model_name,
        "output_dir": output_dir,
        "alpha": alpha,
        "label_noise_prob": label_noise_prob,
        "use_chat_template": use_chat_template,
        "num_training_steps_override": num_training_steps_override,
        "val_kl_mc_max_prompts": val_kl_mc_max_prompts,
        "val_kl_mc_num_samples": val_kl_mc_num_samples,
        "val_kl_mc_max_new_tokens": val_kl_mc_max_new_tokens,
        "val_entropy_max_prompts": val_entropy_max_prompts,
        "val_entropy_num_samples": val_entropy_num_samples,
        "val_entropy_max_new_tokens": val_entropy_max_new_tokens,
        "val_entropy_prompt_batch_size": val_entropy_prompt_batch_size,
        "val_entropy_forward_chunk_size": val_entropy_forward_chunk_size,
        "val_distributions_max_batches": val_distributions_max_batches,
        "capability_eval_dir": capability_eval_dir,
        "capability_eval_limit": capability_eval_limit,
        "capability_eval_max_new_tokens": capability_eval_max_new_tokens,
        "capability_eval_batch_size": capability_eval_batch_size,
        "capability_eval_max_prompt_tokens": capability_eval_max_prompt_tokens,
        "resume_start_epoch_1based": resume_start_epoch_1based,
        "resume_checkpoint_dir": resume_checkpoint_dir,
        "resume_rewarmup_steps": resume_rewarmup_steps,
        "resume_rewarmup_lr_floor": resume_rewarmup_lr_floor,
        "grad_clip_norm": grad_clip_norm,
        "optimizer_name": optimizer_name,
        "save_epoch_checkpoints": save_epoch_checkpoints,
        "probe_margins": probe_margins,
        "probe_size": probe_size,
        "probe_every": probe_every,
        "probe_seed": probe_seed,
    }

    with _mlflow_training_context(
        use_mlflow,
        mlflow_experiment,
        mlflow_run_name,
        mlflow_tracking_uri,
        mlflow_param_dict,
        log_path,
    ):
        run_started_at = datetime.now()
        run_started_perf = perf_counter()
        use_bayes = mode == "bayes"
        train_collate, train_loss_fn, loss_kwargs, mode_label = _build_loss_spec(
            mode=mode,
            beta=beta,
            use_chat_template=use_chat_template,
            p_pred_target_temperature=p_pred_target_temperature,
            soft_loss_type=soft_loss_type,
        )

        g = torch.Generator()
        g.manual_seed(seed)

        train_loader = _make_shuffled_train_loader(
            train_ds, train_collate, batch_size, g
        )
        val_loader = _make_ordered_loader(val_ds, collate_fn_hard, batch_size)

        if probe_margins:
            if int(probe_size) < 1:
                raise ValueError(f"probe_size must be >= 1, got {probe_size!r}")
            if int(probe_every) < 1:
                raise ValueError(f"probe_every must be >= 1, got {probe_every!r}")

        probe_indices: List[int] = []
        probe_loader = None
        probe_logp_c_ref: Optional[np.ndarray] = None
        probe_logp_r_ref: Optional[np.ndarray] = None
        probe_jsonl_path = os.path.join(output_dir, "probe_margins.jsonl")
        step_tracker = StepTracker()

        if probe_margins and len(val_ds) > 0:
            probe_indices = load_or_create_probe_indices(
                os.path.join(output_dir, "probe_indices.json"),
                n_val=len(val_ds),
                size=int(probe_size),
                probe_seed=int(probe_seed),
                log=log_msg,
                dataset_name=dataset_name,
                val_ds=val_ds,
            )
            probe_loader = build_probe_loader(
                val_ds, probe_indices, batch_size, collate_fn_hard
            )

        cap_rows: Optional[List[EvalRow]] = None
        cap_ref_cache: List[Optional[List[str]]] = [None]
        cap_ref_cache_path_obj: Optional[Path] = None
        cap_ref_cache_meta: Optional[Dict[str, Any]] = None

        def _safe_tokenizer_name_or_path() -> str:
            v = getattr(tokenizer, "name_or_path", None)
            if isinstance(v, str) and v.strip():
                return v
            return "N/A"

        def _safe_ref_model_revision() -> str:
            cfg = getattr(ref_model, "config", None)
            if cfg is None:
                return "N/A"
            rev = getattr(cfg, "_commit_hash", None)
            if isinstance(rev, str) and rev.strip():
                return rev
            rev2 = getattr(cfg, "revision", None)
            if isinstance(rev2, str) and rev2.strip():
                return rev2
            return "N/A"

        if capability_eval_dir:
            eval_p = Path(capability_eval_dir).expanduser().resolve()
            if eval_p.is_dir():
                try:
                    cap_rows = load_eval_rows(eval_p)
                    if capability_eval_limit is not None and capability_eval_limit > 0:
                        cap_rows = cap_rows[: int(capability_eval_limit)]
                    if not cap_rows:
                        log_msg(
                            f"capability_eval_dir={eval_p}: no examples found, skipping retention."
                        )
                        cap_rows = None
                    else:
                        safe_model = (model_name or "unknown_model").replace("/", "__")
                        if capability_ref_cache_path:
                            cap_ref_cache_path_obj = Path(capability_ref_cache_path).expanduser().resolve()
                        else:
                            cap_ref_cache_path_obj = (
                                eval_p / "ref_cache" / f"{safe_model}_ref_texts.json"
                            )
                        cap_ref_cache_meta = build_ref_cache_metadata(
                            cap_rows,
                            model_name=model_name or "N/A",
                            tokenizer_name_or_path=_safe_tokenizer_name_or_path(),
                            ref_model_revision=_safe_ref_model_revision(),
                            max_new_tokens=capability_eval_max_new_tokens,
                            max_prompt_tokens=capability_eval_max_prompt_tokens,
                            use_chat_template=use_chat_template,
                        )
                        loaded, reason = load_ref_texts_cache_if_compatible(
                            cap_ref_cache_path_obj, cap_ref_cache_meta
                        )
                        if loaded is not None:
                            cap_ref_cache[0] = loaded
                            log_msg(
                                f"Capability retention: loaded ref cache ({len(loaded)} responses) from {cap_ref_cache_path_obj} [{reason}]"
                            )
                        else:
                            log_msg(
                                f"Capability retention: ref cache miss at {cap_ref_cache_path_obj} [{reason}]"
                            )
                        log_msg(
                            f"Capability retention: {len(cap_rows)} examples from {eval_p} "
                            f"(ref cached after first generation)."
                        )
                except Exception as e:
                    log_msg(f"Capability retention: eval load error: {e}")
                    cap_rows = None
            else:
                log_msg(f"Capability retention: directory not found {eval_p}, skipping.")

        def _epoch_tag_for_files(epoch_display: str) -> str:
            return epoch_display.replace(".", "_")

        def _val_diff_dump_kwargs(epoch_display: str, step: int) -> Dict[str, Any]:
            if not probe_margins:
                return {}
            tag = _epoch_tag_for_files(str(epoch_display))
            return {
                "save_npz_path": os.path.join(
                    output_dir, "val_deltas", f"epoch_{tag}.npz"
                ),
                "extra_npz": {
                    "step": np.int64(step),
                    "lr": np.float64(step_tracker.last_lr),
                    "H_cum": np.float64(step_tracker.H_cum),
                    "grad_norm": np.float64(step_tracker.last_grad_norm),
                },
            }

        def _run_probe_snapshot(step: int, epoch_1based: int) -> None:
            if probe_loader is None or probe_logp_c_ref is None or probe_logp_r_ref is None:
                return
            stats = snapshot_probe(
                policy_model,
                tokenizer,
                probe_loader,
                probe_logp_c_ref,
                probe_logp_r_ref,
                device,
                use_chat_template,
                jsonl_path=probe_jsonl_path,
                step=int(step),
                epoch_1based=int(epoch_1based),
                beta=float(beta),
                lr=float(step_tracker.last_lr),
                H_cum=float(step_tracker.H_cum),
                grad_norm=float(step_tracker.last_grad_norm),
                log=log_msg,
            )
            if use_mlflow and stats:
                if stats.get("mean") == stats.get("mean"):
                    mlflow.log_metric("probe_delta_mean", stats["mean"], step=step)
                if step_tracker.last_lr == step_tracker.last_lr:
                    mlflow.log_metric("probe_lr", step_tracker.last_lr, step=step)
                mlflow.log_metric("probe_H_cum", step_tracker.H_cum, step=step)
                if step_tracker.last_grad_norm == step_tracker.last_grad_norm:
                    mlflow.log_metric(
                        "probe_grad_norm", step_tracker.last_grad_norm, step=step
                    )

        def _maybe_probe_after_step(step: int, epoch_1based: int) -> None:
            if probe_loader is None:
                return
            if int(step) % int(probe_every) != 0:
                return
            _run_probe_snapshot(step, epoch_1based)

        # Mutated as soon as pairwise NLL is known (before generate metrics).
        best_val_nll_box = [float("inf")]

        def _save_best_checkpoint(val_nll: float) -> None:
            if val_nll >= best_val_nll_box[0]:
                return
            best_val_nll_box[0] = val_nll
            ckpt_dir = os.path.join(output_dir, "best")
            os.makedirs(ckpt_dir, exist_ok=True)
            tokenizer.save_pretrained(ckpt_dir)
            policy_model.save_pretrained(ckpt_dir)
            log_msg(f"New best NLL {val_nll:.4f} -> checkpoint saved: {ckpt_dir}")

        def _run_capability_retention(epoch_display: str, step_m: int) -> None:
            if cap_rows is None:
                return
            try:
                desc_ref = (
                    f"cap_ret ref [ep {epoch_display}]"
                    if cap_ref_cache[0] is None
                    else None
                )
                summary, cap_ref_cache[0] = run_retention_eval_pair(
                    tokenizer,
                    ref_model,
                    policy_model,
                    device,
                    cap_rows,
                    cap_ref_cache[0],
                    capability_eval_max_new_tokens,
                    capability_eval_batch_size,
                    capability_eval_max_prompt_tokens,
                    desc_ref=desc_ref,
                    desc_pol=f"cap_ret policy [ep {epoch_display}]",
                )
            except Exception as e:
                log_msg(f"Capability retention: generation/scoring error: {e}")
                return
            if (
                cap_ref_cache_path_obj is not None
                and cap_ref_cache_meta is not None
                and cap_ref_cache[0] is not None
            ):
                try:
                    save_ref_texts_cache(
                        cap_ref_cache_path_obj, cap_ref_cache_meta, cap_ref_cache[0]
                    )
                except Exception as err:
                    log_msg(
                        f"Capability retention: failed to save ref cache {cap_ref_cache_path_obj}: {err}"
                    )
                else:
                    log_msg(
                        f"Capability retention: ref cache saved to {cap_ref_cache_path_obj}"
                    )
            for line in format_capability_retention_log_lines(summary, epoch_display):
                log_msg(line)
            tag = _epoch_tag_for_files(epoch_display)
            cap_ret_dir = os.path.join(output_dir, "capability_retention")
            os.makedirs(cap_ret_dir, exist_ok=True)
            cap_json = os.path.join(
                cap_ret_dir, f"capability_retention_epoch{tag}.json"
            )
            try:
                with open(cap_json, "w", encoding="utf-8") as f:
                    json.dump(summary, f, ensure_ascii=False, indent=2)
            except OSError as err:
                log_msg(f"Capability retention: failed to write {cap_json}: {err}")
            if use_mlflow:
                log_mlflow_capability_metrics(
                    summary,
                    step_m,
                    lambda n, v, s: mlflow.log_metric(n, v, step=s),
                )

        def _run_validation(
            epoch_display: str,
            mlflow_step: Optional[int] = None,
            training_seconds: Optional[float] = None,
            save_best: bool = False,
        ) -> float:
            """epoch_display: '1', '0.5', '1.5', ... for logs and artifact names. Returns val NLL."""
            tag = _epoch_tag_for_files(epoch_display)
            step_m = global_step if mlflow_step is None else mlflow_step
            t_validation_total_start = perf_counter()
            if device.type == "cuda" and torch.cuda.is_available():
                try:
                    idx = device.index if device.index is not None else torch.cuda.current_device()
                    torch.cuda.reset_peak_memory_stats(idx)
                except Exception:
                    pass
            policy_model.eval()
            val_dpo_sum = 0.0
            val_kl_sum = 0.0
            val_n = 0
            t_validation_core_start = perf_counter()
            with torch.no_grad():
                for batch in tqdm(
                    val_loader, desc=f"val DPO [ep {epoch_display}]", leave=False
                ):
                    loss, kl_b, *_ = hard_dpo_loss(
                        batch,
                        tokenizer,
                        policy_model,
                        ref_model,
                        device,
                        beta=beta,
                        use_chat_template=use_chat_template,
                    )
                    n = len(batch["prompt"])
                    val_dpo_sum += loss.item() * n
                    val_kl_sum += kl_b * n
                    val_n += n
            val_dpo = val_dpo_sum / max(1, val_n)
            val_kl = val_kl_sum / max(1, val_n)
            val_nll = eval_pairwise_nll(
                val_loader,
                tokenizer,
                policy_model,
                device,
                beta=1.0,
                use_chat_template=use_chat_template,
                desc=f"val NLL [ep {epoch_display}]",
            )
            val_acc = eval_pairwise_accuracy(
                val_loader,
                tokenizer,
                policy_model,
                device,
                use_chat_template=use_chat_template,
                desc=f"val acc [ep {epoch_display}]",
            )
            validation_core_seconds = perf_counter() - t_validation_core_start

            log_msg("")
            log_msg(f"=== Validation, epoch {epoch_display} ===")
            log_msg("")
            log_msg(f"validation DPO loss   : {val_dpo:.4f}")
            log_msg(f"validation logp_gap_mean : {val_kl:.4f}")
            log_msg(f"validation pair NLL   : {val_nll:.4f}")
            log_msg(f"validation pair acc   : {100 * val_acc:.2f}%")

            # Always compute full-val DPO margin (diff = Δ_θ − Δ_ref).
            val_diff_stats = log_val_diff_from_loader(
                policy_model,
                ref_model,
                tokenizer,
                val_loader,
                device,
                log_msg,
                use_chat_template=use_chat_template,
                **_val_diff_dump_kwargs(epoch_display, step_m),
            )

            if use_mlflow:
                mlflow.log_metric("val_dpo_loss", val_dpo, step=step_m)
                mlflow.log_metric("logp_gap_mean", val_kl, step=step_m)
                mlflow.log_metric("val_pair_nll", val_nll, step=step_m)
                mlflow.log_metric("val_pair_acc", val_acc, step=step_m)
                if val_diff_stats:
                    mlflow.log_metric(
                        "val_diff_mean", val_diff_stats["mean"], step=step_m
                    )
                    mlflow.log_metric(
                        "val_diff_std", val_diff_stats["std"], step=step_m
                    )
                    mlflow.log_metric(
                        "val_diff_median", val_diff_stats["median"], step=step_m
                    )
                try:
                    ef = float(epoch_display)
                except ValueError:
                    ef = float("nan")
                if not math.isnan(ef):
                    mlflow.log_metric("epoch_float", ef, step=step_m)

            if (
                val_distributions_max_batches is not None
                and val_distributions_max_batches > 0
                and len(val_ds) > 0
            ):
                # Optional subset histograms / npz; full-val mean already logged above.
                try:
                    dist = compute_val_delta_distributions(
                        policy_model,
                        ref_model,
                        tokenizer,
                        val_loader,
                        device,
                        use_chat_template=use_chat_template,
                        max_batches=val_distributions_max_batches,
                    )
                    dt = dist["delta_theta"]
                    dr = dist["delta_ref"]
                    margin = dist["diff"]

                    def _val_dist_stats_line(label: str, arr: np.ndarray) -> str:
                        if arr.size == 0:
                            return f"{label}: (no samples)"
                        mean = float(np.mean(arr))
                        std = float(np.std(arr))
                        med = float(np.median(arr))
                        p5 = float(np.percentile(arr, 5))
                        p95 = float(np.percentile(arr, 95))
                        return (
                            f"{label}: mean={mean:.2f} std={std:.2f} median={med:.2f} "
                            f"p5={p5:.2f} p95={p95:.2f}"
                        )

                    log_msg(_val_dist_stats_line("val_delta_theta  ", dt))
                    log_msg(_val_dist_stats_line("val_delta_ref    ", dr))
                    log_msg(_val_dist_stats_line("val_diff (margin, subset)", margin))

                    if use_mlflow and margin.size > 0:
                        mlflow.log_metric(
                            "val_delta_theta_mean", float(np.mean(dt)), step=step_m
                        )
                        mlflow.log_metric(
                            "val_delta_ref_mean", float(np.mean(dr)), step=step_m
                        )

                    npz_path = os.path.join(
                        output_dir, f"val_distributions_epoch{tag}.npz"
                    )
                    np.savez_compressed(
                        npz_path,
                        delta_theta=dt,
                        delta_ref=dr,
                        diff=margin,
                    )
                except Exception as e:
                    log_msg(
                        "validation delta distributions: FAILED "
                        f"({type(e).__name__}: {e}); continuing without margin-distribution metrics"
                    )

            # Save best before generate (KL-MC / entropy / cap_ret). Those can
            # run for ~1h; a kill there must not drop an already-known better NLL.
            if save_best:
                _save_best_checkpoint(val_nll)

            if val_kl_mc_max_prompts > 0 and len(val_ds) > 0:
                n_mc = min(int(val_kl_mc_max_prompts), len(val_ds))
                log_msg(
                    f"validation KL_MC: computing (first {n_mc} val prompts × {val_kl_mc_num_samples} samples)..."
                )
                t_mc_kl_start = perf_counter()
                try:
                    mc_prompts = val_ds.select(range(n_mc))["prompt"]
                    kl_mc_stats = estimate_val_kl_mc(
                        policy_model,
                        ref_model,
                        tokenizer,
                        mc_prompts,
                        device,
                        num_samples_per_prompt=val_kl_mc_num_samples,
                        max_new_tokens=val_kl_mc_max_new_tokens,
                        use_chat_template=use_chat_template,
                        prompt_batch_size=val_kl_mc_prompt_batch_size,
                    )
                    val_kl_mc_per_seq = float(kl_mc_stats["per_seq"])
                    val_kl_mc_per_token = float(kl_mc_stats["per_token"])
                    n_tokens_mc = int(kl_mc_stats["total_tokens"])
                    log_msg(
                        f"validation KL_MC (π‖ref, MC samples from policy, {n_mc} prompts × "
                        f"{val_kl_mc_num_samples}): per_seq={val_kl_mc_per_seq:.4f}, "
                        f"per_token={val_kl_mc_per_token:.6f} (total_tokens={n_tokens_mc})"
                    )
                    if use_mlflow:
                        # per-seq kept under legacy name for plot compatibility;
                        # per-token is the primary metric for cross-run comparison.
                        mlflow.log_metric("val_kl_mc", val_kl_mc_per_seq, step=step_m)
                        mlflow.log_metric(
                            "val_kl_mc_per_seq", val_kl_mc_per_seq, step=step_m
                        )
                        mlflow.log_metric(
                            "val_kl_mc_per_token", val_kl_mc_per_token, step=step_m
                        )
                except Exception as e:
                    log_msg(
                        f"validation KL_MC: FAILED ({type(e).__name__}: {e}); continuing without val_kl_mc metric"
                    )
                mc_kl_seconds = perf_counter() - t_mc_kl_start
            elif val_kl_mc_max_prompts <= 0:
                log_msg("validation KL_MC: skipped (val_kl_mc_max_prompts<=0).")
                mc_kl_seconds = 0.0
            elif len(val_ds) == 0:
                log_msg("validation KL_MC: skipped (empty val_ds).")
                mc_kl_seconds = 0.0

            if val_entropy_max_prompts > 0 and len(val_ds) > 0:
                n_ent = min(int(val_entropy_max_prompts), len(val_ds))
                log_msg(
                    "validation response entropy: computing "
                    f"(first {n_ent} val prompts × {val_entropy_num_samples} samples, "
                    f"L={val_entropy_max_new_tokens})..."
                )
                try:
                    ent_prompts = val_ds.select(range(n_ent))["prompt"]
                    ent_stats = estimate_val_response_entropy(
                        policy_model,
                        tokenizer,
                        ent_prompts,
                        str(device),
                        num_samples_per_prompt=val_entropy_num_samples,
                        max_new_tokens=val_entropy_max_new_tokens,
                        entropy_tokens_limit=val_entropy_max_new_tokens,
                        use_chat_template=use_chat_template,
                        prompt_batch_size=val_entropy_prompt_batch_size,
                        forward_chunk_size=val_entropy_forward_chunk_size,
                    )
                    _log_val_response_entropy_two_lines(
                        log_msg,
                        ent_stats,
                        tokenizer,
                        policy_model,
                        l_tokens=val_entropy_max_new_tokens,
                        n_prompts=n_ent,
                        num_samples=val_entropy_num_samples,
                    )
                    if use_mlflow:
                        mlflow.log_metric("val_resp_entropy_mean", ent_stats["mean"], step=step_m)
                        mlflow.log_metric(
                            "val_resp_entropy_median", ent_stats["median"], step=step_m
                        )
                        mlflow.log_metric("val_resp_entropy_p10", ent_stats["p10"], step=step_m)
                        mlflow.log_metric("val_resp_entropy_p90", ent_stats["p90"], step=step_m)
                except Exception as e:
                    log_msg(
                        "validation response entropy: FAILED "
                        f"({type(e).__name__}: {e}); continuing without response entropy metric"
                    )
            elif val_entropy_max_prompts <= 0:
                log_msg("validation response entropy: skipped (val_entropy_max_prompts<=0).")
            elif len(val_ds) == 0:
                log_msg("validation response entropy: skipped (empty val_ds).")

            t_capability_start = perf_counter()
            _run_capability_retention(epoch_display, step_m)
            capability_seconds = perf_counter() - t_capability_start
            validation_total_seconds = perf_counter() - t_validation_total_start
            validation_peak_mem_gb = _gpu_peak_memory_gb(device)

            timing_parts = [f"validation={_fmt_seconds(validation_core_seconds)}"]
            if training_seconds is not None:
                timing_parts.insert(0, f"training={_fmt_seconds(training_seconds)}")
            timing_parts.append(f"mc_kl={_fmt_seconds(mc_kl_seconds)}")
            timing_parts.append(
                f"capability_retention={_fmt_seconds(capability_seconds)}"
            )
            timing_parts.append(f"validation_total={_fmt_seconds(validation_total_seconds)}")
            log_msg("timings: " + ", ".join(timing_parts))
            # Training VRAM peak is logged right after train_one_epoch_dpo (separate line).
            log_msg(f"gpu_mem_peak: validation={_fmt_mem_gb(validation_peak_mem_gb)}")
            log_msg("")
            # generate() in KL-MC / entropy / cap-retention can leave a large
            # caching allocator footprint; free it before the next train batch.
            _cuda_empty_cache(device)
            return val_nll

        use_mid_epoch_val = (
            mode != "hard"
            and epochs >= 2
            and dataset_name in ULTRAFB_MID_EPOCH_DATASETS
        )

        actual_steps_per_epoch = len(train_loader)
        total_actual_steps = epochs * actual_steps_per_epoch
        if num_training_steps_override is not None:
            num_training_steps = num_training_steps_override
        else:
            num_training_steps = total_actual_steps

        steps_per_schedule_epoch = max(1, num_training_steps // max(1, epochs))
        if num_training_steps % epochs != 0:
            log_msg(
                "Warning: num_training_steps is not divisible by epochs "
                "evenly; LR start offset: "
                f"{g0_start} * floor({num_training_steps}/{epochs})."
            )
        start_global_step = g0_start * steps_per_schedule_epoch
        if start_global_step >= num_training_steps:
            raise ValueError(
                f"resume_start_epoch_1based={resume_start_epoch_1based} yields "
                f"start_global_step={start_global_step} >= num_training_steps={num_training_steps}"
            )

        # Log planned vs actual steps so the LR schedule matches real run length
        # (common pitfall with num_training_steps_override from hard_train_size: if soft train is longer,
        # scheduler hits lr=0 before training ends; if shorter, lr never reaches 0).
        steps_delta = total_actual_steps - num_training_steps
        steps_delta_pct = (
            100.0 * steps_delta / max(1, num_training_steps)
        )
        log_msg(
            f"LR schedule: num_training_steps={num_training_steps}"
            + (" (override)" if num_training_steps_override is not None else " (auto: epochs*len(train_loader))")
        )
        log_msg(
            f"Actual steps: epochs={epochs} × len(train_loader)={actual_steps_per_epoch} "
            f"= {total_actual_steps}; delta(actual-planned)={steps_delta:+d} "
            f"({steps_delta_pct:+.2f}%)"
        )
        if steps_delta > 0:
            log_msg(
                "Warning: actual steps exceed planned num_training_steps — "
                f"last {steps_delta} steps run at lr=0 (linear decay already at zero)."
            )
        elif steps_delta < 0:
            final_lr_frac = 1.0 - (total_actual_steps / max(1, num_training_steps))
            log_msg(
                "Warning: actual steps are below planned num_training_steps — "
                f"by end of run lr will not reach zero (≈ {final_lr_frac:.2%} of max lr remains)."
            )

        if use_mlflow:
            mlflow.log_param("num_training_steps", num_training_steps)
            mlflow.log_param("total_actual_steps", total_actual_steps)
            mlflow.log_param("actual_steps_per_epoch", actual_steps_per_epoch)
            mlflow.log_param("steps_delta_actual_minus_planned", steps_delta)

        log_msg(f"=== {mode_label} ===")
        log_msg(f"Run started at: {run_started_at.strftime('%Y-%m-%d %H:%M:%S')}")
        log_msg(f"Model: {model_name or 'N/A'}")
        log_msg(
            f"Dataset: {dataset_name or 'N/A'}, train size: {len(train_ds)}, "
            f"val size: {len(val_ds)}, batch_size={batch_size}"
        )
        _lnp = label_noise_prob if label_noise_prob is not None else "N/A"
        loss_log = soft_loss_type if mode in ("soft", "bayes") else mode
        log_msg(
            f"train_dpo start: loss={loss_log}, epochs_total={epochs}, seed={seed}\n"
            f"beta={beta}, lr={lr},\n"
            f"epochs_this_run={epochs - g0_start}, resume_start_epoch_1based={resume_start_epoch_1based}\n"
            f"lambda_min={lambda_min}, lambda_schedule={lambda_schedule}, lambda_full_epochs={lambda_full_epochs}, "
            f"p_pred_target_temperature={p_pred_target_temperature}, label_noise_prob={_lnp}\n"
            f"optimizer={optimizer_name}, grad_clip_norm={grad_clip_norm}, "
            f"save_epoch_checkpoints={save_epoch_checkpoints}\n"
            f"probe_margins={probe_margins}, probe_size={probe_size}, "
            f"probe_every={probe_every}, probe_seed={probe_seed}"
        )
        log_msg(f"MAX_PROMPT_LEN={MAX_PROMPT_LEN}, MAX_FULL_LEN={MAX_FULL_LEN}, use_chat_template={use_chat_template}")
        log_msg(
            f"val_KL_MC: max_prompts={val_kl_mc_max_prompts}, samples_per_prompt={val_kl_mc_num_samples}, "
            f"max_new_tokens={val_kl_mc_max_new_tokens} (max_prompts=0 disables MC-KL after each epoch val)"
        )
        log_msg(
            f"val_response_entropy: max_prompts={val_entropy_max_prompts}, "
            f"samples_per_prompt={val_entropy_num_samples}, max_new_tokens={val_entropy_max_new_tokens}, "
            f"prompt_batch_size={val_entropy_prompt_batch_size}, "
            f"forward_chunk_size={val_entropy_forward_chunk_size} "
            "(max_prompts=0 disables response entropy after each epoch val)"
        )

        # Initial validation: with --start-epoch>1, like end of epoch (start-1), full val as after that epoch.
        pre_val_epoch_done = g0_start  # completed epochs 1..g0_start; weights = after epoch pre_val_epoch_done

        checkpoint_val_nll: Optional[float] = None
        policy_model.eval()
        if g0_start == 0:
            log_msg("")
            log_msg("=== Initial (before training), epoch 0 ===")
            init_dpo_sum = 0.0
            init_kl_sum = 0.0
            init_n = 0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="init DPO loss", leave=False):
                    loss, kl_b, *_ = hard_dpo_loss(
                        batch,
                        tokenizer,
                        policy_model,
                        ref_model,
                        device,
                        beta=beta,
                        use_chat_template=use_chat_template,
                    )
                    n = len(batch["prompt"])
                    init_dpo_sum += loss.item() * n
                    init_kl_sum += kl_b * n
                    init_n += n
            init_dpo = init_dpo_sum / max(1, init_n)
            init_kl = init_kl_sum / max(1, init_n)
            init_nll = eval_pairwise_nll(
                val_loader,
                tokenizer,
                policy_model,
                device,
                beta=1.0,
                use_chat_template=use_chat_template,
                desc="init pairwise NLL",
            )
            init_acc = eval_pairwise_accuracy(
                val_loader,
                tokenizer,
                policy_model,
                device,
                use_chat_template=use_chat_template,
                desc="init pairwise acc",
            )
            log_msg("")
            log_msg(f"validation DPO loss   : {init_dpo:.4f}")
            log_msg(f"validation logp_gap_mean : {init_kl:.4f}")
            log_msg(f"validation pair NLL   : {init_nll:.4f}")
            log_msg(f"validation pair acc   : {100 * init_acc:.2f}%")
            log_val_diff_from_loader(
                policy_model,
                ref_model,
                tokenizer,
                val_loader,
                device,
                log_msg,
                use_chat_template=use_chat_template,
                **_val_diff_dump_kwargs("0", 0),
            )
            if val_entropy_max_prompts > 0 and len(val_ds) > 0:
                n_ent = min(int(val_entropy_max_prompts), len(val_ds))
                log_msg(
                    "validation response entropy: computing "
                    f"(first {n_ent} val prompts × {val_entropy_num_samples} samples, "
                    f"L={val_entropy_max_new_tokens})..."
                )
                try:
                    ent_prompts = val_ds.select(range(n_ent))["prompt"]
                    ent_stats = estimate_val_response_entropy(
                        policy_model,
                        tokenizer,
                        ent_prompts,
                        str(device),
                        num_samples_per_prompt=val_entropy_num_samples,
                        max_new_tokens=val_entropy_max_new_tokens,
                        entropy_tokens_limit=val_entropy_max_new_tokens,
                        use_chat_template=use_chat_template,
                        prompt_batch_size=val_entropy_prompt_batch_size,
                        forward_chunk_size=val_entropy_forward_chunk_size,
                    )
                    _log_val_response_entropy_two_lines(
                        log_msg,
                        ent_stats,
                        tokenizer,
                        policy_model,
                        l_tokens=val_entropy_max_new_tokens,
                        n_prompts=n_ent,
                        num_samples=val_entropy_num_samples,
                    )
                    if use_mlflow:
                        step_m = start_global_step
                        mlflow.log_metric(
                            "val_resp_entropy_mean", ent_stats["mean"], step=step_m
                        )
                        mlflow.log_metric(
                            "val_resp_entropy_median", ent_stats["median"], step=step_m
                        )
                        mlflow.log_metric(
                            "val_resp_entropy_p10", ent_stats["p10"], step=step_m
                        )
                        mlflow.log_metric(
                            "val_resp_entropy_p90", ent_stats["p90"], step=step_m
                        )
                except Exception as e:
                    log_msg(
                        "validation response entropy: FAILED "
                        f"({type(e).__name__}: {e}); continuing without response entropy metric"
                    )
            elif val_entropy_max_prompts <= 0:
                log_msg("validation response entropy: skipped (val_entropy_max_prompts<=0).")
            elif len(val_ds) == 0:
                log_msg("validation response entropy: skipped (empty val_ds).")
            _run_capability_retention("init", 0)
        else:
            checkpoint_val_nll = _run_validation(
                str(pre_val_epoch_done),
                mlflow_step=start_global_step,
            )

        optimizer_key = optimizer_name.lower()
        sgd_momentum = 0.9
        if optimizer_key == "adamw":
            optimizer = torch.optim.AdamW(policy_model.parameters(), lr=lr)
        elif optimizer_key == "sgd":
            optimizer = torch.optim.SGD(
                policy_model.parameters(), lr=lr, momentum=sgd_momentum
            )
        else:  # defensive fallback; primary checks in _validate_train_dpo_args
            raise ValueError(
                f"Unsupported optimizer_name={optimizer_name!r}; expected one of {OPTIMIZER_CHOICES}"
            )
        base_warmup_steps = max(10, num_training_steps // 20)
        do_resume_rewarmup = start_global_step > 0 and resume_rewarmup_steps > 0

        def _lr_lambda(current_step: int) -> float:
            # Main schedule: linear warmup to base_warmup_steps, then linear
            # decay to 0 by num_training_steps — same as get_linear_schedule_with_warmup.
            if current_step < base_warmup_steps:
                base = current_step / max(1, base_warmup_steps)
            else:
                progress = (current_step - base_warmup_steps) / max(
                    1, num_training_steps - base_warmup_steps
                )
                base = max(0.0, 1.0 - progress)
            # On top of main schedule — local re-warmup: for the first
            # resume_rewarmup_steps after restart an extra factor ramps
            # linearly from resume_rewarmup_lr_floor to 1.0.
            # Lets AdamW rebuild first/second moments without huge
            # updates on step 0 (optimizer state is not saved).
            if do_resume_rewarmup:
                rel = current_step - start_global_step
                if 0 <= rel < resume_rewarmup_steps:
                    ramp = resume_rewarmup_lr_floor + (
                        1.0 - resume_rewarmup_lr_floor
                    ) * (rel / resume_rewarmup_steps)
                else:
                    ramp = 1.0
            else:
                ramp = 1.0
            return base * ramp

        scheduler = LambdaLR(optimizer, lr_lambda=_lr_lambda)
        for _ in range(start_global_step):
            scheduler.step()
        # Completed steps 0..N-1 used lr_lambda(k)*lr (rewarmup ramp is 1 for k < start).
        step_tracker.H_cum = float(lr) * sum(
            float(_lr_lambda(k)) for k in range(int(start_global_step))
        )
        step_tracker.last_lr = float(optimizer.param_groups[0]["lr"])

        if do_resume_rewarmup:
            log_msg(
                f"Resume rewarmup: first {resume_rewarmup_steps} steps after "
                f"start_global_step={start_global_step} — extra linear "
                f"lr factor from {resume_rewarmup_lr_floor:g} to 1.0 on top of main "
                "schedule (offsets zeroed AdamW moments on resume)."
            )

        # Anchor mode: in a continuous run p_pred_teacher appears at end of epoch k (lambda_full_epochs).
        # Starting epoch (k+1) without that column — compute teacher from current weights (end of k), same as there.
        if (
            mode != "hard"
            and lambda_full_epochs > 0
            and resume_start_epoch_1based == lambda_full_epochs + 1
            and "p_pred_teacher" not in train_ds.column_names
        ):
            _reset_cuda_peak_memory_stats(device)
            train_ds = precompute_p_pred_teacher(
                train_ds,
                tokenizer,
                policy_model,
                ref_model,
                device=device,
                beta=beta,
                use_chat_template=use_chat_template,
                batch_size=batch_size,
                collate_fn=train_collate,
            )
            log_msg(
                "gpu_mem_peak: "
                f"precompute_p_pred_teacher={_fmt_mem_gb(_gpu_peak_memory_gb(device))}"
            )
            if "p_pred_cached" in train_ds.column_names:
                train_ds = train_ds.remove_columns(["p_pred_cached"])
            train_loader = _make_shuffled_train_loader(
                train_ds, train_collate, batch_size, g
            )
            log_msg(
                f"p_pred_teacher: fixed from loaded weights "
                f"(resume_start_epoch_1based={resume_start_epoch_1based} == lambda_full_epochs+1={lambda_full_epochs + 1}; "
                f"equivalent to end of epoch {lambda_full_epochs} in a continuous run). "
                f"As in continuous run: on all tail epochs (λ<1) "
                f"p_pred = 0.5·p_teacher + 0.5·σ((β·diff)/T)."
            )

        best_val_nll_box[0] = (
            float("inf") if checkpoint_val_nll is None else float(checkpoint_val_nll)
        )
        global_step = start_global_step
        if probe_loader is not None:
            probe_logp_c_ref, probe_logp_r_ref = cache_ref_logps(
                ref_model,
                tokenizer,
                probe_loader,
                device,
                use_chat_template,
                cache_path=os.path.join(output_dir, "probe_ref_logps.npz"),
                indices=probe_indices,
            )
            log_msg(
                f"probe: cached ref logps n={int(probe_logp_c_ref.shape[0])} "
                f"(two policy forwards per snapshot, every {probe_every} steps)"
            )
            _run_probe_snapshot(global_step, max(1, g0_start))
        for g0 in range(g0_start, epochs):
            log_msg("")
            log_msg(f"=== Epoch {g0 + 1}/{epochs} ===")
            if mode == "hard":
                epoch_loss_kw = dict(loss_kwargs)
                mid_hook: Optional[Callable[[int], None]] = None
            else:
                (
                    epoch_loss_kw,
                    lambda_label_epoch,
                    has_teacher_anchor,
                    teacher_blend_w,
                ) = _epoch_lambda_and_loss_kw(
                    g0=g0,
                    epochs=epochs,
                    lambda_full_epochs=lambda_full_epochs,
                    lambda_min=lambda_min,
                    lambda_schedule=lambda_schedule,
                    has_teacher_column="p_pred_teacher" in train_ds.column_names,
                    base_loss_kwargs=loss_kwargs,
                )
                log_msg(
                    f"[epoch {g0 + 1}/{epochs}] lambda_label={lambda_label_epoch:.6f}"
                    + (" (teacher_anchor)" if has_teacher_anchor else "")
                    + (
                        f", p_pred_teacher_blend={teacher_blend_w}"
                        if has_teacher_anchor
                        else ""
                    )
                )

                if lambda_label_epoch < 1.0 and not has_teacher_anchor:
                    train_ds = precompute_p_pred_cached(
                        train_ds,
                        tokenizer,
                        policy_model,
                        ref_model,
                        device=device,
                        beta=beta,
                        use_chat_template=use_chat_template,
                        batch_size=batch_size,
                        collate_fn=train_collate,
                    )
                    train_loader = _make_shuffled_train_loader(
                        train_ds, train_collate, batch_size, g
                    )

                mid_hook = None
                if use_mid_epoch_val and len(train_loader) >= 2:
                    # Split the epoch into two disjoint halves by examples:
                    # one permutation of train_ds indices, cut at
                    # first_count_examples = n_first_batches * batch_size, then
                    # select(idx_first) and select(idx_second). Row order
                    # in train_ds is preserved across add_column/remove_columns/map
                    # in datasets, so idx_second stays a valid sample of
                    # remaining rows even if mid_hook recomputes
                    # p_pred_cached. Result: each example used exactly once per epoch,
                    # no skips or duplicates.
                    n_examples = len(train_ds)
                    perm_t = torch.randperm(n_examples, generator=g).tolist()
                    num_batches_total = (n_examples + batch_size - 1) // batch_size
                    n_first_batches = num_batches_total // 2
                    first_count_examples = min(
                        n_first_batches * batch_size, n_examples
                    )
                    idx_first = perm_t[:first_count_examples]
                    idx_second = perm_t[first_count_examples:]

                    first_ds = train_ds.select(idx_first)
                    first_loader_local = _make_ordered_loader(
                        first_ds, train_collate, batch_size
                    )
                    train_loader_box: List[Optional[DataLoader]] = [
                        first_loader_local,
                        None,
                    ]

                    def mid_hook(gs: int) -> None:
                        nonlocal train_ds, epoch_loss_kw
                        mid_epoch_display = f"{g0 + 0.5:.1f}"
                        # Mid-epoch validation — diagnostics, not save-best:
                        # failures here (KL_MC OOM, network loss during cap-retention, etc.)
                        # must not abort the remaining half-epoch. Log and continue training.
                        try:
                            _run_validation(mid_epoch_display, mlflow_step=gs)
                        except Exception as e:
                            log_msg(
                                f"[epoch {mid_epoch_display}/{epochs}] mid-epoch validation FAILED "
                                f"({type(e).__name__}: {e}); continuing second half of epoch "
                                "without mid-epoch metrics."
                            )
                            _cuda_empty_cache(device)
                        prog_m = _lambda_schedule_progress(
                            g0, epochs, lambda_full_epochs, 0.5
                        )
                        lambda_mid = _lambda_label_at_progress(
                            prog_m, lambda_min, lambda_schedule
                        )
                        log_msg(
                            f"[epoch {mid_epoch_display}/{epochs}] "
                            f"lambda_label={lambda_mid:.6f} (2nd half)"
                        )
                        teacher_here = (
                            lambda_full_epochs > 0
                            and "p_pred_teacher" in train_ds.column_names
                        )
                        tw = 0.5 if teacher_here else 0.0
                        epoch_loss_kw.clear()
                        epoch_loss_kw.update(
                            {
                                **loss_kwargs,
                                "lambda_label": lambda_mid,
                                "p_pred_teacher_blend": tw,
                            }
                        )
                        if lambda_mid < 1.0 and not teacher_here:
                            train_ds = precompute_p_pred_cached(
                                train_ds,
                                tokenizer,
                                policy_model,
                                ref_model,
                                device=device,
                                beta=beta,
                                use_chat_template=use_chat_template,
                                batch_size=batch_size,
                                collate_fn=train_collate,
                            )
                        second_ds = train_ds.select(idx_second)
                        train_loader_box[1] = _make_ordered_loader(
                            second_ds, train_collate, batch_size
                        )
                        policy_model.train()
                else:
                    train_loader_box = [train_loader]

            if mode == "hard":
                train_loader_box = [train_loader]

            if device.type == "cuda" and torch.cuda.is_available():
                try:
                    idx = device.index if device.index is not None else torch.cuda.current_device()
                    torch.cuda.reset_peak_memory_stats(idx)
                except Exception:
                    pass
            t_training_start = perf_counter()
            global_step = train_one_epoch_dpo(
                train_loader_box,
                tokenizer,
                policy_model,
                ref_model,
                device,
                train_loss_fn,
                optimizer,
                scheduler,
                g0 + 1,
                global_step,
                loss_kw=epoch_loss_kw,
                grad_clip_norm=grad_clip_norm,
                log=log_msg,
                use_mlflow=use_mlflow,
                mid_epoch_hook=mid_hook if mode != "hard" else None,
                step_tracker=step_tracker if probe_margins else None,
                after_step_hook=_maybe_probe_after_step if probe_margins else None,
            )
            training_seconds = perf_counter() - t_training_start
            training_peak_mem_gb = _gpu_peak_memory_gb(device)
            log_msg(f"gpu_mem_peak: training={_fmt_mem_gb(training_peak_mem_gb)}")
            # In split mode train_loader_box holds two disjoint halves of the
            # current epoch, so do not treat it as the main train_loader.
            # train_loader is recreated at the next epoch start in any branch that
            # actually iterates it (precompute_p_pred_cached at epoch start or
            # teacher_anchor path below); until then it is only used for len(train_loader)
            # (batch count of the full dataset), unchanged by mid_hook.

            if (
                mode != "hard"
                and lambda_full_epochs > 0
                and (g0 + 1) == lambda_full_epochs
            ):
                _reset_cuda_peak_memory_stats(device)
                train_ds = precompute_p_pred_teacher(
                    train_ds,
                    tokenizer,
                    policy_model,
                    ref_model,
                    device=device,
                    beta=beta,
                    use_chat_template=use_chat_template,
                    batch_size=batch_size,
                    collate_fn=train_collate,
                )
                log_msg(
                    "gpu_mem_peak: "
                    f"precompute_p_pred_teacher={_fmt_mem_gb(_gpu_peak_memory_gb(device))}"
                )
                if "p_pred_cached" in train_ds.column_names:
                    train_ds = train_ds.remove_columns(["p_pred_cached"])
                train_loader = _make_shuffled_train_loader(
                    train_ds, train_collate, batch_size, g
                )
                log_msg(
                    f"p_pred_teacher: fixed at end of epoch {g0 + 1} (1-based k={lambda_full_epochs}); "
                    f"from epoch {g0 + 2} λ<1 on schedule; at λ<1 p_pred_teacher_blend=0.5 on all tail steps."
                )

            policy_model.eval()
            _run_validation(
                str(g0 + 1),
                training_seconds=training_seconds,
                save_best=True,
            )

            if save_epoch_checkpoints:
                epoch_ckpt_dir = os.path.join(
                    output_dir, "epochs", f"epoch_{g0 + 1:03d}"
                )
                os.makedirs(epoch_ckpt_dir, exist_ok=True)
                tokenizer.save_pretrained(epoch_ckpt_dir)
                policy_model.save_pretrained(epoch_ckpt_dir)
                log_msg(
                    f"[epoch {g0 + 1}/{epochs}] checkpoint (full epoch only): {epoch_ckpt_dir}"
                )
            else:
                log_msg(
                    f"[epoch {g0 + 1}/{epochs}] skip epochs/ checkpoint "
                    "(save_epoch_checkpoints=False)"
                )

        run_finished_at = datetime.now()
        run_duration_sec = perf_counter() - run_started_perf
        run_duration_hours = run_duration_sec / 3600.0
        log_msg("")
        log_msg(f"Run finished at: {run_finished_at.strftime('%Y-%m-%d %H:%M:%S')}")
        log_msg("Run status: SUCCESS")
        log_msg(f"Run duration: {run_duration_hours:.2f}h")
