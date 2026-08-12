# -*- coding: utf-8 -*-
"""
Validation distributions of DPO logits: delta_theta, delta_ref, diff (margin).
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from utils.config import MAX_FULL_LEN, MAX_PROMPT_LEN
from utils.metrics import get_logps


def summarize_margin(margin: np.ndarray) -> Dict[str, float]:
    """Summary stats for DPO margin diff = (Δ_θ - Δ_ref) on validation pairs."""
    if margin.size == 0:
        return {}
    return {
        "mean": float(np.mean(margin)),
        "std": float(np.std(margin)),
        "median": float(np.median(margin)),
        "p5": float(np.percentile(margin, 5)),
        "p95": float(np.percentile(margin, 95)),
        "abs_mean": float(np.mean(np.abs(margin))),
        "n": float(margin.size),
    }


def format_val_diff_stats_line(stats: Dict[str, float], label: str = "val_diff (margin)") -> str:
    if not stats:
        return f"{label}: (no samples)"
    return (
        f"{label}: mean={stats['mean']:.4f} std={stats['std']:.4f} "
        f"median={stats['median']:.4f} p5={stats['p5']:.4f} p95={stats['p95']:.4f}"
    )


def log_val_diff_stats(stats: Dict[str, float], log_fn: Callable[[str], None]) -> None:
    """Log margin summary in a parseable train.log format."""
    log_fn(format_val_diff_stats_line(stats))
    if not stats:
        return
    log_fn(f"validation val_diff_mean   : {stats['mean']:.4f}")
    log_fn(f"validation val_diff_median : {stats['median']:.4f}")


def log_train_diff_stats(
    stats: Dict[str, float],
    log_fn: Callable[[str], None],
    epoch_1based: int,
) -> None:
    """Log train-set DPO margin summary for one epoch."""
    log_fn("")
    log_fn(f"=== Train delta, epoch {epoch_1based} ===")
    log_fn(format_val_diff_stats_line(stats, label="train_diff (margin)"))
    if not stats:
        return
    log_fn(f"train_diff_mean   : {stats['mean']:.4f}")
    log_fn(f"train_diff_median : {stats['median']:.4f}")


def compute_val_margin_stats(
    policy_model,
    ref_model,
    tokenizer,
    val_loader,
    device: str,
    use_chat_template: bool = False,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """Compute validation DPO margin stats (full val set unless max_batches is set)."""
    dist = compute_val_delta_distributions(
        policy_model,
        ref_model,
        tokenizer,
        val_loader,
        device,
        use_chat_template=use_chat_template,
        max_batches=max_batches,
    )
    return summarize_margin(dist["diff"])


def log_val_diff_from_loader(
    policy_model,
    ref_model,
    tokenizer,
    val_loader,
    device: str,
    log_fn: Callable[[str], None],
    use_chat_template: bool = False,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """Compute and log validation DPO margin; failures are logged, not raised."""
    try:
        stats = compute_val_margin_stats(
            policy_model,
            ref_model,
            tokenizer,
            val_loader,
            device,
            use_chat_template=use_chat_template,
            max_batches=max_batches,
        )
        log_val_diff_stats(stats, log_fn)
        return stats
    except Exception as exc:
        log_fn(
            f"validation val_diff: FAILED ({type(exc).__name__}: {exc}); "
            "continuing without val_diff metric"
        )
        return {}


def compute_val_delta_distributions(
    policy_model,
    ref_model,
    tokenizer,
    val_loader,
    device: str,
    use_chat_template: bool = False,
    max_batches: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    For each (chosen, rejected) pair on val:
      - delta_theta = logp_theta(chosen|x) - logp_theta(rejected|x)
      - delta_ref   = logp_ref(chosen|x)   - logp_ref(rejected|x)
      - diff        = delta_theta - delta_ref  (DPO margin)

    Returns a dict of three np.ndarray over all collected pairs.
    """
    policy_model.eval()
    ref_model.eval()

    dtheta_chunks: List[np.ndarray] = []
    dref_chunks: List[np.ndarray] = []
    diff_chunks: List[np.ndarray] = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            prompts = batch["prompt"]
            chosen = batch["chosen"]
            rejected = batch["rejected"]

            logp_c_pi = get_logps(
                policy_model,
                tokenizer,
                prompts,
                chosen,
                device,
                MAX_PROMPT_LEN,
                MAX_FULL_LEN,
                use_chat_template=use_chat_template,
            )
            logp_r_pi = get_logps(
                policy_model,
                tokenizer,
                prompts,
                rejected,
                device,
                MAX_PROMPT_LEN,
                MAX_FULL_LEN,
                use_chat_template=use_chat_template,
            )
            logp_c_ref = get_logps(
                ref_model,
                tokenizer,
                prompts,
                chosen,
                device,
                MAX_PROMPT_LEN,
                MAX_FULL_LEN,
                use_chat_template=use_chat_template,
            )
            logp_r_ref = get_logps(
                ref_model,
                tokenizer,
                prompts,
                rejected,
                device,
                MAX_PROMPT_LEN,
                MAX_FULL_LEN,
                use_chat_template=use_chat_template,
            )

            delta_theta = logp_c_pi - logp_r_pi
            delta_ref = logp_c_ref - logp_r_ref
            diff = delta_theta - delta_ref

            # .float() before numpy: policy/ref often run in bf16
            dtheta_chunks.append(delta_theta.detach().float().cpu().numpy())
            dref_chunks.append(delta_ref.detach().float().cpu().numpy())
            diff_chunks.append(diff.detach().float().cpu().numpy())

    if not dtheta_chunks:
        empty = np.array([], dtype=np.float64)
        return {"delta_theta": empty, "delta_ref": empty, "diff": empty}

    return {
        "delta_theta": np.concatenate(dtheta_chunks),
        "delta_ref": np.concatenate(dref_chunks),
        "diff": np.concatenate(diff_chunks),
    }
