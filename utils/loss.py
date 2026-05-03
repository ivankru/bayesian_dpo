# -*- coding: utf-8 -*-
"""
Back-compat layer for loss functions.

New implementations live in `utils.losses.*`.
This module remains for legacy imports:
  from utils.loss import hard_dpo_loss, soft_dpo_loss, soft_dpo_loss_alt
"""

from utils.losses import (
    DpoLossFn,
    LOSS_REGISTRY,
    LossResult,
    SoftLossResult,
    _logps,
    _softplus_centered,
    _softplus_over_beta_small_beta_approx,
    get_loss,
    hard_dpo_loss,
    soft_dpo_approximation_loss,
    soft_dpo_centered_softplus_loss,
    soft_dpo_classic_loss,
    soft_dpo_loss,
    soft_dpo_loss_alt,
    soft_dpo_loss_alt_centered,
)

__all__ = [
    "DpoLossFn",
    "LossResult",
    "SoftLossResult",
    "LOSS_REGISTRY",
    "get_loss",
    "_logps",
    "_softplus_centered",
    "_softplus_over_beta_small_beta_approx",
    "hard_dpo_loss",
    "soft_dpo_classic_loss",
    "soft_dpo_approximation_loss",
    "soft_dpo_centered_softplus_loss",
    "soft_dpo_loss",
    "soft_dpo_loss_alt",
    "soft_dpo_loss_alt_centered",
]
# -*- coding: utf-8 -*-
"""
DPO loss: hard (chosen/rejected) and soft (resp1, resp2, p / p_bayes).
All functions return (loss, kl_approx).

kl_approx: not true KL(π||ref); batch mean 0.5*(mean(log π - log ref)_1 + mean(log π - log ref)_2).
Computed on fixed responses in the batch, so it can be negative (π assigns less mass
to those responses than ref) or very large under strong drift — expected.
"""
import torch
import torch.nn.functional as F
import math

from config.base_config import P_PRED_TARGET_TEMPERATURE, USE_CHAT_TEMPLATE
from utils.config import MAX_PROMPT_LEN, MAX_FULL_LEN
from utils.metrics import get_logps


def _softplus_over_beta_small_beta_approx(diff: torch.Tensor, beta: float) -> torch.Tensor:
    """
    Stable approximation for softplus(beta * diff) / beta using reduced form:
      ln(1 + exp(beta*x)) / beta ~= ln(2)/beta,                if x == 0
                                  x / (1 - exp(-(beta*x)/ln2)), otherwise
    """
    if beta <= 0:
        raise ValueError(f"beta must be > 0 for scaled softplus, got {beta!r}")
    ln2 = diff.new_tensor(0.6931471805599453)
    beta_t = diff.new_tensor(beta)
    denom = -torch.expm1(-(beta_t * diff) / ln2)
    approx = diff / denom
    near_zero = diff.abs() < 1e-12
    return torch.where(near_zero, ln2 / beta_t, approx)


def _softplus_centered(
    x: torch.Tensor, beta: float = 1.0, threshold: float = 20.0
) -> torch.Tensor:
    """
    Centered softplus:
      (1/beta) * log(1 + exp(beta*x)) - log(2)/beta.

    Numerically stable via F.softplus(beta=..., threshold=...).
    For beta -> 0, returns the exact limit x/2.
    """
    if beta == 0:
        return 0.5 * x
    if beta < 0:
        raise ValueError(f"beta must be >= 0 for centered scaled softplus, got {beta!r}")
    sp = F.softplus(x, beta=beta, threshold=threshold)
    return sp - (math.log(2.0) / beta)


def _logps(
    model,
    tokenizer,
    prompts,
    responses,
    device,
    use_chat_template: bool = USE_CHAT_TEMPLATE,
):
    return get_logps(
        model,
        tokenizer,
        prompts,
        responses,
        device,
        MAX_PROMPT_LEN,
        MAX_FULL_LEN,
        use_chat_template=use_chat_template,
    )


def hard_dpo_loss(
    batch,
    tokenizer,
    policy_model,
    ref_model,
    device: str,
    beta: float = 0.1,
    use_chat_template: bool = USE_CHAT_TEMPLATE,
    **kwargs,
):
    """
    Hard DPO: batch with fields prompt, chosen, rejected.
    Returns (loss, kl_approx).
    """
    prompts = batch["prompt"]
    chosen = batch["chosen"]
    rejected = batch["rejected"]

    logp_c = _logps(policy_model, tokenizer, prompts, chosen, device, use_chat_template)
    logp_r = _logps(policy_model, tokenizer, prompts, rejected, device, use_chat_template)
    with torch.no_grad():
        logp_c_ref = _logps(ref_model, tokenizer, prompts, chosen, device, use_chat_template)
        logp_r_ref = _logps(ref_model, tokenizer, prompts, rejected, device, use_chat_template)

    diff = (logp_c - logp_r) - (logp_c_ref - logp_r_ref)
    loss = -F.logsigmoid(beta * diff).mean()
    # proxy "mean log π/ref" over chosen and rejected (not true KL, can be < 0)
    kl_approx = 0.5 * (
        (logp_c - logp_c_ref).mean().item() + (logp_r - logp_r_ref).mean().item()
    )
    return loss, kl_approx


def soft_dpo_loss(
    batch,
    tokenizer,
    policy_model,
    ref_model,
    device: str,
    beta: float = 0.1,
    use_bayes: bool = False,
    lambda_label: float = 1.0,
    use_chat_template: bool = USE_CHAT_TEMPLATE,
    p_pred_target_temperature: float = P_PRED_TARGET_TEMPERATURE,
    p_pred_teacher_blend: float = 0.0,
    **kwargs,
):
    """
    Anchored Soft-DPO (ADPO eq. (9)):
    loss = softplus(beta * diff) - p_target * beta * diff,
    where diff = (Δ_theta - Δ_ref).
    batch: prompt, resp1, resp2, p, p_bayes.
    lambda_label in [0, 1]: at 1.0 target is pure labels p_gt; else blend with p_pred
    (either p_pred_cached for the epoch, or anchor mode with p_pred_teacher — see below).
    p_pred_target_temperature (T): for p_pred_i = σ((beta*diff)/T) in anchor mode; softplus logit stays beta*diff without T.
    p_pred_teacher_blend (w ∈ [0,1]): when p_pred_teacher is supplied from the training loop — anchor mode uses w=0.5
    on all tail steps: p_pred = w*p_teacher + (1-w)*p_pred_i with p_pred_i = σ((beta*diff)/T).

    Returns (loss, kl_approx, diag): diag is a dict of numpy 1d for align logs (target_shift; gap_abs when p_pred exists).
    """
    if not 0.0 <= lambda_label <= 1.0:
        raise ValueError(f"lambda_label must be in [0, 1], got {lambda_label!r}")

    prompts = batch["prompt"]
    resp1 = batch["resp1"]
    resp2 = batch["resp2"]
    target_key = "p_bayes" if use_bayes else "p"
    p_gt = torch.as_tensor(batch[target_key], dtype=torch.float32, device=device)

    # log π_θ
    logp_1 = _logps(policy_model, tokenizer, prompts, resp1, device, use_chat_template)
    logp_2 = _logps(policy_model, tokenizer, prompts, resp2, device, use_chat_template)

    # log π_ref (no grad)
    with torch.no_grad():
        logp_1_ref = _logps(ref_model, tokenizer, prompts, resp1, device, use_chat_template)
        logp_2_ref = _logps(ref_model, tokenizer, prompts, resp2, device, use_chat_template)

    # Δ_theta - Δ_ref (ordering as in the paper:
    # Δ = (logπ1 - logπ2) - (logπ1_ref - logπ2_ref))
    delta_theta = logp_1 - logp_2
    delta_ref = logp_1_ref - logp_2_ref
    diff = delta_theta - delta_ref  # shape: [batch]

    logit = beta * diff

    if lambda_label == 1.0:
        p_target = p_gt
    else:
        p_gt_m = p_gt.to(dtype=logit.dtype)
        lam = logit.new_tensor(lambda_label)
        if "p_pred_teacher" in batch:
            # Anchor teacher: probabilities p_pred_teacher fixed during warmup epochs
            # (σ(beta*diff) without temperature, as in precompute_p_pred_teacher).
            p_teacher = torch.as_tensor(
                batch["p_pred_teacher"], device=device, dtype=logit.dtype
            )
            # Current soft prediction from the same logits as the loss, but only for p_target:
            #   p_pred_i = σ((beta * diff) / T).
            # Important: softplus argument and p_target * logit still use logit = beta*diff
            # without T — only the blended probability in p_target changes, not DPO potential curvature.
            #
            # Equivalent scaling: σ((β·diff)/T) = σ((β/T)·diff), so for this probability
            # raising T is like shrinking the effective coefficient at the same diff:
            # T=2 matches σ((β/2)·diff). Hence T>1 softens extreme p_pred_i
            # (toward 0.5), damping noise from small diff swings without changing β in the main logit.
            #
            # Must .detach(): p_pred_i is the *target* in the BCE-like term softplus(logit) − p_target·logit,
            # it should act as a fixed label (like p_gt, p_teacher, p_pred_cached). If gradients flow,
            # the derivative picks up −(1−λ)(1−w)·∂p_pred_i/∂θ · β·diff, which changes
            # DPO curvature (not ADPO eq. (9) nor the docstring contract).
            T = float(p_pred_target_temperature)
            if T <= 0:
                raise ValueError(
                    f"p_pred_target_temperature must be > 0, got {p_pred_target_temperature!r}"
                )
            p_pred_i = torch.sigmoid((beta * diff) / T).detach()
            w = float(p_pred_teacher_blend)
            if not 0.0 <= w <= 1.0:
                raise ValueError(f"p_pred_teacher_blend must be in [0, 1], got {p_pred_teacher_blend!r}")
            # w=0.5 (anchor mode from train_dpo): half p_teacher, half p_pred_i = σ((beta*diff)/T).
            p_pred = w * p_teacher + (1.0 - w) * p_pred_i
        else:
            p_pred = torch.as_tensor(
                batch["p_pred_cached"], device=device, dtype=logit.dtype
            )
        p_target = lam * p_gt_m + (1.0 - lam) * p_pred

    # Eq. (9): softplus - q * logit
    # softplus(x) = log(1 + exp(x)) — numerically stable primitive
    loss_per_example = F.softplus(logit) - p_target * logit
    loss = loss_per_example.mean()

    # same kl_approx as before
    kl_approx = 0.5 * (
        (logp_1 - logp_1_ref).mean().item()
        + (logp_2 - logp_2_ref).mean().item()
    )

    with torch.no_grad():
        ts = (p_target.detach() - p_gt.detach()).abs().float().cpu().numpy()
        diag: dict = {"target_shift": ts}
        # gap_abs — pure diagnostic |p_gt - p_pred_*| (teacher or cached),
        # independent of lambda_label. Useful even at λ=1 (warmup epochs after
        # teacher fix): shows how far the teacher drifts from labels.
        if "p_pred_teacher" in batch:
            pp = torch.as_tensor(
                batch["p_pred_teacher"], dtype=torch.float32, device=device
            )
            diag["gap_abs"] = (p_gt.detach() - pp.detach()).abs().float().cpu().numpy()
        elif "p_pred_cached" in batch:
            pp = torch.as_tensor(
                batch["p_pred_cached"], dtype=torch.float32, device=device
            )
            diag["gap_abs"] = (p_gt.detach() - pp.detach()).abs().float().cpu().numpy()
        else:
            diag["gap_abs"] = None

    return loss, kl_approx, diag


def soft_dpo_loss_alt(
    batch,
    tokenizer,
    policy_model,
    ref_model,
    device: str,
    beta: float = 0.1,
    use_bayes: bool = False,
    lambda_label: float = 1.0,
    use_chat_template: bool = USE_CHAT_TEMPLATE,
    p_pred_target_temperature: float = P_PRED_TARGET_TEMPERATURE,
    p_pred_teacher_blend: float = 0.0,
    approx_beta_threshold: float = 0.1,
    **kwargs,
):
    """
    Alternative Soft-DPO scaled like old_loss / beta.
    For beta >= approx_beta_threshold compute exactly:
      loss = [softplus(beta*diff) - p_target*beta*diff] / beta
    For beta < approx_beta_threshold use a stable approximation
    for softplus(beta*diff)/beta with reduced beta.
    """
    if not 0.0 <= lambda_label <= 1.0:
        raise ValueError(f"lambda_label must be in [0, 1], got {lambda_label!r}")

    prompts = batch["prompt"]
    resp1 = batch["resp1"]
    resp2 = batch["resp2"]
    target_key = "p_bayes" if use_bayes else "p"
    p_gt = torch.as_tensor(batch[target_key], dtype=torch.float32, device=device)

    logp_1 = _logps(policy_model, tokenizer, prompts, resp1, device, use_chat_template)
    logp_2 = _logps(policy_model, tokenizer, prompts, resp2, device, use_chat_template)

    with torch.no_grad():
        logp_1_ref = _logps(ref_model, tokenizer, prompts, resp1, device, use_chat_template)
        logp_2_ref = _logps(ref_model, tokenizer, prompts, resp2, device, use_chat_template)

    delta_theta = logp_1 - logp_2
    delta_ref = logp_1_ref - logp_2_ref
    diff = delta_theta - delta_ref
    logit = beta * diff

    if lambda_label == 1.0:
        p_target = p_gt
    else:
        p_gt_m = p_gt.to(dtype=logit.dtype)
        lam = logit.new_tensor(lambda_label)
        if "p_pred_teacher" in batch:
            p_teacher = torch.as_tensor(
                batch["p_pred_teacher"], device=device, dtype=logit.dtype
            )
            T = float(p_pred_target_temperature)
            if T <= 0:
                raise ValueError(
                    f"p_pred_target_temperature must be > 0, got {p_pred_target_temperature!r}"
                )
            p_pred_i = torch.sigmoid((beta * diff) / T).detach()
            w = float(p_pred_teacher_blend)
            if not 0.0 <= w <= 1.0:
                raise ValueError(
                    f"p_pred_teacher_blend must be in [0, 1], got {p_pred_teacher_blend!r}"
                )
            p_pred = w * p_teacher + (1.0 - w) * p_pred_i
        else:
            p_pred = torch.as_tensor(
                batch["p_pred_cached"], device=device, dtype=logit.dtype
            )
        p_target = lam * p_gt_m + (1.0 - lam) * p_pred

    if beta <= 0:
        raise ValueError(f"beta must be > 0, got {beta!r}")
    if beta < approx_beta_threshold:
        softplus_over_beta = _softplus_over_beta_small_beta_approx(diff, beta)
    else:
        softplus_over_beta = F.softplus(logit) / beta
    loss_per_example = softplus_over_beta - p_target * diff
    loss = loss_per_example.mean()

    kl_approx = 0.5 * (
        (logp_1 - logp_1_ref).mean().item()
        + (logp_2 - logp_2_ref).mean().item()
    )

    with torch.no_grad():
        ts = (p_target.detach() - p_gt.detach()).abs().float().cpu().numpy()
        diag: dict = {"target_shift": ts}
        if "p_pred_teacher" in batch:
            pp = torch.as_tensor(
                batch["p_pred_teacher"], dtype=torch.float32, device=device
            )
            diag["gap_abs"] = (p_gt.detach() - pp.detach()).abs().float().cpu().numpy()
        elif "p_pred_cached" in batch:
            pp = torch.as_tensor(
                batch["p_pred_cached"], dtype=torch.float32, device=device
            )
            diag["gap_abs"] = (p_gt.detach() - pp.detach()).abs().float().cpu().numpy()
        else:
            diag["gap_abs"] = None

    return loss, kl_approx, diag


def soft_dpo_loss_alt_centered(
    batch,
    tokenizer,
    policy_model,
    ref_model,
    device: str,
    beta: float = 0.1,
    use_bayes: bool = False,
    lambda_label: float = 1.0,
    use_chat_template: bool = USE_CHAT_TEMPLATE,
    p_pred_target_temperature: float = P_PRED_TARGET_TEMPERATURE,
    p_pred_teacher_blend: float = 0.0,
    **kwargs,
):
    """
    Alternative Soft-DPO in old_loss / beta scale with centered softplus:
      loss = [softplus(beta*diff) - ln(2)] / beta - p_target*diff
    Centering subtracts ln(2)/beta and does not change parameter minima.
    """
    if not 0.0 <= lambda_label <= 1.0:
        raise ValueError(f"lambda_label must be in [0, 1], got {lambda_label!r}")

    prompts = batch["prompt"]
    resp1 = batch["resp1"]
    resp2 = batch["resp2"]
    target_key = "p_bayes" if use_bayes else "p"
    p_gt = torch.as_tensor(batch[target_key], dtype=torch.float32, device=device)

    logp_1 = _logps(policy_model, tokenizer, prompts, resp1, device, use_chat_template)
    logp_2 = _logps(policy_model, tokenizer, prompts, resp2, device, use_chat_template)

    with torch.no_grad():
        logp_1_ref = _logps(ref_model, tokenizer, prompts, resp1, device, use_chat_template)
        logp_2_ref = _logps(ref_model, tokenizer, prompts, resp2, device, use_chat_template)

    delta_theta = logp_1 - logp_2
    delta_ref = logp_1_ref - logp_2_ref
    diff = delta_theta - delta_ref
    logit = beta * diff

    if lambda_label == 1.0:
        p_target = p_gt
    else:
        p_gt_m = p_gt.to(dtype=logit.dtype)
        lam = logit.new_tensor(lambda_label)
        if "p_pred_teacher" in batch:
            p_teacher = torch.as_tensor(
                batch["p_pred_teacher"], device=device, dtype=logit.dtype
            )
            T = float(p_pred_target_temperature)
            if T <= 0:
                raise ValueError(
                    f"p_pred_target_temperature must be > 0, got {p_pred_target_temperature!r}"
                )
            p_pred_i = torch.sigmoid((beta * diff) / T).detach()
            w = float(p_pred_teacher_blend)
            if not 0.0 <= w <= 1.0:
                raise ValueError(
                    f"p_pred_teacher_blend must be in [0, 1], got {p_pred_teacher_blend!r}"
                )
            p_pred = w * p_teacher + (1.0 - w) * p_pred_i
        else:
            p_pred = torch.as_tensor(
                batch["p_pred_cached"], device=device, dtype=logit.dtype
            )
        p_target = lam * p_gt_m + (1.0 - lam) * p_pred

    centered_softplus = _softplus_centered(diff, beta=beta)
    loss_per_example = centered_softplus - p_target * diff
    loss = loss_per_example.mean()

    kl_approx = 0.5 * (
        (logp_1 - logp_1_ref).mean().item()
        + (logp_2 - logp_2_ref).mean().item()
    )

    with torch.no_grad():
        ts = (p_target.detach() - p_gt.detach()).abs().float().cpu().numpy()
        diag: dict = {"target_shift": ts}
        if "p_pred_teacher" in batch:
            pp = torch.as_tensor(
                batch["p_pred_teacher"], dtype=torch.float32, device=device
            )
            diag["gap_abs"] = (p_gt.detach() - pp.detach()).abs().float().cpu().numpy()
        elif "p_pred_cached" in batch:
            pp = torch.as_tensor(
                batch["p_pred_cached"], dtype=torch.float32, device=device
            )
            diag["gap_abs"] = (p_gt.detach() - pp.detach()).abs().float().cpu().numpy()
        else:
            diag["gap_abs"] = None

    return loss, kl_approx, diag
