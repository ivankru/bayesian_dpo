from typing import Any, Callable

import torch

from .helpers import _logps


def _prepare_soft_batch_tensors(
    batch: dict[str, Any],
    tokenizer,
    policy_model,
    ref_model,
    device: str,
    *,
    use_bayes: bool,
    use_chat_template: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

    return logp_1, logp_2, logp_1_ref, logp_2_ref, p_gt


def _build_soft_targets(
    batch: dict[str, Any],
    *,
    p_gt: torch.Tensor,
    diff: torch.Tensor,
    beta: float,
    lambda_label: float,
    p_pred_target_temperature: float,
    p_pred_teacher_blend: float,
    device: str,
) -> torch.Tensor:
    if not 0.0 <= lambda_label <= 1.0:
        raise ValueError(f"lambda_label must be in [0, 1], got {lambda_label!r}")

    if lambda_label == 1.0:
        return p_gt

    p_gt_m = p_gt.to(dtype=diff.dtype)
    lam = diff.new_tensor(lambda_label)
    if "p_pred_teacher" in batch:
        p_teacher = torch.as_tensor(
            batch["p_pred_teacher"], device=device, dtype=diff.dtype
        )
        T = float(p_pred_target_temperature)
        if T <= 0:
            raise ValueError(
                f"p_pred_target_temperature must be > 0, got {p_pred_target_temperature!r}"
            )
        p_pred_i = torch.sigmoid((beta * diff) / T).detach()
        w = float(p_pred_teacher_blend)
        if not 0.0 <= w <= 1.0:
            raise ValueError(f"p_pred_teacher_blend must be in [0, 1], got {p_pred_teacher_blend!r}")
        p_pred = w * p_teacher + (1.0 - w) * p_pred_i
    else:
        p_pred = torch.as_tensor(batch["p_pred_cached"], device=device, dtype=diff.dtype)
    return lam * p_gt_m + (1.0 - lam) * p_pred


def _build_soft_diag(
    batch: dict[str, Any], *, p_gt: torch.Tensor, p_target: torch.Tensor, device: str
) -> dict[str, Any]:
    with torch.no_grad():
        ts = (p_target.detach() - p_gt.detach()).abs().float().cpu().numpy()
        diag: dict[str, Any] = {"target_shift": ts}
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
    return diag


def _kl_approx(logp_1: torch.Tensor, logp_2: torch.Tensor, logp_1_ref: torch.Tensor, logp_2_ref: torch.Tensor) -> float:
    return 0.5 * (
        (logp_1 - logp_1_ref).mean().item()
        + (logp_2 - logp_2_ref).mean().item()
    )


def _compute_soft_loss_common(
    batch: dict[str, Any],
    tokenizer,
    policy_model,
    ref_model,
    device: str,
    *,
    beta: float,
    use_bayes: bool,
    lambda_label: float,
    use_chat_template: bool,
    p_pred_target_temperature: float,
    p_pred_teacher_blend: float,
    per_example_loss_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, float], torch.Tensor],
) -> tuple[torch.Tensor, float, dict[str, Any]]:
    logp_1, logp_2, logp_1_ref, logp_2_ref, p_gt = _prepare_soft_batch_tensors(
        batch,
        tokenizer,
        policy_model,
        ref_model,
        device,
        use_bayes=use_bayes,
        use_chat_template=use_chat_template,
    )
    delta_theta = logp_1 - logp_2
    delta_ref = logp_1_ref - logp_2_ref
    diff = delta_theta - delta_ref
    logit = beta * diff
    p_target = _build_soft_targets(
        batch,
        p_gt=p_gt,
        diff=diff,
        beta=beta,
        lambda_label=lambda_label,
        p_pred_target_temperature=p_pred_target_temperature,
        p_pred_teacher_blend=p_pred_teacher_blend,
        device=device,
    )
    loss_per_example = per_example_loss_fn(diff, logit, p_target, beta)
    loss = loss_per_example.mean()
    kl = _kl_approx(logp_1, logp_2, logp_1_ref, logp_2_ref)
    diag = _build_soft_diag(batch, p_gt=p_gt, p_target=p_target, device=device)
    return loss, kl, diag
