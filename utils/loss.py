# -*- coding: utf-8 -*-
"""
Back-compat layer for loss functions.

Новый код находится в `utils.losses.*`.
Этот модуль оставлен для старых импортов:
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
DPO loss: hard (chosen/rejected) и soft (resp1, resp2, p / p_bayes).
Все функции возвращают (loss, kl_approx).

kl_approx: это не истинная KL-дивергенция KL(π||ref), а среднее по батчу 0.5*(mean(log π - log ref)_1 + mean(log π - log ref)_2).
Считается по фиксированным ответам в батче, поэтому может быть отрицательным (π даёт меньшую массу
этим ответам, чем ref) или очень большим при сильном дрифте — это ожидаемо.
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
    Hard DPO: batch с полями prompt, chosen, rejected.
    Возвращает (loss, kl_approx).
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
    # аппроксимация "средний log π/ref" по chosen и rejected (не истинная KL, может быть < 0)
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
    Anchored Soft-DPO (формула (9) из ADPO):
    loss = softplus(beta * diff) - p_target * beta * diff,
    где diff = (Δ_theta - Δ_ref).
    batch: prompt, resp1, resp2, p, p_bayes.
    lambda_label in [0, 1]: при 1.0 цель — чистые метки p_gt; иначе смешивание с p_pred
    (либо p_pred_cached до эпохи, либо якорный режим с p_pred_teacher — см. ниже).
    p_pred_target_temperature (T): для p_pred_i = σ((beta*diff)/T) в якорном режиме; логит в softplus — beta*diff без T.
    p_pred_teacher_blend (w ∈ [0,1]): при p_pred_teacher задаётся из цикла обучения — в якорном режиме w=0.5
    на всех хвостовых шагах: p_pred = w*p_teacher + (1-w)*p_pred_i с p_pred_i = σ((beta*diff)/T).

    Возвращает (loss, kl_approx, diag): diag — dict с numpy 1d для логов align (target_shift; gap_abs если есть p_pred).
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

    # log π_ref (без градиента)
    with torch.no_grad():
        logp_1_ref = _logps(ref_model, tokenizer, prompts, resp1, device, use_chat_template)
        logp_2_ref = _logps(ref_model, tokenizer, prompts, resp2, device, use_chat_template)

    # Δ_theta - Δ_ref (здесь порядок как в статье:
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
            # Якорный учитель: зафиксированные на warmup-эпохе вероятности p_pred_teacher
            # (σ(beta*diff) без температуры, как в precompute_p_pred_teacher).
            p_teacher = torch.as_tensor(
                batch["p_pred_teacher"], device=device, dtype=logit.dtype
            )
            # Текущее «мягкое» предсказание из тех же логитов, что и в лоссе, но только для цели p_target:
            #   p_pred_i = σ((beta * diff) / T).
            # Важно: аргумент softplus и член p_target * logit по-прежнему используют logit = beta*diff
            # без T — меняется только смешиваемая в p_target вероятность, не кривизна DPO-потенциала.
            #
            # Эквивалентная интерпретация масштаба: σ((β·diff)/T) = σ((β/T)·diff), то есть для *этой*
            # вероятности повышение T эквивалентно уменьшению эффективного коэффициента при том же diff:
            # при T=2 то же самое, что σ((β/2)·diff). Именно поэтому T>1 размягчает крайние p_pred_i
            # (ближе к 0.5), снижая шум от малых колебаний diff, не трогая β в основном логите.
            #
            # Обязательно .detach(): p_pred_i — это *цель* в BCE-подобном члене softplus(logit) − p_target·logit,
            # она должна играть роль фиксированной метки (как p_gt, p_teacher, p_pred_cached). Если оставить
            # градиент — к производной добавится член −(1−λ)(1−w)·∂p_pred_i/∂θ · β·diff, и это меняет
            # кривизну DPO-потенциала (не ADPO-формула (9) и не то, что обещает docstring).
            T = float(p_pred_target_temperature)
            if T <= 0:
                raise ValueError(
                    f"p_pred_target_temperature must be > 0, got {p_pred_target_temperature!r}"
                )
            p_pred_i = torch.sigmoid((beta * diff) / T).detach()
            w = float(p_pred_teacher_blend)
            if not 0.0 <= w <= 1.0:
                raise ValueError(f"p_pred_teacher_blend must be in [0, 1], got {p_pred_teacher_blend!r}")
            # w=0.5 (якорный режим из train_dpo): половина p_teacher, половина p_pred_i = σ((beta*diff)/T).
            p_pred = w * p_teacher + (1.0 - w) * p_pred_i
        else:
            p_pred = torch.as_tensor(
                batch["p_pred_cached"], device=device, dtype=logit.dtype
            )
        p_target = lam * p_gt_m + (1.0 - lam) * p_pred

    # Формула (9): softplus - q * logit
    # softplus(x) = log(1 + exp(x)) — устойчивый примитив
    loss_per_example = F.softplus(logit) - p_target * logit
    loss = loss_per_example.mean()

    # тот же kl_approx, что был у тебя
    kl_approx = 0.5 * (
        (logp_1 - logp_1_ref).mean().item()
        + (logp_2 - logp_2_ref).mean().item()
    )

    with torch.no_grad():
        ts = (p_target.detach() - p_gt.detach()).abs().float().cpu().numpy()
        diag: dict = {"target_shift": ts}
        # gap_abs — чистая диагностика |p_gt - p_pred_*| (teacher или cached),
        # не зависит от lambda_label. Информативна и при λ=1 (warmup-эпохи после
        # фиксации teacher): показывает, насколько «учитель» расходится с метками.
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
    Альтернативный Soft-DPO в масштабе old_loss / beta.
    При beta >= approx_beta_threshold считает точно:
      loss = [softplus(beta*diff) - p_target*beta*diff] / beta
    При beta < approx_beta_threshold использует стабильную аппроксимацию
    для softplus(beta*diff)/beta с сокращенным beta.
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
    Альтернативный Soft-DPO в масштабе old_loss / beta с центрированным softplus:
      loss = [softplus(beta*diff) - ln(2)] / beta - p_target*diff
    Центрирование вычитает константу ln(2)/beta и не меняет минимум по параметрам.
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
