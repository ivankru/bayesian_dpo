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

from utils.config import MAX_PROMPT_LEN, MAX_FULL_LEN
from utils.metrics import get_logps

# Защита от log(0) в BCE при soft DPO
_BCE_EPS = 1e-11
# Ограничение beta*diff: защита от взрыва градиентов; занижение soft_B в тесте эквивалентности даёт в основном clamp s (BCE eps), не logit
_SOFT_DPO_LOGIT_CLIP = 20.0


def _logps(
    model,
    tokenizer,
    prompts,
    responses,
    device,
    use_chat_template: bool = False,
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
    use_chat_template: bool = False,
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


# def soft_dpo_loss_bce(
#     batch,
#     tokenizer,
#     policy_model,
#     ref_model,
#     device: str,
#     beta: float = 0.1,
#     use_bayes: bool = False,
#     use_chat_template: bool = False,
#     **kwargs,
# ):
#     """
#     Soft DPO: batch с полями prompt, resp1, resp2, p, p_bayes.
#     use_bayes: если True, целевая вероятность p_bayes, иначе p.
#     Возвращает (loss, kl_approx).
#     """
#     prompts = batch["prompt"]
#     resp1 = batch["resp1"]
#     resp2 = batch["resp2"]
#     target_key = "p_bayes" if use_bayes else "p"
#     p_target = torch.tensor(batch[target_key], dtype=torch.float32, device=device)

#     logp_1 = _logps(policy_model, tokenizer, prompts, resp1, device, use_chat_template)
#     logp_2 = _logps(policy_model, tokenizer, prompts, resp2, device, use_chat_template)
#     with torch.no_grad():
#         logp_1_ref = _logps(ref_model, tokenizer, prompts, resp1, device, use_chat_template)
#         logp_2_ref = _logps(ref_model, tokenizer, prompts, resp2, device, use_chat_template)

#     diff = (logp_2 - logp_1) - (logp_2_ref - logp_1_ref)
#     # Ограничиваем logit, чтобы sigmoid не уходил в 0/1 и не было взрыва BCE и градиентов
#     logit = beta * diff
#     #logit = logit.clamp(-_SOFT_DPO_LOGIT_CLIP, _SOFT_DPO_LOGIT_CLIP)
#     s = torch.sigmoid(logit)
#     p_target = p_target.to(dtype=s.dtype)
#     # Защита от log(0): s в (eps, 1-eps) для численной стабильности BCE
#     #s = s.clamp(_BCE_EPS, 1.0 - _BCE_EPS)
#     # BCE(s, p) = -[p*log(s) + (1-p)*log(1-s)] по элементам, затем .mean()
#     loss = -(p_target * torch.log(s + _BCE_EPS) + (1.0 - p_target) * torch.log(1.0 - s + _BCE_EPS))
#     loss = loss.mean()
#     # то же определение, что и в hard: средний log π/ref по паре ответов (может быть < 0 или большим)
#     kl_approx = 0.5 * (
#         (logp_1 - logp_1_ref).mean().item() + (logp_2 - logp_2_ref).mean().item()
#     )
#     return loss, kl_approx



def soft_dpo_loss(
    batch,
    tokenizer,
    policy_model,
    ref_model,
    device: str,
    beta: float = 0.1,
    use_bayes: bool = False,
    lambda_label: float = 1.0,
    use_chat_template: bool = False,
    **kwargs,
):
    """
    Anchored Soft-DPO (формула (9) из ADPO):
    loss = softplus(beta * diff) - p_target * beta * diff,
    где diff = (Δ_theta - Δ_ref).
    batch: prompt, resp1, resp2, p, p_bayes.
    lambda_label in [0, 1]: при 1.0 цель — чистые метки p_gt; иначе смешивание с p_pred_cached (учитель, посчитанный до эпохи).
    Возвращает (loss, kl_approx, diag): diag — dict с numpy 1d для логов align (target_shift; gap_abs если есть p_pred_cached).
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

    # масштабируем β и, при желании, клипуем для численной стабильности
    logit = beta * diff
    # if _SOFT_DPO_LOGIT_CLIP is not None:
    #     logit = logit.clamp(-_SOFT_DPO_LOGIT_CLIP, _SOFT_DPO_LOGIT_CLIP)

    if lambda_label == 1.0:
        p_target = p_gt
    else:
        p_pred = torch.as_tensor(
            batch["p_pred_cached"], device=device, dtype=logit.dtype
        )
        p_gt_m = p_gt.to(dtype=logit.dtype)
        lam = logit.new_tensor(lambda_label)
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
        if "p_pred_cached" in batch:
            pp = torch.as_tensor(
                batch["p_pred_cached"], dtype=torch.float32, device=device
            )
            diag["gap_abs"] = (p_gt.detach() - pp.detach()).abs().float().cpu().numpy()
        else:
            diag["gap_abs"] = None

    return loss, kl_approx, diag
