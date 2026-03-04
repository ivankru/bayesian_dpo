# -*- coding: utf-8 -*-
"""
DPO loss: hard (chosen/rejected) и soft (resp1, resp2, p / p_bayes).
Все функции возвращают (loss, kl_approx).
"""
import torch
import torch.nn.functional as F

from utils.config import MAX_PROMPT_LEN, MAX_FULL_LEN
from utils.metrics import get_logps


def _logps(model, tokenizer, prompts, responses, device):
    return get_logps(model, tokenizer, prompts, responses, device, MAX_PROMPT_LEN, MAX_FULL_LEN)


def hard_dpo_loss(
    batch,
    tokenizer,
    policy_model,
    ref_model,
    device: str,
    beta: float = 0.1,
    **kwargs,
):
    """
    Hard DPO: batch с полями prompt, chosen, rejected.
    Возвращает (loss, kl_approx).
    """
    prompts = batch["prompt"]
    chosen = batch["chosen"]
    rejected = batch["rejected"]

    logp_c = _logps(policy_model, tokenizer, prompts, chosen, device)
    logp_r = _logps(policy_model, tokenizer, prompts, rejected, device)
    with torch.no_grad():
        logp_c_ref = _logps(ref_model, tokenizer, prompts, chosen, device)
        logp_r_ref = _logps(ref_model, tokenizer, prompts, rejected, device)

    diff = (logp_c - logp_r) - (logp_c_ref - logp_r_ref)
    loss = -F.logsigmoid(beta * diff).mean()
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
    **kwargs,
):
    """
    Soft DPO: batch с полями prompt, resp1, resp2, p, p_bayes.
    use_bayes: если True, целевая вероятность p_bayes, иначе p.
    Возвращает (loss, kl_approx).
    """
    prompts = batch["prompt"]
    resp1 = batch["resp1"]
    resp2 = batch["resp2"]
    target_key = "p_bayes" if use_bayes else "p"
    p_target = torch.tensor(batch[target_key], dtype=torch.float32, device=device)

    logp_1 = _logps(policy_model, tokenizer, prompts, resp1, device)
    logp_2 = _logps(policy_model, tokenizer, prompts, resp2, device)
    with torch.no_grad():
        logp_1_ref = _logps(ref_model, tokenizer, prompts, resp1, device)
        logp_2_ref = _logps(ref_model, tokenizer, prompts, resp2, device)

    diff = (logp_2 - logp_1) - (logp_2_ref - logp_1_ref)
    s = torch.sigmoid(beta * diff)
    p_target = p_target.to(dtype=s.dtype)
    loss = F.binary_cross_entropy(s, p_target)
    kl_approx = 0.5 * (
        (logp_1 - logp_1_ref).mean().item() + (logp_2 - logp_2_ref).mean().item()
    )
    return loss, kl_approx
