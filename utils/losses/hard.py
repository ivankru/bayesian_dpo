import torch
import torch.nn.functional as F

from config.base_config import USE_CHAT_TEMPLATE

from .helpers import _logps


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
