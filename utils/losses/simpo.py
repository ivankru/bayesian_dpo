import torch
import torch.nn.functional as F

from config.base_config import USE_CHAT_TEMPLATE

from .helpers import _logps


def simpo_loss(
    batch,
    tokenizer,
    policy_model,
    ref_model,
    device: str,
    beta: float = 0.1,
    gamma: float = 1.0,
    use_chat_template: bool = USE_CHAT_TEMPLATE,
    **kwargs,
):
    """
    SimPO on hard preference pairs.

    m = mean_logp(chosen) - mean_logp(rejected), where each mean is over
    exactly the completion tokens used in the corresponding log-probability sum.
    The reference model is used only for the existing drift diagnostic.
    """
    prompts = batch["prompt"]
    chosen = batch["chosen"]
    rejected = batch["rejected"]

    logp_c, len_c = _logps(
        policy_model, tokenizer, prompts, chosen, device, use_chat_template, return_lengths=True
    )
    logp_r, len_r = _logps(
        policy_model, tokenizer, prompts, rejected, device, use_chat_template, return_lengths=True
    )

    len_c_f = len_c.to(dtype=logp_c.dtype).clamp_min(1)
    len_r_f = len_r.to(dtype=logp_r.dtype).clamp_min(1)
    mean_logp_c = logp_c / len_c_f
    mean_logp_r = logp_r / len_r_f
    margin = mean_logp_c - mean_logp_r

    loss = -F.logsigmoid(beta * margin - gamma).mean()

    with torch.no_grad():
        # Keep the same train-log KL proxy as hard DPO for drift monitoring;
        # it is not part of the SimPO objective.
        logp_c_ref = _logps(ref_model, tokenizer, prompts, chosen, device, use_chat_template)
        logp_r_ref = _logps(ref_model, tokenizer, prompts, rejected, device, use_chat_template)
        kl_approx = 0.5 * (
            (logp_c.detach() - logp_c_ref).mean().item()
            + (logp_r.detach() - logp_r_ref).mean().item()
        )
        diag = {
            "diff": margin.detach().float().cpu().numpy(),
            "simpo_margin": margin.detach().float().cpu().numpy(),
            "chosen_len": len_c.detach().cpu().numpy(),
            "rejected_len": len_r.detach().cpu().numpy(),
        }
    return loss, kl_approx, diag
