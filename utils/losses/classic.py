import torch.nn.functional as F

from config.base_config import P_PRED_TARGET_TEMPERATURE, USE_CHAT_TEMPLATE

from .soft_common import _compute_soft_loss_common


def soft_dpo_classic_loss(
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
    """
    return _compute_soft_loss_common(
        batch,
        tokenizer,
        policy_model,
        ref_model,
        device,
        beta=beta,
        use_bayes=use_bayes,
        lambda_label=lambda_label,
        use_chat_template=use_chat_template,
        p_pred_target_temperature=p_pred_target_temperature,
        p_pred_teacher_blend=p_pred_teacher_blend,
        per_example_loss_fn=lambda _diff, logit, p_target, _beta: F.softplus(logit) - p_target * logit,
    )
