import torch.nn.functional as F

from config.base_config import P_PRED_TARGET_TEMPERATURE, USE_CHAT_TEMPLATE

from .helpers import _softplus_centered, _softplus_over_beta_small_beta_approx
from .soft_common import _compute_soft_loss_common


def soft_dpo_approximation_loss(
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
    Soft-DPO approximation в масштабе old_loss / beta.
    """
    if beta <= 0:
        raise ValueError(f"beta must be > 0, got {beta!r}")
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
        per_example_loss_fn=lambda diff, logit, p_target, beta_val: (
            (
                _softplus_over_beta_small_beta_approx(diff, beta_val)
                if beta_val < approx_beta_threshold
                else F.softplus(logit) / beta_val
            )
            - p_target * diff
        ),
    )


def soft_dpo_centered_softplus_loss(
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
    Soft-DPO с centered softplus в масштабе old_loss / beta.
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
        per_example_loss_fn=lambda diff, _logit, p_target, beta_val: _softplus_centered(
            diff, beta=beta_val
        )
        - p_target * diff,
    )
