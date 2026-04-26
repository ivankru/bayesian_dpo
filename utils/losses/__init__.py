from .base import DpoLossFn, LossResult, SoftLossResult
from .hard import hard_dpo_loss
from .helpers import _logps, _softplus_centered, _softplus_over_beta_small_beta_approx
from .registry import LOSS_REGISTRY, get_loss
from .classic import soft_dpo_classic_loss
from .approximation import (
    soft_dpo_approximation_loss,
    soft_dpo_centered_softplus_loss,
)

# Backward-compatible aliases.
soft_dpo_loss = soft_dpo_classic_loss
soft_dpo_loss_alt = soft_dpo_approximation_loss
soft_dpo_loss_alt_centered = soft_dpo_centered_softplus_loss

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
