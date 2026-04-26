import math

import torch
import torch.nn.functional as F

from config.base_config import USE_CHAT_TEMPLATE
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
