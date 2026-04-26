from .base import DpoLossFn
from .hard import hard_dpo_loss
from .classic import soft_dpo_classic_loss
from .approximation import (
    soft_dpo_approximation_loss,
    soft_dpo_centered_softplus_loss,
)


LOSS_REGISTRY: dict[str, DpoLossFn] = {
    "hard_dpo_loss": hard_dpo_loss,
    "soft_dpo_classic_loss": soft_dpo_classic_loss,
    "soft_dpo_approximation_loss": soft_dpo_approximation_loss,
    "soft_dpo_centered_softplus_loss": soft_dpo_centered_softplus_loss,
    # Backward-compatible aliases.
    "soft_dpo_loss": soft_dpo_classic_loss,
    "soft_dpo_loss_alt": soft_dpo_approximation_loss,
    "soft_dpo_loss_alt_centered": soft_dpo_centered_softplus_loss,
}


def get_loss(name: str) -> DpoLossFn:
    try:
        return LOSS_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(LOSS_REGISTRY))
        raise ValueError(f"Unknown loss {name!r}. Available: {available}") from exc
