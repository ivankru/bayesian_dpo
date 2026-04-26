from typing import Any, Dict, Protocol, Tuple

import torch


LossResult = Tuple[torch.Tensor, float]
SoftLossResult = Tuple[torch.Tensor, float, Dict[str, Any]]


class DpoLossFn(Protocol):
    def __call__(
        self,
        batch,
        tokenizer,
        policy_model,
        ref_model,
        device: str,
        **kwargs,
    ) -> LossResult | SoftLossResult:
        ...
