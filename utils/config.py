# -*- coding: utf-8 -*-
"""
Shared constants for DPO/soft-DPO and evaluation: length limits, base models, steer datasets.
"""
import os
from typing import Any, Dict, Optional

from config.base_config import MAX_FULL_LEN, MAX_PROMPT_LEN  # noqa: F401 — re-export for imports from utils.config

# Base models
BASE_MODEL_3B = "Qwen/Qwen2.5-3B-Instruct"
BASE_MODEL_4B = "Qwen/Qwen3-4B-Instruct-2507"
BASE_MODEL_7B = "Qwen/Qwen2.5-7B-Instruct"
BASE_MODEL_PHI4MINI = "microsoft/Phi-4-mini-instruct"
BASE_MODEL_CHOICES = {
    "3b": BASE_MODEL_3B,
    "4b": BASE_MODEL_4B,
    "7b": BASE_MODEL_7B,
    "3.8b": BASE_MODEL_PHI4MINI,
    "phi4mini": BASE_MODEL_PHI4MINI,  # alias of 3.8b
}
BASE_MODEL_HELP = (
    "Base model: 3b/7b — Qwen2.5-Instruct; 4b — Qwen3-4B-Instruct-2507; "
    "3.8b — microsoft/Phi-4-mini-instruct (~3.8B, MIT)."
)

# Phi-4-mini is native Phi3 in transformers; do not use Hub modeling_phi3.py
# (it imports LossKwargs, missing in older transformers → ImportError).
_FORCE_NATIVE_TRANSFORMERS_MODELS = frozenset({BASE_MODEL_PHI4MINI})


def hf_pretrained_kwargs(model_id: Optional[str] = None) -> Dict[str, Any]:
    """Kwargs for AutoTokenizer / AutoModel .from_pretrained.

    Token is read from HF_TOKEN or HUGGING_FACE_HUB_TOKEN if set (gated Hub models).
    """
    kwargs: Dict[str, Any] = {}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        kwargs["token"] = token
    if model_id is not None and model_id in _FORCE_NATIVE_TRANSFORMERS_MODELS:
        kwargs["trust_remote_code"] = False
    return kwargs

# --dataset names for hard_dpo_steer / soft_dpo_steer (match dataset_name in train_dpo logs)
DPO_STEER_HARD_DATASET_CHOICES = (
    "helpsteer3",
    "ultrafeedback_binarized",
    "hh_rlhf",
    "orca_dpo",
)
DPO_STEER_SOFT_DATASET_CHOICES = (
    "helpsteer3",
    "ultrafeedback_binarized",
    "ultrafeedback_soft",
    "openbmb",
    "hh_rlhf",
    "orca_dpo",
)
