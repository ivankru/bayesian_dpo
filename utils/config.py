# -*- coding: utf-8 -*-
"""
Shared constants for DPO/soft-DPO and evaluation: length limits, base models, steer datasets.
"""
from config.base_config import MAX_FULL_LEN, MAX_PROMPT_LEN  # noqa: F401 — re-export for imports from utils.config

# Base models
BASE_MODEL_3B = "Qwen/Qwen2.5-3B-Instruct"
BASE_MODEL_4B = "Qwen/Qwen3-4B-Instruct-2507"
BASE_MODEL_7B = "Qwen/Qwen2.5-7B-Instruct"
BASE_MODEL_CHOICES = {"3b": BASE_MODEL_3B, "4b": BASE_MODEL_4B, "7b": BASE_MODEL_7B}

# --dataset names for hard_dpo_steer / soft_dpo_steer (match dataset_name in train_dpo logs)
DPO_STEER_HARD_DATASET_CHOICES = ("helpsteer3", "ultrafeedback_binarized", "hh_rlhf")
DPO_STEER_SOFT_DATASET_CHOICES = (
    "helpsteer3",
    "ultrafeedback_binarized",
    "ultrafeedback_soft",
    "openbmb",
    "hh_rlhf",
)
