# -*- coding: utf-8 -*-
"""
Общие константы для DPO/soft-DPO и оценки: лимиты длины, базовые модели, датасеты steer.
"""
from config.base_config import MAX_FULL_LEN, MAX_PROMPT_LEN  # noqa: F401 — реэкспорт для импортов из utils.config

# Базовые модели
BASE_MODEL_3B = "Qwen/Qwen2.5-3B-Instruct"
BASE_MODEL_4B = "Qwen/Qwen3-4B-Instruct-2507"
BASE_MODEL_7B = "Qwen/Qwen2.5-7B-Instruct"
BASE_MODEL_CHOICES = {"3b": BASE_MODEL_3B, "4b": BASE_MODEL_4B, "7b": BASE_MODEL_7B}

# Имена --dataset для hard_dpo_steer / soft_dpo_steer (совпадают с dataset_name в логах train_dpo)
DPO_STEER_HARD_DATASET_CHOICES = ("helpsteer3", "ultrafeedback_binarized", "hh_rlhf")
DPO_STEER_SOFT_DATASET_CHOICES = (
    "helpsteer3",
    "ultrafeedback_binarized",
    "ultrafeedback_soft",
    "openbmb",
    "hh_rlhf",
)
