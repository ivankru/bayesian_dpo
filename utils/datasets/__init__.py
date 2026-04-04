# -*- coding: utf-8 -*-
"""
Загрузка и преобразование датасетов для DPO / soft-DPO.
HelpSteer3, UltraFeedback (бинарный / score-soft), openbmb/UltraFeedback, PKU processed HH-RLHF.
"""
from .helpsteer_hard import build_dpo_datasets
from .helpsteer_soft import build_helpsteer3_soft_datasets
from .ultrafeedback_hard import build_dpo_datasets_ultrafeedback
from .ultrafeedback_soft import (
    build_ultrafeedback_binarized_soft_datasets,
    build_ultrafeedback_score_soft_datasets,
)
from .openbmb import build_openbmb_soft_datasets
from .hh_rlf_pku import (
    build_dpo_datasets_hh_rlhf,
    build_hh_rlhf_soft_datasets,
    build_hh_rlhf_soft_steer_datasets,
    extract_pair_hh_soft,
)
from .common import precompute_p_pred_cached

__all__ = [
    "build_dpo_datasets",
    "build_dpo_datasets_ultrafeedback",
    "build_helpsteer3_soft_datasets",
    "build_ultrafeedback_binarized_soft_datasets",
    "build_ultrafeedback_score_soft_datasets",
    "build_openbmb_soft_datasets",
    "build_hh_rlhf_soft_datasets",
    "build_hh_rlhf_soft_steer_datasets",
    "build_dpo_datasets_hh_rlhf",
    "extract_pair_hh_soft",
    "precompute_p_pred_cached",
]
