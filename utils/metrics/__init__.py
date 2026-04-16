# -*- coding: utf-8 -*-
"""Метрики DPO, alignment по anchor, capability retention."""

from .anchor_alignment import aggregate_anchor_alignment_window, format_anchor_alignment_log
from .capability_retention_eval import (
    EvalRow,
    build_ref_cache_metadata,
    format_capability_retention_log_lines,
    load_eval_rows,
    load_ref_texts_cache_if_compatible,
    log_mlflow_capability_metrics,
    run_retention_eval_pair,
    save_ref_texts_cache,
)
from .dpo_logps import eval_pairwise_accuracy, eval_pairwise_nll, get_logps
from .val_kl_mc import estimate_val_kl_mc, get_logps_generated
from .val_response_entropy import estimate_val_response_entropy

__all__ = [
    "aggregate_anchor_alignment_window",
    "format_anchor_alignment_log",
    "get_logps",
    "eval_pairwise_accuracy",
    "eval_pairwise_nll",
    "EvalRow",
    "build_ref_cache_metadata",
    "format_capability_retention_log_lines",
    "load_eval_rows",
    "load_ref_texts_cache_if_compatible",
    "log_mlflow_capability_metrics",
    "run_retention_eval_pair",
    "save_ref_texts_cache",
    "estimate_val_kl_mc",
    "get_logps_generated",
    "estimate_val_response_entropy",
]
