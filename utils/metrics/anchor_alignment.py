# -*- coding: utf-8 -*-
"""
Агрегаты по окну шагов: |p_gt − p_pred_cached| и |p_target − p_gt| (anchor / teacher mix).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np


def _mean_top_fraction(x: np.ndarray, upper_frac: float) -> float:
    """Среднее по наибольшим ceil(upper_frac * n) значениям (upper_frac=0.1 → top 10%)."""
    n = int(x.size)
    if n == 0:
        return float("nan")
    k = max(1, int(math.ceil(upper_frac * n)))
    part = np.partition(x, n - k)[n - k :]
    return float(part.mean())


def aggregate_anchor_alignment_window(
    gap_abs_parts: List[np.ndarray],
    target_shift_parts: List[np.ndarray],
) -> Dict[str, float]:
    """
    gap_abs: фрагменты |p_gt − p_pred_cached| (если в батче был p_pred_cached).
    target_shift: фрагменты |p_target − p_gt|.
    """
    out: Dict[str, float] = {}
    if target_shift_parts:
        ts = np.concatenate(target_shift_parts, axis=0).astype(np.float64, copy=False)
        out["target_shift_mean"] = float(ts.mean())
        out["target_shift_frac_gt_0.1"] = float((ts > 0.1).mean())
    else:
        out["target_shift_mean"] = float("nan")
        out["target_shift_frac_gt_0.1"] = float("nan")

    if gap_abs_parts:
        g = np.concatenate(gap_abs_parts, axis=0).astype(np.float64, copy=False)
        out["gap_abs_mean"] = float(g.mean())
        out["gap_abs_median"] = float(np.median(g))
        out["gap_abs_p90"] = float(np.percentile(g, 90))
        out["gap_abs_mean_top10"] = _mean_top_fraction(g, 0.1)
        out["gap_abs_frac_gt_0.5"] = float((g > 0.5).mean())
        out["gap_abs_frac_gt_0.3"] = float((g > 0.3).mean())
    else:
        for k in (
            "gap_abs_mean",
            "gap_abs_median",
            "gap_abs_p90",
            "gap_abs_mean_top10",
            "gap_abs_frac_gt_0.5",
            "gap_abs_frac_gt_0.3",
        ):
            out[k] = float("nan")
    return out


def format_anchor_alignment_log(m: Dict[str, float]) -> str:
    """Одна строка для train.log (после основной строки loss)."""
    parts = [
        f"target_shift_mean={m['target_shift_mean']:.4f}",
        f"target_shift_frac_gt_0.1={m['target_shift_frac_gt_0.1']:.4f}",
    ]
    if not math.isnan(m.get("gap_abs_mean", float("nan"))):
        parts.extend(
            [
                f"gap_abs_mean={m['gap_abs_mean']:.4f}",
                f"gap_abs_median={m['gap_abs_median']:.4f}",
                f"gap_abs_p90={m['gap_abs_p90']:.4f}",
                f"gap_abs_mean_top10={m['gap_abs_mean_top10']:.4f}",
                f"gap_abs_frac_gt_0.5={m['gap_abs_frac_gt_0.5']:.4f}",
                f"gap_abs_frac_gt_0.3={m['gap_abs_frac_gt_0.3']:.4f}",
            ]
        )
    else:
        parts.append("gap_abs_*=nan (no p_pred_cached in window)")
    return "  align " + " ".join(parts)
