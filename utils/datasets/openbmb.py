# -*- coding: utf-8 -*-
"""openbmb/UltraFeedback: soft DPO (4 критерия → p, p_bayes)."""
from typing import Dict, Any, Optional

from datasets import load_dataset, Dataset

from .common import ultrafeedback_message_to_response

CRITERIA = ("helpfulness", "honesty", "instruction_following", "truthfulness")


def _safe_rating(annotations: Dict[str, Any], criterion: str) -> Optional[int]:
    """Возвращает int(annotations[criterion]["Rating"]) или None при ValueError, KeyError, TypeError."""
    try:
        return int(annotations[criterion]["Rating"])
    except (ValueError, KeyError, TypeError):
        return None


def extract_pair_soft_openbmb(
    comp_best: Dict[str, Any],
    comp_worst: Dict[str, Any],
    prompt: str,
    alpha: float = 1.0,
) -> Optional[Dict[str, Any]]:
    """
    Два completion из openbmb/UltraFeedback → {prompt, resp1, resp2, p, p_bayes}.
    resp1=comp_best (лучший), resp2=comp_worst (худший). p = k/n — уверенность в best.
    k = голоса за best по критериям с валидным Rating; N/A пропускаются. Если n == 0 — None.
    """
    n = 0
    k = 0.0
    for c in CRITERIA:
        ann1 = comp_best.get("annotations") or {}
        ann2 = comp_worst.get("annotations") or {}
        r1 = _safe_rating(ann1, c)
        r2 = _safe_rating(ann2, c)
        if r1 is None or r2 is None:
            continue
        n += 1
        if r1 > r2:
            k += 1.0
        elif r1 == r2:
            k += 0.5
    if n == 0:
        return None
    resp1 = comp_best["response"]
    resp2 = comp_worst["response"]
    p = k / n
    p_bayes = (alpha + k) / (2.0 * alpha + n)
    return {
        "prompt": prompt,
        "resp1": resp1,
        "resp2": resp2,
        "p": p,
        "p_bayes": p_bayes,
    }


def build_openbmb_soft_datasets(alpha: float = 1.0):
    """
    Train: openbmb/UltraFeedback (soft пары best vs worst по overall_score).
    Val: HuggingFaceH4/ultrafeedback_binarized test_prefs (hard {prompt, chosen, rejected}).
    Возвращает: (train_soft_ds, val_hard_ds, len(train_soft_processed)).
    """
    ds_bin = load_dataset("HuggingFaceH4/ultrafeedback_binarized")
    val_raw = ds_bin["test_prefs"]

    def norm_prompt(p):
        return p if isinstance(p, str) else p.strip()

    val_processed = []
    val_prompts = set()
    for ex in val_raw:
        prompt = norm_prompt(ex["prompt"])
        chosen = ultrafeedback_message_to_response(ex["chosen"])
        rejected = ultrafeedback_message_to_response(ex["rejected"])
        if chosen and rejected:
            val_processed.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
            val_prompts.add(prompt)
    assert len(val_prompts) >= 1800, f"Val prompts match too low: {len(val_prompts)}"

    ds_open = load_dataset("openbmb/UltraFeedback", split="train")
    train_soft_processed = []
    for example in ds_open:
        instruction = norm_prompt(example["instruction"])
        if instruction in val_prompts:
            continue
        completions = example.get("completions") or []
        if len(completions) < 2:
            continue
        scores = [c.get("overall_score") for c in completions]
        if None in scores:
            continue
        max_score = max(scores)
        min_score = min(scores)
        if max_score <= min_score:
            continue
        idx_best = next(i for i, s in enumerate(scores) if s == max_score)
        idx_worst = next(i for i, s in enumerate(scores) if s == min_score)
        comp_best = completions[idx_best]
        comp_worst = completions[idx_worst]
        out = extract_pair_soft_openbmb(comp_best, comp_worst, instruction, alpha=alpha)
        if out is not None:
            train_soft_processed.append(out)
    return (
        Dataset.from_list(train_soft_processed),
        Dataset.from_list(val_processed),
        len(train_soft_processed),
    )
