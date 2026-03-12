# -*- coding: utf-8 -*-
"""
Загрузка и преобразование датасетов для DPO / soft-DPO.
HelpSteer3 (preference), UltraFeedback Binarized.
"""
import random
from typing import List, Dict, Any, Optional

from datasets import load_dataset, Dataset


def context_to_prompt(context: List[Dict[str, Any]]) -> str:
    """Собирает строку промпта из списка сообщений {role, content} (HelpSteer3)."""
    parts = []
    for turn in context:
        role = turn["role"]
        text = turn["content"]
        parts.append(f"{role.upper()}: {text}")
    return "\n".join(parts)


# ---------- HelpSteer3: hard DPO (pref != 0) ----------

def extract_pair_hard(example: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """
    Один сэмпл HelpSteer3 → {prompt, chosen, rejected} или None при pref == 0.
    """
    prompt = context_to_prompt(example["context"])
    a = example["response1"]
    b = example["response2"]
    pref = example["overall_preference"]

    if pref == 0:
        return None

    if pref < 0:
        chosen, rejected = a, b
    else:
        chosen, rejected = b, a

    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


def build_val_hard(val_raw) -> Dataset:
    """
    Val по HelpSteer3: только пары с явным предпочтением (pref != 0).
    Используется в hard_dpo_steer и в soft_dpo_steer для одинакового val.
    """
    val_processed = []
    for ex in val_raw:
        out = extract_pair_hard(ex)
        if out is not None:
            val_processed.append(out)
    return Dataset.from_list(val_processed)


def build_dpo_datasets():
    """HelpSteer3 (preference): train и val только с явным предпочтением (pref != 0)."""
    ds = load_dataset("nvidia/HelpSteer3", name="preference", streaming=False)
    train_raw = ds["train"]
    val_raw = ds["validation"]

    train_processed = []
    for ex in train_raw:
        out = extract_pair_hard(ex)
        if out is not None:
            train_processed.append(out)

    train_ds = Dataset.from_list(train_processed)
    val_ds = build_val_hard(val_raw)
    return train_ds, val_ds


# ---------- UltraFeedback Binarized ----------

def _ultrafeedback_message_to_response(messages: List[Dict[str, str]]) -> str:
    """Из списка сообщений {role, content} достаёт конкатенацию ответов assistant."""
    parts = [m["content"] for m in messages if m.get("role") == "assistant"]
    return "\n".join(parts).strip() if parts else ""


def build_dpo_datasets_ultrafeedback():
    """
    UltraFeedback Binarized: train_prefs, test_prefs → {prompt, chosen, rejected}.
    """
    ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized")
    train_raw = ds["train_prefs"]
    val_raw = ds["test_prefs"]

    def convert(ex):
        prompt = ex["prompt"] if isinstance(ex["prompt"], str) else ex["prompt"].strip()
        chosen = _ultrafeedback_message_to_response(ex["chosen"])
        rejected = _ultrafeedback_message_to_response(ex["rejected"])
        if not chosen or not rejected:
            return None
        return {"prompt": prompt, "chosen": chosen, "rejected": rejected}

    train_processed = [out for ex in train_raw if (out := convert(ex)) is not None]
    val_processed = [out for ex in val_raw if (out := convert(ex)) is not None]

    return Dataset.from_list(train_processed), Dataset.from_list(val_processed)


# ---------- UltraFeedback Binarized: soft DPO (голоса 0/1) ----------

def extract_pair_soft_ultrafeedback(
    example: Dict[str, Any], alpha: float = 1.0, swap: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Один сэмпл UltraFeedback Binarized → {prompt, resp1, resp2, p, p_bayes}.
    Как в HelpSteer3 soft: порядок resp1/resp2 может быть (chosen, rejected) или (rejected, chosen),
    «кто лучше» кодируется в p (0 = лучше resp1, 1 = лучше resp2).
    swap=True → resp1=rejected, resp2=chosen, p=1; swap=False → resp1=chosen, resp2=rejected, p=0.
    """
    prompt = example["prompt"] if isinstance(example["prompt"], str) else example["prompt"].strip()
    chosen = _ultrafeedback_message_to_response(example["chosen"])
    rejected = _ultrafeedback_message_to_response(example["rejected"])
    if not chosen or not rejected:
        return None
    n = 1
    if swap:
        resp1, resp2 = rejected, chosen
        k = 1.0  # голоса за resp2 (chosen)
    else:
        resp1, resp2 = chosen, rejected
        k = 0.0  # голоса за resp2 (rejected)
    p = k / n
    p_bayes = (alpha + k) / (2.0 * alpha + n)
    return {
        "prompt": prompt,
        "resp1": resp1,
        "resp2": resp2,
        "p": p,
        "p_bayes": p_bayes,
    }


def build_ultrafeedback_soft_datasets(alpha: float = 1.0, seed: int = 42):
    """
    UltraFeedback Binarized для soft-DPO: как HelpSteer3 soft — порядок resp1/resp2 случайный
    (chosen то на первом, то на втором месте), «кто лучше» кодируется в p (0 или 1).
    Возвращает: train_soft_ds, val_hard_ds, hard_train_size.
    """
    ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized")
    train_raw = ds["train_prefs"]
    val_raw = ds["test_prefs"]

    rng = random.Random(seed)
    train_soft_processed = []
    for ex in train_raw:
        swap = rng.random() < 0.5
        out = extract_pair_soft_ultrafeedback(ex, alpha=alpha, swap=swap)
        if out is not None:
            train_soft_processed.append(out)
    train_soft_ds = Dataset.from_list(train_soft_processed)

    val_processed = []
    for ex in val_raw:
        prompt = ex["prompt"] if isinstance(ex["prompt"], str) else ex["prompt"].strip()
        chosen = _ultrafeedback_message_to_response(ex["chosen"])
        rejected = _ultrafeedback_message_to_response(ex["rejected"])
        if chosen and rejected:
            val_processed.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
    val_hard_ds = Dataset.from_list(val_processed)
    hard_train_size = len(train_soft_processed)
    return train_soft_ds, val_hard_ds, hard_train_size


# ---------- openbmb/UltraFeedback: soft DPO (4 критерия → p, p_bayes) ----------

CRITERIA = ("helpfulness", "honesty", "instruction_following", "truthfulness")


def _safe_rating(annotations: Dict[str, Any], criterion: str) -> Optional[int]:
    """Возвращает int(annotations[criterion]["Rating"]) или None при ValueError, KeyError, TypeError."""
    try:
        return int(annotations[criterion]["Rating"])
    except (ValueError, KeyError, TypeError):
        return None


def extract_pair_soft_openbmb(
    comp1: Dict[str, Any],
    comp2: Dict[str, Any],
    prompt: str,
    alpha: float = 1.0,
    swap: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Два completion из openbmb/UltraFeedback → {prompt, resp1, resp2, p, p_bayes}.
    По критериям с валидным Rating: vote = 1 если comp2 лучше, 0 если comp1 лучше, 0.5 если равны.
    Критерии с N/A или невалидным Rating пропускаются. Если n == 0 — None.
    swap=True: меняем местами comp1/comp2 и k = n - k.
    """
    n = 0
    k = 0.0
    for c in CRITERIA:
        ann1 = comp1.get("annotations") or {}
        ann2 = comp2.get("annotations") or {}
        r1 = _safe_rating(ann1, c)
        r2 = _safe_rating(ann2, c)
        if r1 is None or r2 is None:
            continue
        n += 1
        if r2 > r1:
            k += 1.0
        elif r1 == r2:
            k += 0.5
    if n == 0:
        return None
    if swap:
        resp1 = comp2["response"]
        resp2 = comp1["response"]
        k = n - k
    else:
        resp1 = comp1["response"]
        resp2 = comp2["response"]
    p = k / n
    p_bayes = (alpha + k) / (2.0 * alpha + n)
    return {
        "prompt": prompt,
        "resp1": resp1,
        "resp2": resp2,
        "p": p,
        "p_bayes": p_bayes,
    }


def build_openbmb_soft_datasets(alpha: float = 1.0, seed: int = 42):
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
        chosen = _ultrafeedback_message_to_response(ex["chosen"])
        rejected = _ultrafeedback_message_to_response(ex["rejected"])
        if chosen and rejected:
            val_processed.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
            val_prompts.add(prompt)
    assert len(val_prompts) >= 1800, f"Val prompts match too low: {len(val_prompts)}"

    ds_open = load_dataset("openbmb/UltraFeedback", split="train")
    rng = random.Random(seed)
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
        swap = rng.random() < 0.5
        out = extract_pair_soft_openbmb(
            comp_best, comp_worst, instruction, alpha=alpha, swap=swap
        )
        if out is not None:
            train_soft_processed.append(out)
    return (
        Dataset.from_list(train_soft_processed),
        Dataset.from_list(val_processed),
        len(train_soft_processed),
    )


# ---------- HelpSteer3: soft DPO (все пары, p / p_bayes) ----------

def extract_pair_soft(example: Dict[str, Any], alpha: float = 1.0) -> Dict[str, Any]:
    """
    Один сэмпл HelpSteer3 → {prompt, resp1, resp2, p, p_bayes}.
    p = k/n, p_bayes = (α + k) / (2α + n); n — число аннотаторов, k — голоса за response2.
    """
    prompt = context_to_prompt(example["context"])
    a = example["response1"]
    b = example["response2"]
    
    #pref = example["overall_preference"]
        # n = max(1, len(example.get("individual_preference", [])))
    # if pref < 0:
    #     k = 0.0
    # elif pref > 0:
    #     k = float(n)
    # else:
    #     k = n / 2.0

    votes = example.get("individual_preference", [])
    n = max(1, len(votes))
    # Считаем реальные голоса за response2 (score > 0 в HelpSteer3 = response2 лучше)
    k = sum(1.0 for v in votes if v.get("score", 0) > 0)

    p = k / n
    p_bayes = (alpha + k) / (2.0 * alpha + n)

    return {
        "prompt": prompt,
        "resp1": a,
        "resp2": b,
        "p": p,
        "p_bayes": p_bayes,
    }


def build_helpsteer3_soft_datasets(alpha: float = 1.0):
    """
    HelpSteer3 для soft-DPO: все пары (pref -1, 0, +1) с p/p_bayes.
    Возвращает: train_soft_ds, val_hard_ds, hard_train_size (для выравнивания lr с hard).
    """
    ds = load_dataset("nvidia/HelpSteer3", name="preference", streaming=False)
    train_raw = ds["train"]
    val_raw = ds["validation"]

    train_soft_processed = []
    hard_train_size = 0
    for ex in train_raw:
        if extract_pair_hard(ex) is not None:
            hard_train_size += 1
        train_soft_processed.append(extract_pair_soft(ex, alpha=alpha))
    val_hard_ds = build_val_hard(val_raw)
    train_soft_ds = Dataset.from_list(train_soft_processed)

    return train_soft_ds, val_hard_ds, hard_train_size
