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


# ---------- HelpSteer3: soft DPO (все пары, p / p_bayes) ----------

def extract_pair_soft(example: Dict[str, Any], alpha: float = 1.0) -> Dict[str, Any]:
    """
    Один сэмпл HelpSteer3 → {prompt, resp1, resp2, p, p_bayes}.
    p = k/n, p_bayes = (α + k) / (2α + n); n — число аннотаторов, k — голоса за response2.
    """
    prompt = context_to_prompt(example["context"])
    a = example["response1"]
    b = example["response2"]
    pref = example["overall_preference"]

    n = max(1, len(example.get("individual_preference", [])))
    if pref < 0:
        k = 0.0
    elif pref > 0:
        k = float(n)
    else:
        k = n / 2.0

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

    hard_train_size = sum(1 for ex in train_raw if extract_pair_hard(ex) is not None)
    train_soft_processed = [extract_pair_soft(ex, alpha=alpha) for ex in train_raw]
    val_hard_ds = build_val_hard(val_raw)
    train_soft_ds = Dataset.from_list(train_soft_processed)

    return train_soft_ds, val_hard_ds, hard_train_size
