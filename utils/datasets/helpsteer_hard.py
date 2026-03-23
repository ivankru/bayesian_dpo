# -*- coding: utf-8 -*-
"""HelpSteer3 (preference): hard DPO — пары с явным предпочтением (pref != 0)."""
from typing import Dict, Any, Optional, List

from datasets import load_dataset, Dataset


def context_to_prompt(context: List[Dict[str, Any]]) -> str:
    """Собирает строку промпта из списка сообщений {role, content} (HelpSteer3)."""
    parts = []
    for turn in context:
        role = turn["role"]
        text = turn["content"]
        parts.append(f"{role.upper()}: {text}")
    return "\n".join(parts)


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
