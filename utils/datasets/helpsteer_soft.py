# -*- coding: utf-8 -*-
"""HelpSteer3 (preference): soft DPO — все пары с p/p_bayes."""
import random
from typing import Dict, Any

from datasets import Dataset, load_dataset

from .helpsteer_hard import context_to_prompt, extract_pair_hard, build_val_hard


def flip_binary_labels(
    ds: Dataset,
    noise_prob: float,
    seed: int,
    alpha: float,
    col_name: str = "p",
) -> Dataset:
    """
    С заданной вероятностью noise_prob переворачивает бинарные метки в колонке col_name: 0 -> 1, 1 -> 0.
    Шум фиксирован по seed для воспроизводимости.
    """
    if noise_prob <= 0.0:
        return ds

    rng = random.Random(seed)

    def _flip(example, idx):
        p = example[col_name]
        k = float(example["k"])
        n = float(example["n"])
        # Ожидаем p в {0.0, 1.0}; на всякий случай поддержим и "почти" бинарные значения
        if rng.random() < noise_prob:
            if p == 0.0:
                example[col_name] = 1.0
            elif p == 1.0:
                example[col_name] = 0.0
            else:
                example[col_name] = 1.0 - p
            # Для HelpSteer3 апостериор строится из k/n, поэтому инвертируем k.
            k = n - k
            example["k"] = k
        example["p_bayes"] = (alpha + k) / (2.0 * alpha + n)
        return example

    return ds.map(_flip, with_indices=True)


def extract_pair_soft(example: Dict[str, Any], alpha: float = 1.0) -> Dict[str, Any]:
    """
    Один сэмпл HelpSteer3 → {prompt, resp1, resp2, p, p_bayes}.
    resp1=chosen (лучший), resp2=rejected (худший). p = k/n — уверенность в chosen.
    k = голоса за chosen; n = число аннотаторов.
    """
    prompt = context_to_prompt(example["context"])
    a = example["response1"]
    b = example["response2"]
    pref = example["overall_preference"]
    votes = example.get("individual_preference", [])
    n = max(1, len(votes))

    if pref < 0:
        chosen, rejected = a, b
        k = sum(1.0 for v in votes if v.get("score", 0) < 0)  # голоса за response1
    elif pref > 0:
        chosen, rejected = b, a
        k = sum(1.0 for v in votes if v.get("score", 0) > 0)  # голоса за response2
    else:
        chosen, rejected = a, b
        k = sum(1.0 for v in votes if v.get("score", 0) < 0)  # условно за response1

    p = k / n
    p_bayes = (alpha + k) / (2.0 * alpha + n)
    return {
        "prompt": prompt,
        "resp1": chosen,
        "resp2": rejected,
        "p": p,
        "p_bayes": p_bayes,
        "k": k,
        "n": n,
    }


def build_helpsteer3_soft_datasets(
    alpha: float,
    label_noise_prob: float = 0.0,
    seed: int = 42,
):
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
        if ex["overall_preference"] != 0:
            train_soft_processed.append(extract_pair_soft(ex, alpha=alpha))
    val_hard_ds = build_val_hard(val_raw)
    train_soft_ds = Dataset.from_list(train_soft_processed)
    train_soft_ds = flip_binary_labels(
        train_soft_ds,
        noise_prob=label_noise_prob,
        seed=seed,
        alpha=alpha,
        col_name="p",
    )

    return train_soft_ds, val_hard_ds, hard_train_size
