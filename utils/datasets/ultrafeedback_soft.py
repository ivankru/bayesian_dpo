# -*- coding: utf-8 -*-
"""UltraFeedback Binarized: soft DPO (Bradley-Terry)."""
import random
from typing import Dict, Any, Optional

from datasets import Dataset, load_dataset

from .common import sigmoid, ultrafeedback_message_to_response


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
        # Ожидаем p в {0.0, 1.0}; на всякий случай поддержим и "почти" бинарные значения
        if rng.random() < noise_prob:
            if p == 0.0:
                example[col_name] = 1.0
            elif p == 1.0:
                example[col_name] = 0.0
            else:
                example[col_name] = 1.0 - p
        p_cur = float(example[col_name])
        example["p_bayes"] = (alpha + p_cur) / (2.0 * alpha + 1.0)
        return example

    return ds.map(_flip, with_indices=True)


def extract_pair_soft_ultrafeedback(
    example: Dict[str, Any], alpha: float = 1.0
) -> Optional[Dict[str, Any]]:
    """
    Один сэмпл UltraFeedback Binarized → {prompt, resp1, resp2, p, p_bayes}.
    resp1=chosen (лучший), resp2=rejected (худший).
    p = sigmoid(score_chosen - score_rejected), p_bayes = (alpha + p) / (2*alpha + 1).
    """
    prompt = example["prompt"] if isinstance(example["prompt"], str) else example["prompt"].strip()
    chosen = ultrafeedback_message_to_response(example["chosen"])
    rejected = ultrafeedback_message_to_response(example["rejected"])
    if not chosen or not rejected:
        return None
    score_chosen = float(example.get("score_chosen", 0.0))
    score_rejected = float(example.get("score_rejected", 0.0))
    p = sigmoid(score_chosen - score_rejected)
    p_bayes = (alpha + p) / (2.0 * alpha + 1.0)
    return {
        "prompt": prompt,
        "resp1": chosen,
        "resp2": rejected,
        "p": p,
        "p_bayes": p_bayes,
    }


def build_ultrafeedback_soft_datasets(
    alpha: float,
    label_noise_prob: float = 0.0,
    seed: int = 42,
):
    """
    UltraFeedback Binarized для soft-DPO: resp1=chosen, resp2=rejected, p из Bradley-Terry.
    Возвращает: train_soft_ds, val_hard_ds, hard_train_size.
    """
    ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized")
    train_raw = ds["train_prefs"]
    val_raw = ds["test_prefs"]

    train_soft_processed = []
    for ex in train_raw:
        out = extract_pair_soft_ultrafeedback(ex, alpha=alpha)
        if out is not None:
            train_soft_processed.append(out)
    train_soft_ds = Dataset.from_list(train_soft_processed)
    train_soft_ds = flip_binary_labels(
        train_soft_ds,
        noise_prob=label_noise_prob,
        seed=seed,
        alpha=alpha,
        col_name="p",
    )

    val_processed = []
    for ex in val_raw:
        prompt = ex["prompt"] if isinstance(ex["prompt"], str) else ex["prompt"].strip()
        chosen = ultrafeedback_message_to_response(ex["chosen"])
        rejected = ultrafeedback_message_to_response(ex["rejected"])
        if chosen and rejected:
            val_processed.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
    val_hard_ds = Dataset.from_list(val_processed)
    hard_train_size = len(train_soft_processed)
    return train_soft_ds, val_hard_ds, hard_train_size
