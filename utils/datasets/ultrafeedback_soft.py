# -*- coding: utf-8 -*-
"""
UltraFeedback (HuggingFaceH4/ultrafeedback_binarized) для soft-DPO.

- ultrafeedback_binarized (CLI): жёсткое предпочтение chosen > rejected, p ∈ {0, 1} после шума.
- ultrafeedback_soft (CLI): мягкие метки p = sigmoid(score_chosen - score_rejected).

Семантика CLI: ранее в soft_dpo_steer --dataset ultrafeedback_binarized соответствовала
поведению нынешнего ultrafeedback_soft; после разделения бинарный режим вынесен в
ultrafeedback_binarized, прежний score-soft — в ultrafeedback_soft.
"""
import random
from typing import Any, Dict, List, Optional, Tuple

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
    С заданной вероятностью noise_prob инвертирует целевую вероятность в col_name:
    для p ∈ {0, 1} — переворот 0↔1; иначе p → 1−p. После этого пересчитывает p_bayes
    как (alpha + p) / (2*alpha + 1).
    """
    if noise_prob <= 0.0:
        return ds

    rng = random.Random(seed)

    def _flip(example, idx):
        p = example[col_name]
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


def extract_pair_ultrafeedback_binarized(
    example: Dict[str, Any], alpha: float = 1.0
) -> Optional[Dict[str, Any]]:
    """
    Жёсткое предпочтение (как hard-DPO): resp1=chosen, resp2=rejected, p=1.0.
    p_bayes = (alpha + p) / (2*alpha + 1).
    """
    prompt = example["prompt"] if isinstance(example["prompt"], str) else example["prompt"].strip()
    chosen = ultrafeedback_message_to_response(example["chosen"])
    rejected = ultrafeedback_message_to_response(example["rejected"])
    if not chosen or not rejected:
        return None
    p = 1.0
    p_bayes = (alpha + p) / (2.0 * alpha + 1.0)
    return {
        "prompt": prompt,
        "resp1": chosen,
        "resp2": rejected,
        "p": p,
        "p_bayes": p_bayes,
    }


def extract_pair_ultrafeedback_score_soft(
    example: Dict[str, Any], alpha: float = 1.0
) -> Optional[Dict[str, Any]]:
    """
    Мягкие метки по скорам: resp1=chosen, resp2=rejected,
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


def _load_ultrafeedback_binarized_prefs() -> Tuple[Dataset, Dataset]:
    ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized")
    return ds["train_prefs"], ds["test_prefs"]


def _build_val_hard_ultrafeedback_prefs(val_raw: Dataset) -> Dataset:
    val_processed: List[Dict[str, Any]] = []
    for ex in val_raw:
        prompt = ex["prompt"] if isinstance(ex["prompt"], str) else ex["prompt"].strip()
        chosen = ultrafeedback_message_to_response(ex["chosen"])
        rejected = ultrafeedback_message_to_response(ex["rejected"])
        if chosen and rejected:
            val_processed.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
    return Dataset.from_list(val_processed)


def build_ultrafeedback_binarized_soft_datasets(
    alpha: float,
    label_noise_prob: float = 0.0,
    seed: int = 42,
):
    """
    UltraFeedback Binarized: бинарные целевые метки (p=1 до шума), валидация hard.
    Возвращает: train_soft_ds, val_hard_ds, hard_train_size.
    """
    train_raw, val_raw = _load_ultrafeedback_binarized_prefs()

    train_soft_processed: List[Dict[str, Any]] = []
    for ex in train_raw:
        out = extract_pair_ultrafeedback_binarized(ex, alpha=alpha)
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

    val_hard_ds = _build_val_hard_ultrafeedback_prefs(val_raw)
    hard_train_size = len(train_soft_processed)
    return train_soft_ds, val_hard_ds, hard_train_size


def build_ultrafeedback_score_soft_datasets(
    alpha: float,
    label_noise_prob: float = 0.0,
    seed: int = 42,
):
    """
    UltraFeedback Binarized с мягкими метками из score_chosen/score_rejected (sigmoid).
    При --label-noise-prob > 0: с той же вероятностью p → 1−p и пересчёт p_bayes.
    Возвращает: train_soft_ds, val_hard_ds, hard_train_size.
    """
    train_raw, val_raw = _load_ultrafeedback_binarized_prefs()

    train_soft_processed: List[Dict[str, Any]] = []
    for ex in train_raw:
        out = extract_pair_ultrafeedback_score_soft(ex, alpha=alpha)
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

    val_hard_ds = _build_val_hard_ultrafeedback_prefs(val_raw)
    hard_train_size = len(train_soft_processed)
    return train_soft_ds, val_hard_ds, hard_train_size
