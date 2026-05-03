# -*- coding: utf-8 -*-
"""
PKU-Alignment/processed-hh-rlhf.

Hard DPO: {prompt, chosen, rejected}.
Soft DPO: {prompt, resp1, resp2, p, p_bayes} (resp1=chosen, resp2=rejected;
p = 1.0, p_bayes = (alpha + 1) / (2*alpha + 1)).

For Qwen2.5-Instruct by default (config.base_config.USE_CHAT_TEMPLATE) the prompt string
is passed as a single user message in apply_chat_template; log p is computed only on response tokens.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

from datasets import load_dataset, Dataset

RepoId = "PKU-Alignment/processed-hh-rlhf"


def _load_pku_hh_rlhf_raw_splits(
    split_train: str = "train",
    split_val: str = "validation",
) -> Tuple[Any, Optional[Any]]:
    raw_train = load_dataset(RepoId, split=split_train)
    raw_val = None
    for split_name in (split_val, "test"):
        if split_name == split_train:
            continue
        try:
            raw_val = load_dataset(RepoId, split=split_name)
            break
        except Exception:
            continue
    return raw_train, raw_val


def _pku_context_to_prompt(context: List[Dict[str, Any]]) -> str:
    """PKU dialog: list of turns {role, text} → one string (like HelpSteer, but key is text)."""
    parts = []
    for turn in context or []:
        role = turn.get("role", "")
        text = turn.get("text", turn.get("content", ""))
        parts.append(f"{str(role).upper()}: {text}")
    return "\n".join(parts)


def _pku_response_text(resp: Union[str, Dict[str, Any]]) -> str:
    """PKU response: plain string or {role, text} / {role, content}."""
    if isinstance(resp, str):
        return resp
    if isinstance(resp, dict):
        return resp.get("text", resp.get("content", ""))
    return str(resp)


def extract_pair_hh_hard(example: Dict[str, Any]) -> Dict[str, str]:
    """One PKU processed HH-RLHF sample → {prompt, chosen, rejected} for hard DPO."""
    return {
        "prompt": _pku_context_to_prompt(example.get("context", [])),
        "chosen": _pku_response_text(example["chosen"]),
        "rejected": _pku_response_text(example["rejected"]),
    }


def build_dpo_datasets_hh_rlhf(
    split_train: str = "train",
    split_val: str = "validation",
) -> Tuple[Dataset, Dataset]:
    """
    Hard DPO from PKU-Alignment/processed-hh-rlhf: train_ds, val_ds with fields
    prompt, chosen, rejected. Splits match build_hh_rlhf_soft_datasets.
    """
    raw_train, raw_val = _load_pku_hh_rlhf_raw_splits(split_train, split_val)
    train_ds = Dataset.from_list([extract_pair_hh_hard(ex) for ex in raw_train])
    if raw_val is not None:
        val_ds = Dataset.from_list([extract_pair_hh_hard(ex) for ex in raw_val])
    else:
        split = train_ds.train_test_split(test_size=0.05, seed=42)
        train_ds, val_ds = split["train"], split["test"]
    return train_ds, val_ds


def extract_pair_hh_soft(
    example: Dict[str, Any],
    alpha: float = 1.0,
) -> Dict[str, Any]:
    """
    One processed HH-RLHF sample → {prompt, resp1, resp2, p, p_bayes}.

    PKU-Alignment/processed-hh-rlhf: fields context (dialog), chosen, rejected
    (the latter two are usually dicts with key text).
    """
    prompt = _pku_context_to_prompt(example.get("context", []))
    chosen = _pku_response_text(example["chosen"])
    rejected = _pku_response_text(example["rejected"])

    # Binary label: chosen always preferred over rejected
    p = 1.0
    # Bayesian smoothing, equivalent to one vote for chosen at n=1
    p_bayes = (alpha + 1.0) / (2.0 * alpha + 1.0)

    return {
        "prompt": prompt,
        "resp1": chosen,
        "resp2": rejected,
        "p": p,
        "p_bayes": p_bayes,
    }


def build_hh_rlhf_soft_datasets(
    split_train: str = "train",
    split_val: str = "validation",
    alpha: float = 1.0,
) -> Tuple[Dataset, Dataset]:
    """
    Build soft-DPO datasets from PKU-Alignment/processed-hh-rlhf.

    split_train: train split name (usually "train").
    split_val  : validation split. PKU has "test"; "validation" may be missing —
                 on load failure try "test", otherwise use 5% of train.

    Returns:
      train_soft_ds, val_soft_ds

    Both have format:
      - "prompt"
      - "resp1" (chosen)
      - "resp2" (rejected)
      - "p"
      - "p_bayes"
    """
    raw_train, raw_val = _load_pku_hh_rlhf_raw_splits(split_train, split_val)

    train_soft_processed = [
        extract_pair_hh_soft(ex, alpha=alpha) for ex in raw_train
    ]
    train_soft_ds = Dataset.from_list(train_soft_processed)

    if raw_val is not None:
        val_soft_processed = [
            extract_pair_hh_soft(ex, alpha=alpha) for ex in raw_val
        ]
        val_soft_ds = Dataset.from_list(val_soft_processed)
    else:
        # no dedicated val split: split from train
        split = train_soft_ds.train_test_split(test_size=0.05, seed=42)
        train_soft_ds, val_soft_ds = split["train"], split["test"]

    return train_soft_ds, val_soft_ds


def build_hh_rlhf_soft_steer_datasets(alpha: float = 1.0) -> Tuple[Dataset, Dataset, int]:
    """
    Soft-DPO train + hard val for soft_dpo_steer (same pattern as HelpSteer / UltraFeedback).

    Splits match build_dpo_datasets_hh_rlhf: if a separate val exists, use it;
    else train_test_split(test_size=0.05, seed=42) on full train.

    Returns: train_soft_ds, val_hard_ds, hard_train_size (len(train_soft)).
    """
    raw_train, raw_val = _load_pku_hh_rlhf_raw_splits()

    if raw_val is not None:
        train_soft_ds = Dataset.from_list(
            [extract_pair_hh_soft(ex, alpha=alpha) for ex in raw_train]
        )
        val_hard_ds = Dataset.from_list([extract_pair_hh_hard(ex) for ex in raw_val])
        hard_train_size = len(train_soft_ds)
        return train_soft_ds, val_hard_ds, hard_train_size

    train_soft_list = [extract_pair_hh_soft(ex, alpha=alpha) for ex in raw_train]
    full_soft = Dataset.from_list(train_soft_list)
    split = full_soft.train_test_split(test_size=0.05, seed=42)
    train_soft_ds = split["train"]
    val_hard_ds = Dataset.from_list(
        [
            {"prompt": r["prompt"], "chosen": r["resp1"], "rejected": r["resp2"]}
            for r in split["test"]
        ]
    )
    hard_train_size = len(train_soft_ds)
    return train_soft_ds, val_hard_ds, hard_train_size
