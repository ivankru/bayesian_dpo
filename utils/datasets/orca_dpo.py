# -*- coding: utf-8 -*-
"""
Intel/orca_dpo_pairs: classic binary DPO {prompt, chosen, rejected}.

Chosen = GPT-4 / GPT-3.5 (OpenOrca); rejected = Llama-2-13B-chat.
Hub has a single train split (~12.8k); val is 10% of train (seed=42).
"""
from typing import Any, Dict, List, Optional, Tuple, Union

from datasets import Dataset, load_dataset

from .common import ultrafeedback_message_to_response

RepoId = "Intel/orca_dpo_pairs"


def _orca_response_text(resp: Union[str, List[Dict[str, Any]], Dict[str, Any], None]) -> str:
    if resp is None:
        return ""
    if isinstance(resp, str):
        return resp.strip()
    if isinstance(resp, list):
        return ultrafeedback_message_to_response(resp)
    if isinstance(resp, dict):
        return str(resp.get("content", resp.get("text", ""))).strip()
    return str(resp).strip()


def extract_pair_orca_hard(example: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """One Orca DPO row → {prompt, chosen, rejected}, or None if empty."""
    prompt = example.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        question = str(example.get("question") or "").strip()
        system = str(example.get("system") or "").strip()
        prompt = f"{system}\n\n{question}".strip() if system else question
    else:
        prompt = prompt.strip()

    chosen = _orca_response_text(example.get("chosen"))
    rejected = _orca_response_text(example.get("rejected"))
    if not prompt or not chosen or not rejected:
        return None
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


def build_dpo_datasets_orca_dpo(
    split_train: str = "train",
    val_fraction: float = 0.10,
    seed: int = 42,
) -> Tuple[Dataset, Dataset]:
    """
    Hard DPO from Intel/orca_dpo_pairs.

    Returns train_ds, val_ds with fields prompt, chosen, rejected.
    """
    raw = load_dataset(RepoId, split=split_train)
    processed = []
    for ex in raw:
        out = extract_pair_orca_hard(ex)
        if out is not None:
            processed.append(out)
    full = Dataset.from_list(processed)
    split = full.train_test_split(test_size=val_fraction, seed=seed)
    return split["train"], split["test"]


def build_orca_dpo_soft_steer_datasets(
    alpha: float = 1.0,
    val_fraction: float = 0.10,
    seed: int = 42,
) -> Tuple[Dataset, Dataset, int]:
    """
    Soft-DPO train + hard val for soft_dpo_steer (binary p=1, same split as hard).
    """
    train_hard, val_hard = build_dpo_datasets_orca_dpo(
        val_fraction=val_fraction,
        seed=seed,
    )
    p_bayes = (alpha + 1.0) / (2.0 * alpha + 1.0)
    train_soft = Dataset.from_list(
        [
            {
                "prompt": row["prompt"],
                "resp1": row["chosen"],
                "resp2": row["rejected"],
                "p": 1.0,
                "p_bayes": p_bayes,
            }
            for row in train_hard
        ]
    )
    return train_soft, val_hard, len(train_soft)
