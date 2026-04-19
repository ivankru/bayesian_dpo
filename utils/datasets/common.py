# -*- coding: utf-8 -*-
"""Общие хелперы для работы с датасетами DPO."""
import math
from typing import Any, Callable, Dict, List

import torch
from datasets import Dataset
from torch.utils.data import DataLoader

from utils.loss import _logps


def sigmoid(x: float) -> float:
    """Стабильный sigmoid: 1 / (1 + exp(-x))."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def ultrafeedback_message_to_response(messages: List[Dict[str, str]]) -> str:
    """Из списка сообщений {role, content} достаёт конкатенацию ответов assistant."""
    parts = [m["content"] for m in messages if m.get("role") == "assistant"]
    return "\n".join(parts).strip() if parts else ""


def _precompute_p_pred_column(
    train_ds: Dataset,
    tokenizer,
    policy_model,
    ref_model,
    device: str,
    beta: float,
    use_chat_template: bool,
    batch_size: int,
    collate_fn: Callable[..., Any],
    column_name: str,
) -> Dataset:
    """
    Один проход по train_ds: p = sigmoid(beta * diff), diff как в soft_dpo_loss;
    записывает столбец column_name. Политика и ref — в torch.no_grad().
    """
    loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    p_preds: List[float] = []
    policy_model.eval()
    with torch.no_grad():
        for batch in loader:
            prompts = batch["prompt"]
            resp1 = batch["resp1"]
            resp2 = batch["resp2"]
            logp_1 = _logps(
                policy_model, tokenizer, prompts, resp1, device, use_chat_template
            )
            logp_2 = _logps(
                policy_model, tokenizer, prompts, resp2, device, use_chat_template
            )
            logp_1_ref = _logps(
                ref_model, tokenizer, prompts, resp1, device, use_chat_template
            )
            logp_2_ref = _logps(
                ref_model, tokenizer, prompts, resp2, device, use_chat_template
            )
            delta_theta = logp_1 - logp_2
            delta_ref = logp_1_ref - logp_2_ref
            diff = delta_theta - delta_ref
            logit = beta * diff
            p_pred_batch = torch.sigmoid(logit)
            p_preds.extend(p_pred_batch.detach().cpu().float().tolist())

    assert len(p_preds) == len(train_ds), (
        f"{column_name} length {len(p_preds)} != len(train_ds) {len(train_ds)}"
    )

    if column_name in train_ds.column_names:
        train_ds = train_ds.remove_columns([column_name])
    return train_ds.add_column(column_name, p_preds)


def precompute_p_pred_cached(
    train_ds: Dataset,
    tokenizer,
    policy_model,
    ref_model,
    device: str,
    beta: float,
    use_chat_template: bool,
    batch_size: int,
    collate_fn: Callable[..., Any],
) -> Dataset:
    """
    Один проход по train_ds (порядок строк датасета): считаете
    p_pred = sigmoid(beta * ((logp1-logp2) - (logp1_ref-logp2_ref))) как в soft_dpo_loss,
    записывает столбец p_pred_cached. Политика и ref — в torch.no_grad().
    """
    return _precompute_p_pred_column(
        train_ds,
        tokenizer,
        policy_model,
        ref_model,
        device,
        beta,
        use_chat_template,
        batch_size,
        collate_fn,
        "p_pred_cached",
    )


def precompute_p_pred_teacher(
    train_ds: Dataset,
    tokenizer,
    policy_model,
    ref_model,
    device: str,
    beta: float,
    use_chat_template: bool,
    batch_size: int,
    collate_fn: Callable[..., Any],
) -> Dataset:
    """
    Заморозка «учителя» после warmup-эпох: те же σ(beta*diff), что и для p_pred_cached,
    но в отдельный столбец p_pred_teacher (не перезаписывается в фазе смешивания).
    """
    return _precompute_p_pred_column(
        train_ds,
        tokenizer,
        policy_model,
        ref_model,
        device,
        beta,
        use_chat_template,
        batch_size,
        collate_fn,
        "p_pred_teacher",
    )
