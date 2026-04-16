# -*- coding: utf-8 -*-
"""
Метрики и утилиты для DPO: get_logps, eval_pairwise_accuracy, eval_pairwise_nll.
"""
from typing import List, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm


def _apply_chat_template_ids(tokenizer, user_content: str) -> List[int]:
    """Префикс диалога Qwen-Instruct до начала генерации ответа ассистента (включая assistant header)."""
    out = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=True,
        add_generation_prompt=True,
    )
    if isinstance(out, dict):
        out = out["input_ids"]
    return list(out)


def get_logps(
    model,
    tokenizer,
    prompts: List[str],
    responses: List[str],
    device: str,
    max_prompt_len: int = 768,
    max_full_len: int = 1024,
    use_chat_template: bool = False,
) -> torch.Tensor:
    """
    log p(response | prompt) = сумма логвероятностей токенов ответа.

    use_chat_template=False: как раньше — склейка plain text ``prompt + "\\n" + response`` и длина
    префикса по отдельной токенизации prompt (без chat template).

    use_chat_template=True (Qwen2.5-Instruct и аналоги): префикс через ``apply_chat_template`` одного
    user-турна (вся сериализованная реплика/диалог в строке prompt), затем токены ответа ассистента
    без повторного add_special_tokens, чтобы не ломать границу BPE. Лог-вероятности суммируются
    только по суффиксу ответа, не по префиксу чата.
    """
    if not use_chat_template:
        prompt_batch = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=max_prompt_len,
            return_tensors="pt",
        ).to(device)

        prompt_input_ids = prompt_batch["input_ids"]
        prompt_lengths = (prompt_input_ids != tokenizer.pad_token_id).sum(dim=1)

        full_texts = [p + "\n" + r for p, r in zip(prompts, responses)]
        full_batch = tokenizer(
            full_texts,
            padding=True,
            truncation=True,
            max_length=max_full_len,
            return_tensors="pt",
        ).to(device)

        outputs = model(**full_batch)
        logits = outputs.logits
        logprobs = F.log_softmax(logits, dim=-1)

        input_ids = full_batch["input_ids"]
        B, T = input_ids.shape

        logp_list = []
        for i in range(B):
            pl = prompt_lengths[i].item()
            start = pl

            if start >= T:
                logp_list.append(logprobs[i, 0, 0] * 0)
                continue

            lp = logprobs[i, start - 1 : T - 1, :]
            ids = input_ids[i, start:T]
            lp_tokens = lp.gather(-1, ids.unsqueeze(-1)).squeeze(-1)
            logp_list.append(lp_tokens.sum())

        return torch.stack(logp_list, dim=0)

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    batch_input_ids: List[List[int]] = []
    batch_prefix_lens: List[int] = []

    for p, r in zip(prompts, responses):
        p_ids = _apply_chat_template_ids(tokenizer, p)
        r_ids = tokenizer(r, add_special_tokens=False)["input_ids"]
        full = p_ids + r_ids
        if len(full) > max_full_len:
            full = full[:max_full_len]
        pl_orig = len(p_ids)
        pl = min(pl_orig, len(full))
        batch_input_ids.append(full)
        batch_prefix_lens.append(pl)

    max_t = max(len(x) for x in batch_input_ids)
    B = len(batch_input_ids)
    input_ids = torch.full((B, max_t), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((B, max_t), dtype=torch.long, device=device)
    for i, ids in enumerate(batch_input_ids):
        Lloc = len(ids)
        input_ids[i, :Lloc] = torch.tensor(ids, dtype=torch.long, device=device)
        attention_mask[i, :Lloc] = 1

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    logprobs = F.log_softmax(logits, dim=-1)

    logp_list = []
    for i in range(B):
        eff = int(attention_mask[i].sum().item())
        start = batch_prefix_lens[i]
        start = min(start, eff)
        if start < 1 or start >= eff:
            logp_list.append(logprobs[i, 0, 0] * 0)
            continue
        lp = logprobs[i, start - 1 : eff - 1, :]
        ids = input_ids[i, start:eff]
        lp_tokens = lp.gather(-1, ids.unsqueeze(-1)).squeeze(-1)
        logp_list.append(lp_tokens.sum())

    return torch.stack(logp_list, dim=0)


def eval_pairwise_accuracy(
    val_loader,
    tokenizer,
    policy_model,
    device: str,
    max_prompt_len: int = 768,
    max_full_len: int = 1024,
    use_chat_template: bool = False,
    desc: Optional[str] = None,
):
    """Доля пар, где модель правильно предпочитает chosen над rejected."""
    policy_model.eval()
    correct = 0
    total = 0
    loader = tqdm(val_loader, desc=desc or "pairwise acc", leave=False) if desc else val_loader
    with torch.no_grad():
        for batch in loader:
            prompts = batch["prompt"]
            chosen = batch["chosen"]
            rejected = batch["rejected"]

            s_c = get_logps(
                policy_model,
                tokenizer,
                prompts,
                chosen,
                device,
                max_prompt_len,
                max_full_len,
                use_chat_template=use_chat_template,
            )
            s_r = get_logps(
                policy_model,
                tokenizer,
                prompts,
                rejected,
                device,
                max_prompt_len,
                max_full_len,
                use_chat_template=use_chat_template,
            )

            preds = s_c > s_r
            correct += preds.sum().item()
            total += preds.numel()
    return correct / max(1, total)


def eval_pairwise_nll(
    val_loader,
    tokenizer,
    policy_model,
    device: str,
    beta: float = 1.0,
    max_prompt_len: int = 768,
    max_full_len: int = 1024,
    use_chat_template: bool = False,
    desc: Optional[str] = None,
):
    """Pairwise NLL по всем сэмплам (chosen vs rejected). Агрегация по сумме NLL / кол-во сэмплов, не по батчам."""
    policy_model.eval()
    total_nll = 0.0
    total_count = 0
    loader = tqdm(val_loader, desc=desc or "pairwise NLL", leave=False) if desc else val_loader
    with torch.no_grad():
        for batch in loader:
            prompts = batch["prompt"]
            chosen = batch["chosen"]
            rejected = batch["rejected"]

            s_c = get_logps(
                policy_model,
                tokenizer,
                prompts,
                chosen,
                device,
                max_prompt_len,
                max_full_len,
                use_chat_template=use_chat_template,
            )
            s_r = get_logps(
                policy_model,
                tokenizer,
                prompts,
                rejected,
                device,
                max_prompt_len,
                max_full_len,
                use_chat_template=use_chat_template,
            )

            diff = beta * (s_c - s_r)
            q = torch.sigmoid(diff)
            nll = -torch.log(q + 1e-12)
            total_nll += nll.sum().item()
            total_count += nll.numel()
    return total_nll / max(1, total_count)
