# -*- coding: utf-8 -*-
"""
Оценка энтропии токенов ответов policy на валидационных промптах.

Метрика считает H_t = -sum_v p_t(v) log p_t(v) по logits генерации и агрегирует:
- по токенам ответа (с ограничением на первые L токенов),
- по K сэмплам для каждого prompt,
- по множеству prompt'ов (mean / median / p10 / p90 / std).
"""
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def _response_token_mask(gen_tail: torch.Tensor, eos_id) -> torch.Tensor:
    """
    [B, T] — True для токенов ответа, входящих в среднее (до первого EOS включительно;
    если EOS в префиксе из T токенов нет — все T позиций).
    """
    b, t = gen_tail.shape
    if eos_id is None:
        return torch.ones((b, t), dtype=torch.bool, device=gen_tail.device)
    if isinstance(eos_id, int):
        hit = gen_tail == eos_id
    else:
        eos_t = torch.as_tensor(eos_id, device=gen_tail.device, dtype=gen_tail.dtype).view(1, -1)
        hit = (gen_tail.unsqueeze(-1) == eos_t).any(dim=-1)
    col = torch.arange(t, device=gen_tail.device).unsqueeze(0).expand(b, -1)
    has = hit.any(dim=1, keepdim=True)
    first = torch.where(
        has,
        hit.int().argmax(dim=1, keepdim=True),
        torch.full((b, 1), t - 1, device=gen_tail.device, dtype=torch.long),
    )
    return col <= first


def _effective_tokenizer_cap(tokenizer) -> int:
    """Безопасный cap для tokenizer.model_max_length."""
    cap = getattr(tokenizer, "model_max_length", None)
    if cap is None or cap <= 0 or cap > 1_000_000:
        cap = 8192
    return int(min(cap, 8192))


def estimate_val_response_entropy(
    policy_model,
    tokenizer,
    val_prompts: Sequence[str],
    device: str,
    num_samples_per_prompt: int = 4,
    max_new_tokens: int = 128,
    entropy_tokens_limit: int = 128,
    top_k: int = 0,
    top_p: float = 1.0,
    temperature: float = 1.0,
    use_chat_template: bool = False,
    prompt_batch_size: int = 6,
    forward_chunk_size: int = 2,
    cuda_empty_cache_between_batches: bool = True,
    show_progress: bool = True,
) -> Dict[str, float]:
    """
    Возвращает статистики распределения mean-token-entropy по prompt'ам.

    Для каждого prompt генерируется K ответов, для каждого ответа считается средняя
    энтропия по первым min(L, T_resp) токенам, затем среднее по K.
    """
    if num_samples_per_prompt < 1:
        raise ValueError(
            f"num_samples_per_prompt must be >= 1, got {num_samples_per_prompt}"
        )
    if prompt_batch_size < 1:
        raise ValueError(f"prompt_batch_size must be >= 1, got {prompt_batch_size}")
    if forward_chunk_size < 1:
        raise ValueError(f"forward_chunk_size must be >= 1, got {forward_chunk_size}")
    if max_new_tokens < 1:
        raise ValueError(f"max_new_tokens must be >= 1, got {max_new_tokens}")
    if entropy_tokens_limit < 1:
        raise ValueError(
            f"entropy_tokens_limit must be >= 1, got {entropy_tokens_limit}"
        )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts_list: List[str] = (
        list(val_prompts) if not isinstance(val_prompts, list) else val_prompts
    )
    if not prompts_list:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "p10": float("nan"),
            "p90": float("nan"),
            "std": float("nan"),
            "num_prompts": 0.0,
        }

    cap_enc = _effective_tokenizer_cap(tokenizer)

    # Не используем output_scores: на части связок (Peft + Qwen + device_map / конфиг generate)
    # `generated.scores` приходит пустым, и метрика превращается в NaN. Энтропию считаем по одному
    # forward на полной сгенерированной последовательности (logits[t] предсказывает token[t+1]).
    gen_kwargs = {
        "max_new_tokens": int(max_new_tokens),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "num_return_sequences": int(num_samples_per_prompt),
    }
    if temperature is not None and temperature > 0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = float(temperature)
        gen_kwargs["top_k"] = int(top_k)
        if top_p < 1.0:
            gen_kwargs["top_p"] = float(top_p)
    else:
        gen_kwargs["do_sample"] = False

    prompt_entropy_values: List[float] = []
    saved_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    policy_model.eval()

    try:
        outer = range(0, len(prompts_list), prompt_batch_size)
        if show_progress:
            outer = tqdm(
                outer,
                desc="val response entropy",
                leave=False,
                total=(len(prompts_list) + prompt_batch_size - 1) // prompt_batch_size,
            )

        with torch.no_grad():
            for start in outer:
                batch_prompts = prompts_list[start : start + prompt_batch_size]
                if use_chat_template:
                    texts = []
                    for p in batch_prompts:
                        t = tokenizer.apply_chat_template(
                            [{"role": "user", "content": p}],
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                        texts.append(t)
                    inputs = tokenizer(
                        texts,
                        padding=True,
                        return_tensors="pt",
                        truncation=True,
                        max_length=cap_enc,
                    )
                else:
                    inputs = tokenizer(
                        batch_prompts,
                        padding=True,
                        return_tensors="pt",
                        truncation=True,
                        max_length=cap_enc,
                    )
                inputs = {k: v.to(device) for k, v in inputs.items()}

                generated = policy_model.generate(**inputs, **gen_kwargs)
                seq = getattr(generated, "sequences", generated)
                in_len = int(inputs["input_ids"].shape[1])
                gen_tokens = seq[:, in_len:]
                t_resp = int(gen_tokens.shape[1])
                if t_resp <= 0:
                    continue
                limit = int(min(entropy_tokens_limit, t_resp))
                eos_id = tokenizer.eos_token_id

                pad_id = tokenizer.pad_token_id
                attn = (
                    (seq != pad_id).long()
                    if pad_id is not None
                    else torch.ones_like(seq, dtype=torch.long)
                )

                per_seq_entropy: List[float] = []
                fc = int(forward_chunk_size)
                for c0 in range(0, seq.shape[0], fc):
                    c1 = min(c0 + fc, seq.shape[0])
                    sub = seq[c0:c1]
                    sub_attn = attn[c0:c1]
                    logits = policy_model(
                        input_ids=sub,
                        attention_mask=sub_attn,
                    ).logits.float()
                    # logits[b, pos] -> token[b, pos+1]; первый новый токен в pos=in_len
                    slice_logits = logits[:, in_len - 1 : in_len - 1 + limit, :]
                    if temperature is not None and float(temperature) > 0:
                        slice_logits = slice_logits / float(temperature)
                    step_logp = F.log_softmax(slice_logits, dim=-1)
                    step_p = torch.exp(step_logp)
                    h_tok = -(step_p * step_logp).sum(dim=-1)
                    tail = gen_tokens[c0:c1, :limit]
                    keep = _response_token_mask(tail, eos_id)
                    for j in range(h_tok.shape[0]):
                        sel = h_tok[j][keep[j]]
                        if sel.numel() == 0:
                            continue
                        per_seq_entropy.append(float(sel.mean().item()))
                    del logits, slice_logits, step_logp, step_p, h_tok, tail, keep

                if (
                    cuda_empty_cache_between_batches
                    and isinstance(device, str)
                    and device.startswith("cuda")
                    and torch.cuda.is_available()
                ):
                    torch.cuda.empty_cache()

                bsz = len(batch_prompts)
                k = int(num_samples_per_prompt)
                for p_ix in range(bsz):
                    lo = p_ix * k
                    hi = min((p_ix + 1) * k, len(per_seq_entropy))
                    if lo >= hi:
                        continue
                    prompt_entropy_values.append(
                        float(np.mean(per_seq_entropy[lo:hi]))
                    )
    finally:
        tokenizer.padding_side = saved_padding_side

    if not prompt_entropy_values:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "p10": float("nan"),
            "p90": float("nan"),
            "std": float("nan"),
            "num_prompts": 0.0,
        }

    arr = np.asarray(prompt_entropy_values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "std": float(np.std(arr)),
        "num_prompts": float(arr.size),
    }
