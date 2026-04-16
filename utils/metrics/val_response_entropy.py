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

    gen_kwargs = {
        "max_new_tokens": int(max_new_tokens),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "num_return_sequences": int(num_samples_per_prompt),
        "return_dict_in_generate": True,
        "output_scores": True,
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
                scores = generated.scores
                if not scores:
                    continue
                # [steps, B*K, vocab] -> [B*K, steps]
                h_steps = []
                for step_scores in scores:
                    step_logp = F.log_softmax(step_scores.float(), dim=-1)
                    step_p = torch.exp(step_logp)
                    h_t = -(step_p * step_logp).sum(dim=-1)
                    h_steps.append(h_t)
                h_by_seq = torch.stack(h_steps, dim=0).transpose(0, 1)

                in_len = inputs["input_ids"].shape[1]
                gen_tokens = generated.sequences[:, in_len:]
                steps = int(h_by_seq.shape[1])
                limit = int(min(entropy_tokens_limit, steps))
                eos_id = tokenizer.eos_token_id

                per_seq_entropy: List[float] = []
                for i in range(h_by_seq.shape[0]):
                    valid = limit
                    if eos_id is not None and gen_tokens.shape[1] > 0:
                        eos_pos = (gen_tokens[i] == eos_id).nonzero(as_tuple=False)
                        if eos_pos.numel() > 0:
                            valid = min(valid, int(eos_pos[0, 0].item()) + 1)
                    if valid <= 0:
                        continue
                    per_seq_entropy.append(float(h_by_seq[i, :valid].mean().item()))

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
