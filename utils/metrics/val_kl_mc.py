# -*- coding: utf-8 -*-
"""
Monte Carlo оценка forward KL(π_θ || π_ref) по сэмплам y ~ π_θ(·|x).

Это среднее лог-отношения E_{x~D, y~π_θ(·|x)}[ log π_θ(y|x) - log π_ref(y|x) ], где D — выбор промптов
из val (отдельно от распределения данных preference-датасета). При достаточном числе сэмплов это даёт
несмещённую (по y) оценку ∫ π_θ(y|x) log(π_θ(y|x)/π_ref(y|x)) dy для каждого x, усреднённую по
выбранным промптам — т.е. sample-based MC по политике, а не по фиксированным chosen/rejected из данных.
"""
from typing import Dict, List, Sequence, Tuple

import torch
from tqdm import tqdm

from .dpo_logps import get_logps


def _effective_tokenizer_cap(tokenizer) -> int:
    """Верхняя граница длины для truncation из tokenizer; без привязки к MAX_PROMPT_LEN в config."""
    cap = getattr(tokenizer, "model_max_length", None)
    if cap is None or cap <= 0 or cap > 1_000_000:
        cap = 8192
    return int(min(cap, 8192))


def get_logps_generated(
    model,
    tokenizer,
    prompts: List[str],
    responses: List[str],
    device: str,
    use_chat_template: bool = False,
) -> torch.Tensor:
    """
    log p(response | prompt) для произвольных строковых пар (в т.ч. после decode сгенерированных токенов).

    Делегирует в get_logps с max_prompt_len=max_full_len=ef_cap, где ef_cap из tokenizer.model_max_length
    (с разумной планкой), чтобы не отрезать ответ при типичных длинах генерации.
    """
    cap = _effective_tokenizer_cap(tokenizer)
    return get_logps(
        model,
        tokenizer,
        prompts,
        responses,
        device,
        max_prompt_len=cap,
        max_full_len=cap,
        use_chat_template=use_chat_template,
    )


def _count_response_tokens(
    tokenizer,
    responses: List[str],
) -> List[int]:
    """Число токенов ответа (без обрезки и паддинга) для per-token нормализации KL-MC.

    Использует ту же токенизацию без специальных токенов, что и get_logps в chat-template
    ветке; даёт верхнюю оценку числа токенов, по которым берётся log p. Для согласованности с
    get_logps при чрезмерно длинных ответах (> max_full_len - prefix) можно было бы учесть
    обрезку, но для KL-MC (cap ~8192) обрезка крайне редка при max_new_tokens порядка сотен.
    """
    out: List[int] = []
    for r in responses:
        ids = tokenizer(r, add_special_tokens=False)["input_ids"]
        out.append(max(1, len(ids)))
    return out


def estimate_val_kl_mc(
    policy_model,
    ref_model,
    tokenizer,
    val_prompts: Sequence[str],
    device: str,
    num_samples_per_prompt: int = 4,
    max_new_tokens: int = 128,
    top_k: int = 0,
    top_p: float = 1.0,
    temperature: float = 1.0,
    use_chat_template: bool = False,
    prompt_batch_size: int = 6,
    logp_score_batch_size: int = 16,
    show_progress: bool = True,
) -> Dict[str, float]:
    """
    MC-оценка forward KL(π_θ || π_ref). Возвращает dict:

    - ``per_seq``: (1/N) Σ_i [ log π_θ(y_i|x_i) - log π_ref(y_i|x_i) ], где y_i ~ π_θ(·|x_i), N = P*K.
      «Среднее лог-отношение на последовательность». Зависит от средней длины генерации —
      плохо для кросс-прогонных сравнений.
    - ``per_token``: Σ_i [ log π_θ(y_i|x_i) - log π_ref(y_i|x_i) ] / Σ_i n_tokens(y_i).
      «KL на токен». Инвариантна к длине ответа, что делает её предпочтительной для
      сравнения прогонов с разной длиной генерации.
    - ``total_seqs``: фактическое число просчитанных последовательностей.
    - ``total_tokens``: суммарное число ответных токенов (∑ n_tokens).

    Память: промпты батчами по ``prompt_batch_size``; лог-вероятности и лог-отношения накоплениями по
    микробатчам ``logp_score_batch_size`` без хранения всех строк одновременно.
    """
    if num_samples_per_prompt < 1:
        raise ValueError(f"num_samples_per_prompt must be >= 1, got {num_samples_per_prompt}")
    if prompt_batch_size < 1 or logp_score_batch_size < 1:
        raise ValueError("prompt_batch_size and logp_score_batch_size must be >= 1")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts_list: List[str] = (
        list(val_prompts) if not isinstance(val_prompts, list) else val_prompts
    )
    cap_enc = _effective_tokenizer_cap(tokenizer)

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "num_return_sequences": num_samples_per_prompt,
    }
    if temperature is not None and temperature > 0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = float(temperature)
        # top_k=0 отключает TopKLogitsWarper в HF (иначе остался бы дефолт generation_config.top_k=50)
        gen_kwargs["top_k"] = int(top_k)
        if top_p < 1.0:
            gen_kwargs["top_p"] = float(top_p)
    else:
        gen_kwargs["do_sample"] = False

    total_log_ratio = 0.0
    total_count = 0
    total_resp_tokens = 0

    saved_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    policy_model.eval()
    ref_model.eval()

    try:
        outer = range(0, len(prompts_list), prompt_batch_size)
        if show_progress:
            outer = tqdm(
                outer,
                desc="val KL_MC (generate+logp)",
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
                in_len = inputs["input_ids"].shape[1]
                b_times_k = generated.shape[0]

                for sub in range(0, b_times_k, logp_score_batch_size):
                    sub_end = min(sub + logp_score_batch_size, b_times_k)
                    mp: List[str] = []
                    mr: List[str] = []
                    for idx in range(sub, sub_end):
                        p_ix = idx // num_samples_per_prompt
                        gen_ids = generated[idx, in_len:]
                        resp_text = tokenizer.decode(
                            gen_ids, skip_special_tokens=True
                        )
                        mp.append(batch_prompts[p_ix])
                        mr.append(resp_text)

                    log_pi = get_logps_generated(
                        policy_model,
                        tokenizer,
                        mp,
                        mr,
                        device,
                        use_chat_template=use_chat_template,
                    )
                    log_ref = get_logps_generated(
                        ref_model,
                        tokenizer,
                        mp,
                        mr,
                        device,
                        use_chat_template=use_chat_template,
                    )
                    total_log_ratio += (log_pi - log_ref).sum().item()
                    total_count += len(mp)
                    total_resp_tokens += sum(_count_response_tokens(tokenizer, mr))

    finally:
        tokenizer.padding_side = saved_padding_side

    n_seqs = max(1, total_count)
    n_tok = max(1, total_resp_tokens)
    return {
        "per_seq": total_log_ratio / n_seqs,
        "per_token": total_log_ratio / n_tok,
        "total_seqs": float(total_count),
        "total_tokens": float(total_resp_tokens),
    }
