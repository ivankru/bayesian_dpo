# -*- coding: utf-8 -*-
"""
Monte Carlo estimate of forward KL(π_θ || π_ref) from samples y ~ π_θ(·|x).

Mean log-ratio E_{x~D, y~π_θ(·|x)}[ log π_θ(y|x) - log π_ref(y|x) ] where D is a prompt draw
from val (distinct from the preference dataset distribution). With enough samples this is an
unbiased-in-y estimate of ∫ π_θ(y|x) log(π_θ(y|x)/π_ref(y|x)) dy per x, averaged over prompts —
i.e. policy-based sample MC, not fixed chosen/rejected strings from data.
"""
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .dpo_logps import get_logps

def _effective_tokenizer_cap(tokenizer) -> int:
    """Upper bound on length for tokenizer truncation; not tied to MAX_PROMPT_LEN in config."""
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
    """Backward-compatible helper for callers importing this symbol from utils.metrics."""
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


def _sum_logprobs_on_generated_suffix(
    model,
    seq_ids: torch.Tensor,
    seq_attn: torch.Tensor,
    *,
    suffix_start: int,
    pad_token_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sum log p(y_t | x, y_<t) over suffix y = seq_ids[:, suffix_start:].
    Returns:
      - seq_logp: [B] total log-prob on the suffix
      - tok_count: [B] counted tokens in the suffix
    """
    # logits[:, t-1] predicts token at position t
    out = model(input_ids=seq_ids, attention_mask=seq_attn)
    logprobs = F.log_softmax(out.logits, dim=-1)

    tgt = seq_ids[:, suffix_start:]
    if tgt.numel() == 0:
        bsz = seq_ids.size(0)
        z = torch.zeros(bsz, device=seq_ids.device, dtype=logprobs.dtype)
        return z, z.to(torch.long)

    # predictor positions are shifted by -1
    pred = logprobs[:, suffix_start - 1 : -1, :]
    tok_lp = pred.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)

    # ignore right-padding introduced after EOS
    valid = (tgt != pad_token_id).to(tok_lp.dtype)
    seq_logp = (tok_lp * valid).sum(dim=1)
    tok_count = valid.sum(dim=1).to(torch.long)
    return seq_logp, tok_count


def _build_full_attention_mask(
    base_attention_mask: torch.Tensor,
    generated: torch.Tensor,
) -> torch.Tensor:
    """
    base_attention_mask: [B, Tin]
    generated: [B*K, Tout] where Tout >= Tin
    Returns attention mask [B*K, Tout]:
      - first Tin positions: repeated base mask
      - generated tail: ones
    """
    b = base_attention_mask.size(0)
    bk = generated.size(0)
    k = bk // b
    tin = base_attention_mask.size(1)
    tout = generated.size(1)
    rep = base_attention_mask.repeat_interleave(k, dim=0)
    tail = torch.ones((bk, max(0, tout - tin)), dtype=rep.dtype, device=rep.device)
    return torch.cat([rep, tail], dim=1)


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
    MC estimate of forward KL(π_θ || π_ref). Returns dict:

    - ``per_seq``: (1/N) Σ_i [ log π_θ(y_i|x_i) - log π_ref(y_i|x_i) ], y_i ~ π_θ(·|x_i), N = P*K.
      Mean log-ratio per sequence. Depends on average generation length —
      weak for cross-run comparisons.
    - ``per_token``: Σ_i [ log π_θ(y_i|x_i) - log π_ref(y_i|x_i) ] / Σ_i n_tokens(y_i).
      KL per token. Length-invariant, preferred when comparing runs with different generation lengths.
    - ``total_seqs``: number of sequences actually scored.
    - ``total_tokens``: total response tokens (∑ n_tokens).

    Memory: prompts in batches of ``prompt_batch_size``; log-probs and log-ratios accumulated in
    microbatches of ``logp_score_batch_size`` without holding all strings at once.
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
        # top_k=0 disables TopKLogitsWarper in HF (else default generation_config.top_k=50 would apply)
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
                full_attn = _build_full_attention_mask(inputs["attention_mask"], generated)

                for sub in range(0, b_times_k, logp_score_batch_size):
                    sub_end = min(sub + logp_score_batch_size, b_times_k)
                    gen_sub = generated[sub:sub_end]
                    attn_sub = full_attn[sub:sub_end]

                    log_pi, tok_pi = _sum_logprobs_on_generated_suffix(
                        policy_model,
                        gen_sub,
                        attn_sub,
                        suffix_start=in_len,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                    log_ref, tok_ref = _sum_logprobs_on_generated_suffix(
                        ref_model,
                        gen_sub,
                        attn_sub,
                        suffix_start=in_len,
                        pad_token_id=tokenizer.pad_token_id,
                    )

                    if not torch.equal(tok_pi, tok_ref):
                        raise RuntimeError(
                            "KL_MC token-count mismatch between policy/ref scoring"
                        )

                    total_log_ratio += (log_pi - log_ref).sum().item()
                    total_count += int(gen_sub.size(0))
                    total_resp_tokens += int(tok_pi.sum().item())

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
