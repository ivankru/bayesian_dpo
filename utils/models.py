# -*- coding: utf-8 -*-
"""
Load policy, reference, and tokenizer for DPO (soft/hard).

By default no separate base copy is loaded for ref: ref is the same PeftModel with LoRA
temporarily disabled (_PeftRefProxy). This saves ~14 GB VRAM on 7B bf16 (≈6 GB on 3B, ≈8 GB on 4B)
and frees room for a larger batch or disabling gradient checkpointing. For code that needs a real
standalone PreTrainedModel (e.g. TRL DPOTrainer), pass share_ref_with_policy=False.
"""
import os
from contextlib import contextmanager
from typing import List, Optional, Sequence, Union

import torch
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.config import BASE_MODEL_3B

# LoRA target_modules presets (Qwen2/Qwen3; any arch with these module names):
#   "all" — all linear projections (Q/K/V/O + MLP: gate/up/down); default. Standard
#           choice in modern LoRA/QLoRA recipes (incl. TRL DPO). Yields ~2–3× more
#           trainable params at the same rank, still < 1% of base; VRAM growth is mostly
#           AdamW state (2× fp32 per trainable param) — tens of MB vs hundreds of GB activations, negligible.
#   "attn" — Q/K/V/O only (~1/3 of block params); lowest VRAM per step,
#           but LoRA touches attention only, not MLP (which holds most model
#           "knowledge"). Often limits adapter expressiveness for preference alignment
#           (especially soft/Bayes-DPO). Kept for reproducing older runs.
_LORA_TARGETS_ATTN: List[str] = ["q_proj", "k_proj", "v_proj", "o_proj"]
_LORA_TARGETS_ALL: List[str] = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

LoraTargetSpec = Union[str, Sequence[str]]


def _resolve_lora_target_modules(spec: LoraTargetSpec) -> List[str]:
    if isinstance(spec, str):
        key = spec.lower()
        if key == "attn":
            return list(_LORA_TARGETS_ATTN)
        if key == "all":
            return list(_LORA_TARGETS_ALL)
        raise ValueError(
            f"Unknown lora_target_modules preset={spec!r}; expected 'attn', 'all' "
            "or a list of module names."
        )
    mods = [str(m).strip() for m in spec if str(m).strip()]
    if not mods:
        raise ValueError("lora_target_modules: empty list")
    return mods


def resolve_peft_adapter_dir(resume_from: str) -> str:
    """
    Directory with PEFT adapter (adapter_config.json + weights).

    Run root (.../run_id) with best/ subdir is accepted, as after train_dpo.
    """
    p = os.path.abspath(os.path.expanduser(resume_from))
    if not os.path.isdir(p):
        raise FileNotFoundError(f"Checkpoint not found (not a directory): {resume_from!r} -> {p}")
    if os.path.isfile(os.path.join(p, "adapter_config.json")):
        return p
    best = os.path.join(p, "best")
    if os.path.isfile(os.path.join(best, "adapter_config.json")):
        return best
    raise ValueError(
        f"No PEFT adapter in {p!r}: expected adapter_config.json in this directory or in best/."
    )


@contextmanager
def _temporarily_disable_gradient_checkpointing(model):
    """If model has gradient checkpointing on, disable for the block and restore in finally.
    Otherwise no-op. Safe if model lacks *_disable/*_enable (also no-op).

    Why: `_PeftRefProxy` ref forward always runs under `torch.no_grad()`. If the shared
    PeftModel has gradient checkpointing enabled, `torch.utils.checkpoint.checkpoint(...)`
    still runs per layer and (a) wastes time on recompute (no backward), (b) emits
    UserWarning «None of the inputs have requires_grad=True. Gradients will be None»
    because under no_grad `enable_input_require_grads` is effectively a no-op.
    Temporarily disabling removes both without affecting the trainable policy forward.
    """
    is_on = bool(getattr(model, "is_gradient_checkpointing", False))
    if not is_on or not hasattr(model, "gradient_checkpointing_disable"):
        yield
        return
    model.gradient_checkpointing_disable()
    try:
        yield
    finally:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()


class _PeftRefProxy:
    """
    Transparent ref-model proxy over PeftModel.

    Any forward/generate temporarily disables LoRA via peft_model.disable_adapter(),
    so the call matches base-model forward without LoRA. Other attributes
    (config, generation_config, eval/train, etc.) delegate to the underlying PeftModel.

    Avoids holding a second full base copy in VRAM.

    Behavior notes:
      - forward/generate do not wrap no_grad automatically; callers must use
        torch.no_grad() as with a real ref_model.
      - Gradient checkpointing on the PeftModel is disabled for the call (if enabled)
        — otherwise ref under no_grad still runs torch.utils.checkpoint (recompute without backward)
        and warns about requires_grad. Checkpointing state is restored afterward; trainable policy forward unchanged.
      - .generate() temporarily sets config.use_cache=True because policy_model.config.use_cache
        is False for training; without KV-cache generation is much slower.
        (use_cache conflicts with gradient checkpointing, but we disable checkpointing here — belt and suspenders.)
      - Proxy only holds a reference to peft_model — no extra parameters, savings ~= one base model.
    """

    def __init__(self, peft_model) -> None:
        if not hasattr(peft_model, "disable_adapter"):
            raise TypeError(
                "_PeftRefProxy expects PeftModel (with disable_adapter); "
                f"got {type(peft_model).__name__}"
            )
        object.__setattr__(self, "_peft", peft_model)

    def __getattr__(self, name: str):
        # __getattr__ only if normal lookup misses the attribute:
        # _peft lives in __dict__, so no recursion.
        return getattr(self._peft, name)

    def __setattr__(self, name: str, value) -> None:
        if name == "_peft":
            object.__setattr__(self, name, value)
        else:
            setattr(self._peft, name, value)

    def __call__(self, *args, **kwargs):
        with _temporarily_disable_gradient_checkpointing(self._peft):
            with self._peft.disable_adapter():
                return self._peft(*args, **kwargs)

    def generate(self, *args, **kwargs):
        cfg = getattr(self._peft, "config", None)
        saved_cache = getattr(cfg, "use_cache", None) if cfg is not None else None
        if cfg is not None:
            cfg.use_cache = True
        try:
            with _temporarily_disable_gradient_checkpointing(self._peft):
                with self._peft.disable_adapter():
                    return self._peft.generate(*args, **kwargs)
        finally:
            if cfg is not None and saved_cache is not None:
                cfg.use_cache = saved_cache

    def __repr__(self) -> str:
        return f"_PeftRefProxy(over={type(self._peft).__name__})"


def load_models_and_tokenizer(
    model_name: str = BASE_MODEL_3B,
    use_lora: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_target_modules: LoraTargetSpec = "all",
    resume_from: Optional[str] = None,
    share_ref_with_policy: bool = True,
):
    """
    Load tokenizer, policy (base + LoRA or from checkpoint), and reference.

    resume_from: directory with adapter_config.json or run root with best/ subdir.
    If set, tokenizer and policy (LoRA) load from the resolved adapter.

    lora_target_modules: modules to wrap with LoRA (only when use_lora=True
        and resume_from is None — on resume, target_modules come from saved adapter_config.json):
          - "all" (default): Q/K/V/O + MLP (gate_proj, up_proj, down_proj) — standard
                             modern LoRA/QLoRA choice. MLP holds most model
                             "knowledge"; without it LoRA only shifts attention patterns,
                             often too weak for soft/Bayes-DPO.
          - "attn":          Q/K/V/O — legacy repo behavior (fewest trainable params,
                             limited adapter capacity).
          - list of strings: custom module names (must exist in the architecture).
        Switching changes trainable param count (~×2–3 at same rank) and AdamW state memory;
        activation VRAM barely moves because LoRA deltas are tiny vs main forward. IMPORTANT: preset changes training semantics, so
        runs with different lora_target_modules are not directly comparable.

    share_ref_with_policy:
      - True (default) with LoRA: no separate base for ref
        is loaded; returns _PeftRefProxy over policy_model that disables LoRA during
        forward/generate. Memory savings ~= one full base.
      - False: legacy — separate frozen base copy (needed when ref
        is passed to code expecting a real PreTrainedModel, e.g. TRL DPOTrainer).
      - If use_lora=False and resume_from=None, policy is not PeftModel, so
        shared ref is impossible and a separate copy is always used (stderr warning).

    Returns:
        tokenizer, policy_model, ref_model, device_gpu
    """
    dtype_gpu = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    device_gpu = "cuda" if torch.cuda.is_available() else "cpu"

    if resume_from:
        adapter_dir = resolve_peft_adapter_dir(resume_from)
        tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    policy_is_peft = bool(resume_from) or use_lora
    load_separate_ref = (not share_ref_with_policy) or (not policy_is_peft)
    if not share_ref_with_policy and not policy_is_peft:
        # Explicit separate ref with non-PEFT policy — valid, just noting.
        pass
    if share_ref_with_policy and not policy_is_peft:
        import sys

        print(
            "Warning: share_ref_with_policy=True requested but policy is not PeftModel "
            "(use_lora=False and resume_from=None) — loading separate ref_model.",
            file=sys.stderr,
        )

    ref_model = None
    if load_separate_ref:
        ref_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype_gpu,
            device_map=device_gpu,
        )
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False

    policy_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype_gpu,
        device_map=device_gpu,
    )
    policy_model.config.use_cache = False

    if resume_from:
        # Without is_trainable=True PEFT keeps LoRA frozen (trainable params: 0)
        # and finetuning silently does nothing; checkpoints often saved under inference_mode=True.
        policy_model = PeftModel.from_pretrained(
            policy_model, adapter_dir, is_trainable=True
        )
        policy_model.print_trainable_parameters()
    elif use_lora:
        resolved_targets = _resolve_lora_target_modules(lora_target_modules)
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=resolved_targets,
            bias="none",
        )
        policy_model = get_peft_model(policy_model, lora_config)
        import sys

        print(
            f"LoRA target_modules ({lora_target_modules!r} -> {resolved_targets})",
            file=sys.stderr,
        )
        policy_model.print_trainable_parameters()

    if hasattr(policy_model, "enable_input_require_grads"):
        policy_model.enable_input_require_grads()
        if hasattr(policy_model, "gradient_checkpointing_enable"):
            policy_model.gradient_checkpointing_enable()
    elif hasattr(policy_model, "gradient_checkpointing_enable"):
        import sys
        print("Warning: gradient checkpointing skipped (no enable_input_require_grads); training will use more VRAM.", file=sys.stderr)

    if ref_model is None:
        # policy_is_peft is guaranteed (else load_separate_ref would be True above).
        ref_model = _PeftRefProxy(policy_model)

    return tokenizer, policy_model, ref_model, device_gpu
