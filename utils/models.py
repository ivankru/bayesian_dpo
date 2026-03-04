# -*- coding: utf-8 -*-
"""
Загрузка модели, reference и токенайзера для DPO (soft/hard).
"""
from typing import Optional

import torch
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.config import BASE_MODEL_3B


def load_models_and_tokenizer(
    model_name: str = BASE_MODEL_3B,
    use_lora: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    resume_from: Optional[str] = None,
):
    """
    Загружает tokenizer, policy (база + LoRA или из чекпоинта) и reference (замороженная база).

    resume_from: путь к папке чекпоинта (например checkpoints/soft_dpo_steer/best).
    Если задан, загружаются tokenizer и policy (LoRA) из чекпоинта; ref — свежая база из model_name.

    Returns:
        tokenizer, policy_model, ref_model, device_gpu
    """
    dtype_gpu = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    device_gpu = "cuda" if torch.cuda.is_available() else "cpu"

    if resume_from:
        tokenizer = AutoTokenizer.from_pretrained(resume_from)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

    # Reference: базовая модель (замороженная)
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype_gpu,
        device_map=device_gpu,
    )
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # Policy: из чекпоинта или новая база + LoRA
    policy_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype_gpu,
        device_map=device_gpu,
    )
    policy_model.config.use_cache = False

    if resume_from:
        policy_model = PeftModel.from_pretrained(policy_model, resume_from)
        policy_model.print_trainable_parameters()
    elif use_lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
        )
        policy_model = get_peft_model(policy_model, lora_config)
        policy_model.print_trainable_parameters()

    # Gradient checkpointing
    if hasattr(policy_model, "enable_input_require_grads"):
        policy_model.enable_input_require_grads()
        if hasattr(policy_model, "gradient_checkpointing_enable"):
            policy_model.gradient_checkpointing_enable()
    elif hasattr(policy_model, "gradient_checkpointing_enable"):
        import sys
        print("Warning: gradient checkpointing skipped (no enable_input_require_grads); training will use more VRAM.", file=sys.stderr)

    return tokenizer, policy_model, ref_model, device_gpu
