# -*- coding: utf-8 -*-
"""
Классический DPO через TRL (DPOTrainer). Те же входные параметры, что у hard_dpo_steer.
На валидации логируются те же метрики: NLL, acc, KL (и DPO loss), как в hard_dpo_steer.
"""
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from torch.utils.data import DataLoader
from transformers import TrainerCallback
from trl import DPOConfig, DPOTrainer
from tqdm import tqdm

from config.base_config import (
    CAPABILITY_EVAL_BATCH_SIZE,
    CAPABILITY_EVAL_LIMIT,
    CAPABILITY_EVAL_MAX_NEW_TOKENS,
    CAPABILITY_EVAL_MAX_PROMPT_TOKENS,
    USE_CHAT_TEMPLATE,
    VAL_ENTROPY_FORWARD_CHUNK_SIZE,
    VAL_ENTROPY_MAX_NEW_TOKENS,
    VAL_ENTROPY_MAX_PROMPTS,
    VAL_ENTROPY_NUM_SAMPLES,
    VAL_ENTROPY_PROMPT_BATCH_SIZE,
    VAL_KL_MC_NUM_SAMPLES,
    VAL_KL_MC_PROMPT_BATCH_SIZE,
)
from utils.config import BASE_MODEL_CHOICES, MAX_FULL_LEN, MAX_PROMPT_LEN
from utils.seed import set_seed
from utils.datasets import (
    build_dpo_datasets,
    build_dpo_datasets_ultrafeedback,
)
from utils.loss import hard_dpo_loss
from utils.metrics import (
    estimate_val_kl_mc,
    estimate_val_response_entropy,
    eval_pairwise_accuracy,
    eval_pairwise_nll,
    format_capability_retention_log_lines,
    load_eval_rows,
    run_retention_eval_pair,
)
from utils.models import load_models_and_tokenizer
from utils.training import DEFAULT_VAL_KL_MC_MAX_PROMPTS, collate_fn_hard


# ======================
# main
# ======================

DATASET_CHOICES = ("helpsteer3", "ultrafeedback_binarized")


def _classic_capability_retention(
    cap_rows,
    cap_ref_holder: List[Optional[List[str]]],
    tokenizer,
    ref_model,
    policy_model,
    device: str,
    output_dir: str,
    epoch_display: str,
    log_fn,
    max_new_tokens: int,
    batch_size: int,
    max_prompt_tokens: int,
) -> None:
    if cap_rows is None or ref_model is None:
        return
    try:
        desc_ref = (
            f"cap_ret ref ep{epoch_display}"
            if cap_ref_holder[0] is None
            else None
        )
        summary, cap_ref_holder[0] = run_retention_eval_pair(
            tokenizer,
            ref_model,
            policy_model,
            device,
            cap_rows,
            cap_ref_holder[0],
            max_new_tokens,
            batch_size,
            max_prompt_tokens,
            desc_ref=desc_ref,
            desc_pol=f"cap_ret policy ep{epoch_display}",
        )
    except Exception as e:
        log_fn(f"Capability retention: ошибка: {e}")
        return
    for line in format_capability_retention_log_lines(summary, epoch_display):
        log_fn(line)
    tag = epoch_display.replace(".", "_")
    cap_ret_dir = os.path.join(output_dir, "capability_retention")
    os.makedirs(cap_ret_dir, exist_ok=True)
    cap_json = os.path.join(cap_ret_dir, f"capability_retention_epoch{tag}.json")
    try:
        with open(cap_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    except OSError as err:
        log_fn(f"Capability retention: не удалось записать {cap_json}: {err}")


def _classic_log_val_kl_mc(
    val_ds,
    tokenizer,
    policy_model,
    ref_model,
    device: str,
    log_fn,
    val_kl_mc_max_prompts: int,
    val_kl_mc_num_samples: int,
    val_kl_mc_max_new_tokens: int,
    val_kl_mc_prompt_batch_size: int,
    use_chat_template: bool,
) -> None:
    if val_kl_mc_max_prompts <= 0:
        log_fn("validation KL_MC: skipped (val_kl_mc_max_prompts<=0).")
        return
    if len(val_ds) == 0:
        log_fn("validation KL_MC: skipped (empty val_ds).")
        return
    n_mc = min(int(val_kl_mc_max_prompts), len(val_ds))
    log_fn(
        f"validation KL_MC: computing (first {n_mc} val prompts x {val_kl_mc_num_samples} samples)..."
    )
    try:
        mc_prompts = val_ds.select(range(n_mc))["prompt"]
        kl_mc_stats = estimate_val_kl_mc(
            policy_model,
            ref_model,
            tokenizer,
            mc_prompts,
            device,
            num_samples_per_prompt=val_kl_mc_num_samples,
            max_new_tokens=val_kl_mc_max_new_tokens,
            use_chat_template=use_chat_template,
            prompt_batch_size=val_kl_mc_prompt_batch_size,
        )
        val_kl_mc_per_seq = float(kl_mc_stats["per_seq"])
        val_kl_mc_per_token = float(kl_mc_stats["per_token"])
        n_tokens_mc = int(kl_mc_stats["total_tokens"])
        log_fn(
            f"validation KL_MC (pi||ref, MC samples from policy, {n_mc} prompts x "
            f"{val_kl_mc_num_samples}): per_seq={val_kl_mc_per_seq:.4f}, "
            f"per_token={val_kl_mc_per_token:.6f} (total_tokens={n_tokens_mc})"
        )
    except Exception as e:
        log_fn(
            f"validation KL_MC: FAILED ({type(e).__name__}: {e}); continuing without val_kl_mc metric"
        )


def _classic_log_val_response_entropy(
    val_ds,
    tokenizer,
    policy_model,
    device: str,
    log_fn,
    val_entropy_max_prompts: int,
    val_entropy_num_samples: int,
    val_entropy_max_new_tokens: int,
    val_entropy_prompt_batch_size: int,
    val_entropy_forward_chunk_size: int,
    use_chat_template: bool,
) -> None:
    if val_entropy_max_prompts <= 0:
        log_fn("validation response entropy: skipped (val_entropy_max_prompts<=0).")
        return
    if len(val_ds) == 0:
        log_fn("validation response entropy: skipped (empty val_ds).")
        return
    n_ent = min(int(val_entropy_max_prompts), len(val_ds))
    log_fn(
        "validation response entropy: computing "
        f"(first {n_ent} val prompts x {val_entropy_num_samples} samples, "
        f"L={val_entropy_max_new_tokens})..."
    )
    try:
        ent_prompts = val_ds.select(range(n_ent))["prompt"]
        ent_stats = estimate_val_response_entropy(
            policy_model,
            tokenizer,
            ent_prompts,
            str(device),
            num_samples_per_prompt=val_entropy_num_samples,
            max_new_tokens=val_entropy_max_new_tokens,
            entropy_tokens_limit=val_entropy_max_new_tokens,
            use_chat_template=use_chat_template,
            prompt_batch_size=val_entropy_prompt_batch_size,
            forward_chunk_size=val_entropy_forward_chunk_size,
        )
        cfg = getattr(policy_model, "config", None)
        vocab_size = getattr(cfg, "vocab_size", None) if cfg is not None else None
        if not isinstance(vocab_size, int) or vocab_size <= 0:
            vocab_size = getattr(tokenizer, "vocab_size", None)
        if not isinstance(vocab_size, int) or vocab_size <= 0:
            vocab_size = len(tokenizer)
        log_v = math.log(float(vocab_size)) if int(vocab_size) > 1 else float("nan")

        m = float(ent_stats["mean"])
        med = float(ent_stats["median"])
        p10 = float(ent_stats["p10"])
        p90 = float(ent_stats["p90"])
        hdr = (
            "validation response entropy "
            f"(L={val_entropy_max_new_tokens}, {n_ent} prompts x {val_entropy_num_samples})"
        )
        if math.isfinite(log_v) and log_v > 0:
            inv_pct = 100.0 / log_v
            log_fn(
                f"{hdr} - abs (nats): mean={m:.4f} median={med:.4f} p10={p10:.4f} p90={p90:.4f} "
                f"(max uniform = log V = {log_v:.4f}, V={int(vocab_size)})"
            )
            log_fn(
                f"{hdr} - % of max: mean={m * inv_pct:.2f}% median={med * inv_pct:.2f}% "
                f"p10={p10 * inv_pct:.2f}% p90={p90 * inv_pct:.2f}%"
            )
        else:
            log_fn(
                f"{hdr}: mean={m:.4f} median={med:.4f} p10={p10:.4f} p90={p90:.4f}"
            )
    except Exception as e:
        log_fn(
            "validation response entropy: FAILED "
            f"({type(e).__name__}: {e}); continuing without entropy metric"
        )


class DPOValidationMetricsCallback(TrainerCallback):
    """После каждой эпохи валидации пишет в log те же метрики, что и hard_dpo_steer: NLL, acc, KL, DPO loss."""

    def __init__(
        self,
        val_ds,
        tokenizer,
        ref_model,
        device: str,
        log_fn,
        beta: float,
        eval_batch_size: int = 2,
        cap_rows=None,
        cap_ref_holder: Optional[List[Optional[List[str]]]] = None,
        cap_output_dir: Optional[str] = None,
        cap_max_new_tokens: int = 256,
        cap_batch_size: int = 2,
        cap_max_prompt_tokens: int = 2048,
        val_kl_mc_max_prompts: int = DEFAULT_VAL_KL_MC_MAX_PROMPTS,
        val_kl_mc_num_samples: int = VAL_KL_MC_NUM_SAMPLES,
        val_kl_mc_max_new_tokens: int = 128,
        val_kl_mc_prompt_batch_size: int = VAL_KL_MC_PROMPT_BATCH_SIZE,
        val_entropy_max_prompts: int = VAL_ENTROPY_MAX_PROMPTS,
        val_entropy_num_samples: int = VAL_ENTROPY_NUM_SAMPLES,
        val_entropy_max_new_tokens: int = VAL_ENTROPY_MAX_NEW_TOKENS,
        val_entropy_prompt_batch_size: int = VAL_ENTROPY_PROMPT_BATCH_SIZE,
        val_entropy_forward_chunk_size: int = VAL_ENTROPY_FORWARD_CHUNK_SIZE,
        use_chat_template: bool = USE_CHAT_TEMPLATE,
    ):
        self.val_ds = val_ds
        self.tokenizer = tokenizer
        self.ref_model = ref_model
        self.device = device
        self.log_fn = log_fn
        self.beta = beta
        self.eval_batch_size = eval_batch_size
        self._cap_rows = cap_rows
        self._cap_ref_holder = cap_ref_holder
        self._cap_output_dir = cap_output_dir or "."
        self._cap_max_new_tokens = cap_max_new_tokens
        self._cap_batch_size = cap_batch_size
        self._cap_max_prompt_tokens = cap_max_prompt_tokens
        self._val_kl_mc_max_prompts = val_kl_mc_max_prompts
        self._val_kl_mc_num_samples = val_kl_mc_num_samples
        self._val_kl_mc_max_new_tokens = val_kl_mc_max_new_tokens
        self._val_kl_mc_prompt_batch_size = val_kl_mc_prompt_batch_size
        self._val_entropy_max_prompts = val_entropy_max_prompts
        self._val_entropy_num_samples = val_entropy_num_samples
        self._val_entropy_max_new_tokens = val_entropy_max_new_tokens
        self._val_entropy_prompt_batch_size = val_entropy_prompt_batch_size
        self._val_entropy_forward_chunk_size = val_entropy_forward_chunk_size
        self._use_chat_template = use_chat_template

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        val_loader = DataLoader(
            self.val_ds,
            batch_size=self.eval_batch_size,
            shuffle=False,
            collate_fn=collate_fn_hard,
        )
        model.eval()
        if self.ref_model is not None:
            self.ref_model.eval()
        val_dpo_sum = 0.0
        val_kl_sum = 0.0
        val_n = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="val DPO metrics", leave=False):
                loss, kl_b = hard_dpo_loss(
                    batch, self.tokenizer, model, self.ref_model, self.device, beta=self.beta
                )
                n = len(batch["prompt"])
                val_dpo_sum += loss.item() * n
                val_kl_sum += kl_b * n
                val_n += n
        val_dpo = val_dpo_sum / max(1, val_n)
        val_kl = val_kl_sum / max(1, val_n)
        val_nll = eval_pairwise_nll(
            val_loader, self.tokenizer, model, self.device,
            beta=1.0, max_prompt_len=MAX_PROMPT_LEN, max_full_len=MAX_FULL_LEN,
            desc="val pairwise NLL",
        )
        val_acc = eval_pairwise_accuracy(
            val_loader, self.tokenizer, model, self.device,
            max_prompt_len=MAX_PROMPT_LEN, max_full_len=MAX_FULL_LEN,
            desc="val pairwise acc",
        )
        epoch = int(state.epoch) if state.epoch is not None else 1
        self.log_fn("")
        self.log_fn(f"=== Validation, epoch {epoch} ===")
        self.log_fn(f"validation DPO loss   : {val_dpo:.4f}")
        self.log_fn(f"validation logp_gap_mean : {val_kl:.4f}")
        self.log_fn(f"validation pair NLL   : {val_nll:.4f}")
        self.log_fn(f"validation pair acc   : {100 * val_acc:.2f}%")
        _classic_log_val_kl_mc(
            self.val_ds,
            self.tokenizer,
            model,
            self.ref_model,
            self.device,
            self.log_fn,
            self._val_kl_mc_max_prompts,
            self._val_kl_mc_num_samples,
            self._val_kl_mc_max_new_tokens,
            self._val_kl_mc_prompt_batch_size,
            self._use_chat_template,
        )
        _classic_log_val_response_entropy(
            self.val_ds,
            self.tokenizer,
            model,
            self.device,
            self.log_fn,
            self._val_entropy_max_prompts,
            self._val_entropy_num_samples,
            self._val_entropy_max_new_tokens,
            self._val_entropy_prompt_batch_size,
            self._val_entropy_forward_chunk_size,
            self._use_chat_template,
        )
        if self._cap_rows is not None and self._cap_ref_holder is not None:
            _classic_capability_retention(
                self._cap_rows,
                self._cap_ref_holder,
                self.tokenizer,
                self.ref_model,
                model,
                self.device,
                self._cap_output_dir,
                str(epoch),
                self.log_fn,
                self._cap_max_new_tokens,
                self._cap_batch_size,
                self._cap_max_prompt_tokens,
            )


def main(
    resume_from: Optional[str] = None,
    seed: int = 42,
    output_dir: str = "checkpoints/classic_dpo",
    dataset: str = "helpsteer3",
    base_model: str = "3b",
    batch_size: int = 8,
    grad_accum_steps: int = 1,
    epochs: int = 3,
    lr: float = 2e-5,
    beta: float = 0.2,
    use_separate_ref_model: bool = False,
    capability_eval_dir: Optional[str] = None,
    capability_eval_limit: Optional[int] = CAPABILITY_EVAL_LIMIT,
    capability_eval_max_new_tokens: int = CAPABILITY_EVAL_MAX_NEW_TOKENS,
    capability_eval_batch_size: int = CAPABILITY_EVAL_BATCH_SIZE,
    capability_eval_max_prompt_tokens: int = CAPABILITY_EVAL_MAX_PROMPT_TOKENS,
    use_chat_template: bool = USE_CHAT_TEMPLATE,
    val_kl_mc_max_prompts: int = DEFAULT_VAL_KL_MC_MAX_PROMPTS,
    val_kl_mc_num_samples: int = VAL_KL_MC_NUM_SAMPLES,
    val_kl_mc_max_new_tokens: int = 128,
    val_kl_mc_prompt_batch_size: int = VAL_KL_MC_PROMPT_BATCH_SIZE,
    val_entropy_max_prompts: int = VAL_ENTROPY_MAX_PROMPTS,
    val_entropy_num_samples: int = VAL_ENTROPY_NUM_SAMPLES,
    val_entropy_max_new_tokens: int = VAL_ENTROPY_MAX_NEW_TOKENS,
    val_entropy_prompt_batch_size: int = VAL_ENTROPY_PROMPT_BATCH_SIZE,
    val_entropy_forward_chunk_size: int = VAL_ENTROPY_FORWARD_CHUNK_SIZE,
):
    """
    resume_from: путь к чекпоинту для продолжения обучения.
    seed, output_dir, dataset, base_model, batch_size, grad_accum_steps, epochs, lr, beta — по смыслу как в hard_dpo_steer.
    Обучение — классический DPO из библиотеки TRL (DPOTrainer).
    """
    if dataset not in DATASET_CHOICES:
        raise ValueError(f"dataset должен быть один из {DATASET_CHOICES}, получено: {dataset!r}")
    set_seed(seed)
    if dataset == "helpsteer3":
        print("Загружаю HelpSteer3-Preference...")
        train_ds, val_ds = build_dpo_datasets()
    else:
        print("Загружаю UltraFeedback Binarized...")
        train_ds, val_ds = build_dpo_datasets_ultrafeedback()
    model_name = BASE_MODEL_CHOICES[base_model]
    print(f"Model: {model_name}, Dataset: {dataset}, train size: {len(train_ds)}, val size: {len(val_ds)}")
    if resume_from:
        print(f"Загружаю модель из чекпоинта: {resume_from} (база {model_name})")
    else:
        print(f"Загружаю модель и токенайзер: {model_name} (LoRA)")
    if grad_accum_steps < 1:
        raise ValueError(f"grad_accum_steps должен быть >= 1, получено: {grad_accum_steps}")
    if epochs < 1:
        raise ValueError(f"epochs должен быть >= 1, получено: {epochs}")

    tokenizer, policy_model, ref_model, device = load_models_and_tokenizer(
        model_name,
        use_lora=True,
        lora_r=16,
        lora_alpha=32,
        resume_from=resume_from,
        share_ref_with_policy=not use_separate_ref_model,
    )

    policy_model.config.use_cache = False
    if ref_model is not None:
        ref_model.config.use_cache = False

    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "train.log")
    run_started_at = datetime.now()
    run_started_perf = time.perf_counter()
    # Каждый новый запуск начинает train.log с чистого листа.
    with open(log_path, "w", encoding="utf-8"):
        pass

    def log_fn(msg: str) -> None:
        # Лог-строки идут в stdout, чтобы не смешиваться с tqdm-прогрессами (stderr).
        # Это даёт чистое `>run.log` с только осмысленными строками.
        print(msg, flush=True, file=sys.stdout)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    log_fn(f"Run started at: {run_started_at.strftime('%Y-%m-%d %H:%M:%S')}")

    training_args = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        per_device_eval_batch_size=2,
        learning_rate=lr,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported() and torch.cuda.is_available(),
        gradient_checkpointing=True,
        max_length=MAX_FULL_LEN,
        max_prompt_length=MAX_PROMPT_LEN,
        remove_unused_columns=False,
        beta=beta,
    )

    log_fn("=== Classic DPO (TRL DPOTrainer) ===")
    log_fn(f"Model: {model_name}, Dataset: {dataset}, train size: {len(train_ds)}, val size: {len(val_ds)}")
    log_fn(
        f"beta={beta}, lr={lr}, batch_size={batch_size}, "
        f"grad_accum_steps={grad_accum_steps}, epochs={epochs}, "
        f"use_separate_ref_model={use_separate_ref_model}"
    )
    log_fn(f"MAX_PROMPT_LEN={MAX_PROMPT_LEN}, MAX_FULL_LEN={MAX_FULL_LEN}")
    log_fn(
        "val_KL_MC: "
        f"max_prompts={val_kl_mc_max_prompts}, samples_per_prompt={val_kl_mc_num_samples}, "
        f"max_new_tokens={val_kl_mc_max_new_tokens}, prompt_batch_size={val_kl_mc_prompt_batch_size} "
        "(max_prompts=0 disables MC-KL)"
    )
    log_fn(
        "val_response_entropy: "
        f"max_prompts={val_entropy_max_prompts}, samples_per_prompt={val_entropy_num_samples}, "
        f"max_new_tokens={val_entropy_max_new_tokens}, prompt_batch_size={val_entropy_prompt_batch_size}, "
        f"forward_chunk_size={val_entropy_forward_chunk_size} "
        "(max_prompts=0 disables val entropy)"
    )

    cap_rows = None
    cap_ref_holder: List[Optional[List[str]]] = [None]
    if capability_eval_dir:
        try:
            ep = Path(capability_eval_dir).expanduser().resolve()
            if ep.is_dir():
                loaded = load_eval_rows(ep)
                if capability_eval_limit is not None and capability_eval_limit > 0:
                    loaded = loaded[: int(capability_eval_limit)]
                if loaded:
                    cap_rows = loaded
                    log_fn(
                        f"Capability retention: {len(cap_rows)} примеров из {ep} "
                        f"(ref кэшируется после первой генерации)."
                    )
        except Exception as e:
            log_fn(f"Capability retention: ошибка загрузки eval: {e}")
            cap_rows = None

    # Начальная валидация — те же метрики, что в hard_dpo_steer
    val_loader_init = DataLoader(
        val_ds,
        batch_size=training_args.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn_hard,
    )
    policy_model.eval()
    if ref_model is not None:
        ref_model.eval()
    init_dpo_sum = 0.0
    init_kl_sum = 0.0
    init_n = 0
    with torch.no_grad():
        for batch in tqdm(val_loader_init, desc="init DPO loss", leave=False):
            loss, kl_b = hard_dpo_loss(batch, tokenizer, policy_model, ref_model, device, beta=beta)
            n = len(batch["prompt"])
            init_dpo_sum += loss.item() * n
            init_kl_sum += kl_b * n
            init_n += n
    init_dpo = init_dpo_sum / max(1, init_n)
    init_kl = init_kl_sum / max(1, init_n)
    init_nll = eval_pairwise_nll(
        val_loader_init, tokenizer, policy_model, device,
        beta=1.0, max_prompt_len=MAX_PROMPT_LEN, max_full_len=MAX_FULL_LEN,
        desc="init pairwise NLL",
    )
    init_acc = eval_pairwise_accuracy(
        val_loader_init, tokenizer, policy_model, device,
        max_prompt_len=MAX_PROMPT_LEN, max_full_len=MAX_FULL_LEN,
        desc="init pairwise acc",
    )
    log_fn("")
    log_fn("=== Initial (before training), epoch 0 ===")
    log_fn(f"validation DPO loss   : {init_dpo:.4f}")
    log_fn(f"validation logp_gap_mean : {init_kl:.4f}")
    log_fn(f"validation pair NLL   : {init_nll:.4f}")
    log_fn(f"validation pair acc   : {100 * init_acc:.2f}%")
    _classic_capability_retention(
        cap_rows,
        cap_ref_holder,
        tokenizer,
        ref_model,
        policy_model,
        device,
        output_dir,
        "init",
        log_fn,
        capability_eval_max_new_tokens,
        capability_eval_batch_size,
        capability_eval_max_prompt_tokens,
    )

    validation_callback = DPOValidationMetricsCallback(
        val_ds=val_ds,
        tokenizer=tokenizer,
        ref_model=ref_model,
        device=device,
        log_fn=log_fn,
        beta=beta,
        eval_batch_size=training_args.per_device_eval_batch_size,
        cap_rows=cap_rows,
        cap_ref_holder=cap_ref_holder,
        cap_output_dir=output_dir,
        cap_max_new_tokens=capability_eval_max_new_tokens,
        cap_batch_size=capability_eval_batch_size,
        cap_max_prompt_tokens=capability_eval_max_prompt_tokens,
        val_kl_mc_max_prompts=val_kl_mc_max_prompts,
        val_kl_mc_num_samples=val_kl_mc_num_samples,
        val_kl_mc_max_new_tokens=val_kl_mc_max_new_tokens,
        val_kl_mc_prompt_batch_size=val_kl_mc_prompt_batch_size,
        val_entropy_max_prompts=val_entropy_max_prompts,
        val_entropy_num_samples=val_entropy_num_samples,
        val_entropy_max_new_tokens=val_entropy_max_new_tokens,
        val_entropy_prompt_batch_size=val_entropy_prompt_batch_size,
        val_entropy_forward_chunk_size=val_entropy_forward_chunk_size,
        use_chat_template=use_chat_template,
    )

    trainer_ref_model = ref_model if use_separate_ref_model else None
    trainer = DPOTrainer(
        model=policy_model,
        ref_model=trainer_ref_model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        callbacks=[validation_callback],
    )

    run_status = "SUCCESS"
    try:
        trainer.train()

        best_ckpt = os.path.join(output_dir, "best")
        os.makedirs(best_ckpt, exist_ok=True)
        trainer.save_model(best_ckpt)
        tokenizer.save_pretrained(best_ckpt)
        log_fn(f"Checkpoint сохранён: {best_ckpt}")
    except Exception as exc:
        run_status = f"FAILED: {exc}"
        raise
    finally:
        run_finished_at = datetime.now()
        run_duration_sec = time.perf_counter() - run_started_perf
        log_fn("")
        log_fn(f"Run finished at: {run_finished_at.strftime('%Y-%m-%d %H:%M:%S')}")
        log_fn(f"Run status: {run_status}")
        log_fn(f"Run duration: {run_duration_sec:.1f}s")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Классический DPO (TRL) на HelpSteer3 / UltraFeedback")
    parser.add_argument(
        "--resume", "-r",
        type=str,
        default=None,
        help="Путь к чекпоинту для продолжения обучения (например checkpoints/classic_dpo/best)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed для воспроизводимости (по умолчанию 42)")
    parser.add_argument("--output-dir", "-o", type=str, default="checkpoints/classic_dpo", help="Папка для чекпоинтов и train.log")
    parser.add_argument("--dataset", "-d", type=str, default="helpsteer3", choices=list(DATASET_CHOICES), help="Датасет: helpsteer3 или ultrafeedback_binarized")
    parser.add_argument("--base-model", type=str, choices=list(BASE_MODEL_CHOICES.keys()), default="3b", help="Базовая модель: 3b или 7b.")
    parser.add_argument("--batch-size", "-b", type=int, default=8, help="Размер батча для train и validation (по умолчанию: 8).")
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help="Шаги накопления градиента (effective_batch = batch_size * grad_accum_steps).",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Количество эпох обучения (по умолчанию: 3).")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate (по умолчанию: 2e-5).")
    parser.add_argument("--beta", type=float, default=0.2, help="Параметр beta для DPO loss (по умолчанию: 0.2).")
    parser.add_argument(
        "--use-separate-ref-model",
        action="store_true",
        help=(
            "Загрузить отдельную reference-модель (старое поведение; требует больше VRAM). "
            "По умолчанию ref в DPOTrainer берётся без второй копии модели."
        ),
    )
    parser.add_argument(
        "--capability-eval-dir",
        type=str,
        default=None,
        help="Каталог eval_datasets: на каждой валидации лог capability retention.",
    )
    parser.add_argument(
        "--use-chat-template",
        action="store_true",
        default=USE_CHAT_TEMPLATE,
        help=f"Считать log p через apply_chat_template (по умолчанию: {USE_CHAT_TEMPLATE}).",
    )
    parser.add_argument(
        "--val-kl-mc-max-prompts",
        type=int,
        default=DEFAULT_VAL_KL_MC_MAX_PROMPTS,
        help=(
            "MC-оценка forward KL(pi||ref) на val: первые N промптов; "
            f"0 — отключить (по умолчанию {DEFAULT_VAL_KL_MC_MAX_PROMPTS})."
        ),
    )
    parser.add_argument(
        "--val-kl-mc-num-samples",
        type=int,
        default=VAL_KL_MC_NUM_SAMPLES,
        help=f"Число генераций на промпт для MC-KL (по умолчанию {VAL_KL_MC_NUM_SAMPLES}).",
    )
    parser.add_argument(
        "--val-kl-mc-max-new-tokens",
        type=int,
        default=128,
        help="Максимум новых токенов в одной генерации для MC-KL (по умолчанию 128).",
    )
    parser.add_argument(
        "--val-kl-mc-prompt-batch-size",
        type=int,
        default=VAL_KL_MC_PROMPT_BATCH_SIZE,
        help=(
            "Батч промптов при генерации/оценке MC-KL "
            f"(по умолчанию {VAL_KL_MC_PROMPT_BATCH_SIZE})."
        ),
    )
    args = parser.parse_args()
    main(
        resume_from=args.resume,
        seed=args.seed,
        output_dir=args.output_dir,
        dataset=args.dataset,
        base_model=args.base_model,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        epochs=args.epochs,
        lr=args.lr,
        beta=args.beta,
        use_separate_ref_model=args.use_separate_ref_model,
        capability_eval_dir=args.capability_eval_dir,
        use_chat_template=args.use_chat_template,
        val_kl_mc_max_prompts=args.val_kl_mc_max_prompts,
        val_kl_mc_num_samples=args.val_kl_mc_num_samples,
        val_kl_mc_max_new_tokens=args.val_kl_mc_max_new_tokens,
        val_kl_mc_prompt_batch_size=args.val_kl_mc_prompt_batch_size,
    )
