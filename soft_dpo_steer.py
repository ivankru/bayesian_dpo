#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
from typing import Optional

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

from utils.config import BASE_MODEL_CHOICES, DPO_STEER_SOFT_DATASET_CHOICES as DATASET_CHOICES
from utils.seed import set_seed
from utils.datasets import (
    build_helpsteer3_soft_datasets,
    build_hh_rlhf_soft_steer_datasets,
    build_openbmb_soft_datasets,
    build_ultrafeedback_binarized_soft_datasets,
    build_ultrafeedback_score_soft_datasets,
)
from utils.models import load_models_and_tokenizer
from utils.training import DEFAULT_VAL_KL_MC_MAX_PROMPTS, train_dpo


# ======================
# main
# ======================


def main(
    resume_from: Optional[str] = None,
    seed: int = 42,
    alpha: float = 1.0,
    label_noise_prob: float = 0.0,
    use_bayes: bool = False,
    output_dir: str = "checkpoints/soft_dpo_steer",
    base_model: str = "3b",
    dataset: str = "helpsteer3",
    batch_size: int = 5,
    lr: float = 2e-5,
    beta: float = 0.2,
    epochs: int = 8,
    lambda_min: float = 1.0,
    lambda_schedule: str = "linear",
    use_chat_template: Optional[bool] = None,
    capability_eval_dir: Optional[str] = None,
    capability_eval_limit: Optional[int] = None,
    capability_eval_max_new_tokens: int = 256,
    capability_eval_batch_size: int = 2,
    capability_eval_max_prompt_tokens: int = 2048,
    capability_ref_cache_path: Optional[str] = None,
    val_kl_mc_max_prompts: int = DEFAULT_VAL_KL_MC_MAX_PROMPTS,
):
    """
    Soft-train + hard-validation.
    seed: для воспроизводимости; тот же seed, что в hard_dpo_steer (по умолчанию 42), даёт совпадающие начальные метрики на val.
    alpha: параметр бета-приора для p_bayes (по умолчанию 1.0).
    use_bayes: если True, в loss используется p_bayes, иначе p (по умолчанию).
    base_model: "3b" | "7b" — Qwen2.5-*B-Instruct; "4b" — Qwen3-4B-Instruct-2507.
    dataset: helpsteer3 | ultrafeedback_binarized (бинарные p) | ultrafeedback_soft (p из скоров) | openbmb | hh_rlhf.
    batch_size: размер батча для train и validation.
    lambda_min: нижняя граница lambda_label по эпохам (смешивание меток с p_pred); 1.0 — как раньше, без смешивания.
    use_chat_template: если None — False, кроме hh_rlhf (True, как в hard_dpo_steer); иначе явное значение.
    """
    if dataset not in DATASET_CHOICES:
        raise ValueError(f"dataset должен быть один из {DATASET_CHOICES}, получено: {dataset!r}")
    set_seed(seed)
    model_name = BASE_MODEL_CHOICES[base_model]
    if dataset == "helpsteer3":
        print("Загружаю HelpSteer3-Preference...")
        train_soft_ds, val_hard_ds, hard_train_size = build_helpsteer3_soft_datasets(
            alpha=alpha,
            label_noise_prob=label_noise_prob,
            seed=seed,
        )
    elif dataset == "ultrafeedback_binarized":
        print("Загружаю UltraFeedback Binarized (бинарные метки chosen>rejected)...")
        train_soft_ds, val_hard_ds, hard_train_size = build_ultrafeedback_binarized_soft_datasets(
            alpha=alpha,
            label_noise_prob=label_noise_prob,
            seed=seed,
        )
    elif dataset == "ultrafeedback_soft":
        print("Загружаю UltraFeedback (мягкие метки по score_chosen/score_rejected)...")
        train_soft_ds, val_hard_ds, hard_train_size = build_ultrafeedback_score_soft_datasets(
            alpha=alpha,
            label_noise_prob=label_noise_prob,
            seed=seed,
        )
    elif dataset == "hh_rlhf":
        print("Загружаю PKU processed HH-RLHF (soft train, hard val)...")
        train_soft_ds, val_hard_ds, hard_train_size = build_hh_rlhf_soft_steer_datasets(alpha=alpha)
    else:  # openbmb
        print("Загружаю openbmb/UltraFeedback (soft) + val ultrafeedback_binarized...")
        train_soft_ds, val_hard_ds, hard_train_size = build_openbmb_soft_datasets(alpha=alpha)
    prob_type = "p_bayes" if use_bayes else "p"
    print(f"Model: {model_name}, Dataset: {dataset}")
    print(f"Train soft size: {len(train_soft_ds)}, val hard size: {len(val_hard_ds)}, hard train size: {hard_train_size}, alpha={alpha}, target_prob={prob_type}")

    if resume_from:
        print(f"Загружаю модель из чекпоинта: {resume_from} (база {model_name})")
    else:
        print(f"Загружаю модель и токенайзер: {model_name} (LoRA)")
    tokenizer, policy_model, ref_model, device = load_models_and_tokenizer(
        model_name, use_lora=True, lora_r=16, lora_alpha=32, resume_from=resume_from
    )

    if use_chat_template is None:
        use_chat_template = dataset == "hh_rlhf"

    def log_fn(msg: str) -> None:
        print(msg, flush=True, file=sys.stderr)

    mode = "bayes" if use_bayes else "soft"
    num_steps_override = epochs * ((hard_train_size + batch_size - 1) // batch_size) if hard_train_size else None
    print(f"Начинаю обучение {mode.upper()}-DPO (train {mode}, validation hard)...")
    train_dpo(
        train_soft_ds,
        val_hard_ds,
        tokenizer,
        policy_model,
        ref_model,
        device,
        mode=mode,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        beta=beta,
        alpha=alpha,
        output_dir=output_dir,
        num_training_steps_override=num_steps_override,
        dataset_name=dataset,
        model_name=model_name,
        lambda_min=lambda_min,
        lambda_schedule=lambda_schedule,
        seed=seed,
        label_noise_prob=label_noise_prob,
        use_chat_template=use_chat_template,
        log=log_fn,
        capability_eval_dir=capability_eval_dir,
        capability_eval_limit=capability_eval_limit,
        capability_eval_max_new_tokens=capability_eval_max_new_tokens,
        capability_eval_batch_size=capability_eval_batch_size,
        capability_eval_max_prompt_tokens=capability_eval_max_prompt_tokens,
        capability_ref_cache_path=capability_ref_cache_path,
        val_kl_mc_max_prompts=val_kl_mc_max_prompts,
    )


def _lambda_min_type(x: str) -> float:
    v = float(x)
    if not 0.0 <= v <= 1.0:
        raise ValueError(f"--lambda-min must be in [0, 1], got {v}")
    return v


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Soft-DPO (train soft, validation hard): HelpSteer3; UltraFeedback бинарный или score-soft; "
            "openbmb; HH-RLHF (PKU)."
        )
    )
    parser.add_argument(
        "--resume", "-r",
        type=str,
        default=None,
        help="Путь к чекпоинту для продолжения обучения (например checkpoints/soft_dpo_steer/best)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed для воспроизводимости (должен совпадать с hard_dpo_steer для сравнения)")
    parser.add_argument(
        "--label-noise-prob",
        type=float,
        default=0.0,
        help=(
            "Train label noise: для бинарных датасетов (HelpSteer3, ultrafeedback_binarized) — "
            "переворот p 0↔1 с заданной вероятностью; для ultrafeedback_soft — замена p на 1−p "
            "(и пересчёт p_bayes) с той же вероятностью."
        ),
    )
    parser.add_argument("--alpha", type=float, default=0.2, help="Параметр бета-приора для p_bayes; имеет смысл только при --use-bayes (по умолчанию 1.0)")
    parser.add_argument("--use-bayes", action="store_true", help="Использовать p_bayes вместо p в качестве целевой вероятности (по умолчанию: p)")
    parser.add_argument("--output-dir", "-o", type=str, default="checkpoints/soft_dpo_steer", help="Папка для чекпоинтов и train.log (для разных запусков задавайте разные папки)")
    parser.add_argument(
        "--base-model",
        type=str,
        choices=list(BASE_MODEL_CHOICES.keys()),
        default="3b",
        help="Базовая модель: 3b/7b — Qwen2.5-Instruct; 4b — Qwen3-4B-Instruct-2507. По умолчанию: 3b.",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="helpsteer3",
        choices=list(DATASET_CHOICES),
        help=(
            "Датасет: helpsteer3; ultrafeedback_binarized (жёсткое chosen>rejected, p∈{0,1}); "
            "ultrafeedback_soft (p=sigmoid(Δscore)); openbmb (soft); hh_rlhf (PKU processed)."
        ),
    )
    parser.add_argument("--batch-size", "-b", type=int, default=8, help="Размер батча для train и validation (по умолчанию: 5).")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate (по умолчанию: 2e-5).")
    parser.add_argument("--beta", type=float, default=0.3, help="Параметр beta для DPO loss (по умолчанию: 0.2).")
    parser.add_argument("--epochs", "-e", type=int, default=8, help="Количество эпох обучения (по умолчанию: 8).")
    parser.add_argument(
        "--lambda-min",
        type=_lambda_min_type,
        default=1.0,
        help="Минимум lambda_label по эпохам [0, 1]; 1.0 = только метки из датасета (по умолчанию: 1.0).",
    )
    parser.add_argument(
        "--lambda-schedule",
        type=str,
        default="linear",
        choices=["linear", "cosine"],
        help="Schedule for lambda_label over epochs (linear or cosine).",
    )
    chat_group = parser.add_mutually_exclusive_group()
    chat_group.add_argument(
        "--use-chat-template",
        action="store_true",
        help="Считать log p через apply_chat_template (Qwen-Instruct). По умолчанию включено только для hh_rlhf.",
    )
    chat_group.add_argument(
        "--no-use-chat-template",
        action="store_true",
        help="Считать log p как plain prompt\\nresponse (отключить chat template).",
    )
    parser.add_argument(
        "--capability-eval-dir",
        type=str,
        default=None,
        help="Каталог eval_datasets: на каждой валидации лог capability retention (gold).",
    )
    parser.add_argument("--capability-eval-limit", type=int, default=None)
    parser.add_argument("--capability-eval-max-new-tokens", type=int, default=256)
    parser.add_argument("--capability-eval-batch-size", type=int, default=2)
    parser.add_argument("--capability-eval-max-prompt-tokens", type=int, default=2048)
    parser.add_argument(
        "--capability-ref-cache-path",
        type=str,
        default=None,
        help="Явный путь к JSON-кэшу ref ответов для capability retention (опционально).",
    )
    parser.add_argument(
        "--val-kl-mc-max-prompts",
        type=int,
        default=DEFAULT_VAL_KL_MC_MAX_PROMPTS,
        help=(
            "MC-оценка forward KL(π‖ref) на val: первые N промптов; 0 — отключить "
            f"(по умолчанию {DEFAULT_VAL_KL_MC_MAX_PROMPTS})."
        ),
    )
    args = parser.parse_args()
    use_chat_template: Optional[bool] = None
    if args.use_chat_template:
        use_chat_template = True
    elif args.no_use_chat_template:
        use_chat_template = False
    main(
        resume_from=args.resume,
        seed=args.seed,
        alpha=args.alpha,
        label_noise_prob=args.label_noise_prob,
        use_bayes=args.use_bayes,
        output_dir=args.output_dir,
        base_model=args.base_model,
        dataset=args.dataset,
        batch_size=args.batch_size,
        lr=args.lr,
        beta=args.beta,
        epochs=args.epochs,
        lambda_min=args.lambda_min,
        lambda_schedule=args.lambda_schedule,
        use_chat_template=use_chat_template,
        capability_eval_dir=args.capability_eval_dir,
        capability_eval_limit=args.capability_eval_limit,
        capability_eval_max_new_tokens=args.capability_eval_max_new_tokens,
        capability_eval_batch_size=args.capability_eval_batch_size,
        capability_eval_max_prompt_tokens=args.capability_eval_max_prompt_tokens,
        capability_ref_cache_path=args.capability_ref_cache_path,
        val_kl_mc_max_prompts=args.val_kl_mc_max_prompts,
    )
