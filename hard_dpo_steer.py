# -*- coding: utf-8 -*-
import os
import sys
from typing import Optional

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

from config.base_config import (
    CAPABILITY_EVAL_BATCH_SIZE,
    CAPABILITY_EVAL_LIMIT,
    CAPABILITY_EVAL_MAX_NEW_TOKENS,
    CAPABILITY_EVAL_MAX_PROMPT_TOKENS,
    USE_CHAT_TEMPLATE,
)
from utils.config import BASE_MODEL_CHOICES, DPO_STEER_HARD_DATASET_CHOICES as DATASET_CHOICES
from utils.seed import set_seed
from utils.datasets import (
    build_dpo_datasets,
    build_dpo_datasets_hh_rlhf,
    build_dpo_datasets_ultrafeedback,
)
from utils.models import load_models_and_tokenizer
from utils.training import DEFAULT_VAL_KL_MC_MAX_PROMPTS, train_dpo


# ======================
# main
# ======================


def main(
    resume_from: Optional[str] = None,
    seed: int = 42,
    output_dir: str = "checkpoints/hard_dpo_steer",
    dataset: str = "helpsteer3",
    base_model: str = "3b",
    batch_size: int = 8,
    lr: float = 2e-5,
    beta: float = 0.2,
    epochs: int = 8,
    lambda_min: float = 1.0,
    use_chat_template: bool = USE_CHAT_TEMPLATE,
    capability_eval_dir: Optional[str] = None,
    capability_eval_limit: Optional[int] = CAPABILITY_EVAL_LIMIT,
    capability_eval_max_new_tokens: int = CAPABILITY_EVAL_MAX_NEW_TOKENS,
    capability_eval_batch_size: int = CAPABILITY_EVAL_BATCH_SIZE,
    capability_eval_max_prompt_tokens: int = CAPABILITY_EVAL_MAX_PROMPT_TOKENS,
    capability_ref_cache_path: Optional[str] = None,
    val_kl_mc_max_prompts: int = DEFAULT_VAL_KL_MC_MAX_PROMPTS,
    resume_start_epoch_1based: int = 1,
):
    """
    resume_from: путь к чекпоинту (например "checkpoints/hard_dpo_steer/best").
    Если задан, policy и tokenizer загружаются из чекпоинта, обучение продолжается с этих весов.
    seed: для воспроизводимости; одинаковый seed в hard_dpo_steer и soft_steer даёт совпадающие начальные метрики на val.
    output_dir: папка для чекпоинтов и train.log.
    dataset: "helpsteer3" | "ultrafeedback_binarized" | "hh_rlhf" (PKU processed HH-RLHF).
    base_model: "3b" | "7b" — Qwen2.5-*B-Instruct; "4b" — Qwen3-4B-Instruct-2507.
    batch_size: размер батча для train и validation.
    lambda_min: для режима hard не используется (оставлено для единообразия CLI с soft_dpo_steer).
    use_chat_template: log p через apply_chat_template (дефолт в config.base_config).
    capability_eval_dir: если задан — на каждой валидации eval_datasets (gold), см. train_dpo.
    resume_start_epoch_1based: см. utils.training.train_dpo (--epochs = полный план, --start-epoch).
    """
    if dataset not in DATASET_CHOICES:
        raise ValueError(f"dataset должен быть один из {DATASET_CHOICES}, получено: {dataset!r}")
    set_seed(seed)
    if dataset == "helpsteer3":
        print("Загружаю HelpSteer3-Preference...")
        train_ds, val_ds = build_dpo_datasets()
    elif dataset == "ultrafeedback_binarized":
        print("Загружаю UltraFeedback Binarized...")
        train_ds, val_ds = build_dpo_datasets_ultrafeedback()
    else:
        print("Загружаю PKU processed HH-RLHF...")
        train_ds, val_ds = build_dpo_datasets_hh_rlhf()
    model_name = BASE_MODEL_CHOICES[base_model]
    print(f"Model: {model_name}, Dataset: {dataset}, train size: {len(train_ds)}, val size: {len(val_ds)}")
    if resume_from:
        print(f"Загружаю модель из чекпоинта: {resume_from} (база {model_name})")
    else:
        print(f"Загружаю модель и токенайзер: {model_name} (LoRA)")
    tokenizer, policy_model, ref_model, device = load_models_and_tokenizer(
        model_name, use_lora=True, lora_r=16, lora_alpha=32, resume_from=resume_from
    )

    def log_fn(msg: str) -> None:
        # Лог-строки идут в stdout, чтобы не смешиваться с tqdm-прогрессами (stderr).
        # Это даёт чистое `>run.log` с только осмысленными строками.
        print(msg, flush=True, file=sys.stdout)

    print("Начинаю обучение DPO (hard)...")
    train_dpo(
        train_ds,
        val_ds,
        tokenizer,
        policy_model,
        ref_model,
        device,
        mode="hard",
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        beta=beta,
        output_dir=output_dir,
        dataset_name=dataset,
        model_name=model_name,
        lambda_min=lambda_min,
        seed=seed,
        use_chat_template=use_chat_template,
        log=log_fn,
        capability_eval_dir=capability_eval_dir,
        capability_eval_limit=capability_eval_limit,
        capability_eval_max_new_tokens=capability_eval_max_new_tokens,
        capability_eval_batch_size=capability_eval_batch_size,
        capability_eval_max_prompt_tokens=capability_eval_max_prompt_tokens,
        capability_ref_cache_path=capability_ref_cache_path,
        val_kl_mc_max_prompts=val_kl_mc_max_prompts,
        resume_start_epoch_1based=resume_start_epoch_1based,
        resume_checkpoint_dir=resume_from,
    )


def _lambda_min_type(x: str) -> float:
    v = float(x)
    if not 0.0 <= v <= 1.0:
        raise ValueError(f"--lambda-min must be in [0, 1], got {v}")
    return v


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Hard DPO: HelpSteer3, UltraFeedback Binarized или HH-RLHF (PKU-Alignment/processed-hh-rlhf)."
    )
    parser.add_argument(
        "--resume", "-r",
        type=str,
        default=None,
        help="Путь к чекпоинту для продолжения обучения (например checkpoints/hard_dpo_steer/best)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed для воспроизводимости (по умолчанию 42)")
    parser.add_argument("--output-dir", "-o", type=str, default="checkpoints/hard_dpo_steer", help="Папка для чекпоинтов и train.log (для разных запусков задавайте разные папки)")
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="helpsteer3",
        choices=list(DATASET_CHOICES),
        help="Датасет: helpsteer3, ultrafeedback_binarized или hh_rlhf (PKU-Alignment/processed-hh-rlhf).",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        choices=list(BASE_MODEL_CHOICES.keys()),
        default="3b",
        help="Базовая модель: 3b/7b — Qwen2.5-Instruct; 4b — Qwen3-4B-Instruct-2507. По умолчанию: 3b.",
    )
    parser.add_argument("--batch-size", "-b", type=int, default=8, help="Размер батча для train и validation (по умолчанию: 8).")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate (по умолчанию: 2e-5).")
    parser.add_argument("--beta", type=float, default=0.2, help="Параметр beta для DPO loss (по умолчанию: 0.2).")
    parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=8,
        help="Всего эпох в плане (LR по шкале 1..epochs). При --start-epoch>1 обучаются эпохи start..epochs.",
    )
    parser.add_argument(
        "--start-epoch",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Первая эпоха запуска (1-based), <= --epochs; --resume после эпохи N-1. "
            "Эпох в этом запуске: epochs - start + 1. "
            "При --resume и N>1 в начало train.log в --output-dir переносится история "
            "старого train.log (рядом с чекпоинтом) до эпохи N, если граница найдена в логе."
        ),
    )
    parser.add_argument(
        "--lambda-min",
        type=_lambda_min_type,
        default=1.0,
        help="Для hard не влияет; единый флаг с soft_dpo_steer [0, 1] (по умолчанию: 1.0).",
    )
    parser.add_argument(
        "--capability-eval-dir",
        type=str,
        default=None,
        help="Каталог eval_datasets (knowledge/*.jsonl, reasoning/*.jsonl): на каждой валидации лог retention.",
    )
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
    main(
        resume_from=args.resume,
        seed=args.seed,
        output_dir=args.output_dir,
        dataset=args.dataset,
        base_model=args.base_model,
        batch_size=args.batch_size,
        lr=args.lr,
        beta=args.beta,
        epochs=args.epochs,
        lambda_min=args.lambda_min,
        capability_eval_dir=args.capability_eval_dir,
        capability_ref_cache_path=args.capability_ref_cache_path,
        val_kl_mc_max_prompts=args.val_kl_mc_max_prompts,
        resume_start_epoch_1based=args.start_epoch,
    )
