# -*- coding: utf-8 -*-
import sys
from typing import Optional

import torch

from utils.config import BASE_MODEL_CHOICES, DPO_STEER_HARD_DATASET_CHOICES as DATASET_CHOICES
from utils.seed import set_seed
from utils.datasets import (
    build_dpo_datasets,
    build_dpo_datasets_hh_rlhf,
    build_dpo_datasets_ultrafeedback,
)
from utils.models import load_models_and_tokenizer
from utils.training import train_dpo


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
    use_chat_template: Optional[bool] = None,
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
    use_chat_template: если None — для hh_rlhf True (PKU HH), иначе False; иначе явное значение для get_logps.
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

    if use_chat_template is None:
        use_chat_template = dataset == "hh_rlhf"

    def log_fn(msg: str) -> None:
        print(msg, flush=True, file=sys.stderr)

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
    parser.add_argument("--epochs", "-e", type=int, default=8, help="Количество эпох обучения (по умолчанию: 8).")
    parser.add_argument(
        "--lambda-min",
        type=_lambda_min_type,
        default=1.0,
        help="Для hard не влияет; единый флаг с soft_dpo_steer [0, 1] (по умолчанию: 1.0).",
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
        help="Считать log p как plain prompt\\nresponse (отключить chat template, в т.ч. для hh_rlhf).",
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
        output_dir=args.output_dir,
        dataset=args.dataset,
        base_model=args.base_model,
        batch_size=args.batch_size,
        lr=args.lr,
        beta=args.beta,
        epochs=args.epochs,
        lambda_min=args.lambda_min,
        use_chat_template=use_chat_template,
    )
