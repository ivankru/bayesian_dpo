# -*- coding: utf-8 -*-
import random
import sys
from typing import Optional

import numpy as np
import torch

from utils.config import BASE_MODEL_CHOICES
from utils.datasets import (
    build_dpo_datasets,
    build_dpo_datasets_ultrafeedback,
)
from utils.models import load_models_and_tokenizer
from utils.training import train_dpo


# ======================
# main
# ======================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


DATASET_CHOICES = ("helpsteer3", "ultrafeedback_binarized")


def main(resume_from: Optional[str] = None, seed: int = 42, output_dir: str = "checkpoints/hard_dpo_steer", dataset: str = "helpsteer3", base_model: str = "3b", batch_size: int = 8, lr: float = 2e-5, beta: float = 0.2):
    """
    resume_from: путь к чекпоинту (например "checkpoints/hard_dpo_steer/best").
    Если задан, policy и tokenizer загружаются из чекпоинта, обучение продолжается с этих весов.
    seed: для воспроизводимости; одинаковый seed в hard_dpo_steer и soft_steer даёт совпадающие начальные метрики на val.
    output_dir: папка для чекпоинтов и train.log.
    dataset: "helpsteer3" | "ultrafeedback_binarized".
    base_model: "3b" (Qwen2.5-3B) или "7b" (Qwen2.5-7B).
    batch_size: размер батча для train и validation.
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
    tokenizer, policy_model, ref_model, device = load_models_and_tokenizer(
        model_name, use_lora=True, lora_r=16, lora_alpha=32, resume_from=resume_from
    )

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
        epochs=8,
        batch_size=batch_size,
        lr=lr,
        beta=beta,
        output_dir=output_dir,
        dataset_name=dataset,
        model_name=model_name,
        log=log_fn,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DPO на HelpSteer3")
    parser.add_argument(
        "--resume", "-r",
        type=str,
        default=None,
        help="Путь к чекпоинту для продолжения обучения (например checkpoints/hard_dpo_steer/best)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed для воспроизводимости (по умолчанию 42)")
    parser.add_argument("--output-dir", "-o", type=str, default="checkpoints/hard_dpo_steer", help="Папка для чекпоинтов и train.log (для разных запусков задавайте разные папки)")
    parser.add_argument("--dataset", "-d", type=str, default="helpsteer3", choices=list(DATASET_CHOICES), help="Датасет: helpsteer3 или ultrafeedback_binarized")
    parser.add_argument("--base-model", type=str, choices=list(BASE_MODEL_CHOICES.keys()), default="3b", help="Базовая модель: 3b (Qwen2.5-3B-Instruct) или 7b (Qwen2.5-7B-Instruct). По умолчанию: 3b.")
    parser.add_argument("--batch-size", "-b", type=int, default=8, help="Размер батча для train и validation (по умолчанию: 8).")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate (по умолчанию: 2e-5).")
    parser.add_argument("--beta", type=float, default=0.2, help="Параметр beta для DPO loss (по умолчанию: 0.2).")
    args = parser.parse_args()
    main(resume_from=args.resume, seed=args.seed, output_dir=args.output_dir, dataset=args.dataset, base_model=args.base_model, batch_size=args.batch_size, lr=args.lr, beta=args.beta)
