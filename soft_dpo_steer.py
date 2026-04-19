#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
from dataclasses import dataclass
from typing import Optional

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

from config.base_config import (
    CAPABILITY_EVAL_BATCH_SIZE,
    CAPABILITY_EVAL_LIMIT,
    CAPABILITY_EVAL_MAX_NEW_TOKENS,
    CAPABILITY_EVAL_MAX_PROMPT_TOKENS,
    P_PRED_TARGET_TEMPERATURE,
    USE_CHAT_TEMPLATE,
    VAL_ENTROPY_FORWARD_CHUNK_SIZE,
    VAL_ENTROPY_MAX_NEW_TOKENS,
    VAL_ENTROPY_MAX_PROMPTS,
    VAL_ENTROPY_NUM_SAMPLES,
    VAL_ENTROPY_PROMPT_BATCH_SIZE,
)
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
# Config + main
# ======================


@dataclass
class SoftDPOConfig:
    """Единый источник гиперпараметров soft/bayes-DPO прогона.

    Все значения, которые могут быть перекрыты из CLI, имеют дефолты. Часть полей
    (p_pred_target_temperature, val_entropy_*, capability_eval_*) в CLI не
    экспонируется — они тянут значения из config.base_config и меняются коммитом
    конфига, а не флагами запуска (ради сопоставимости экспериментов).
    """

    # --- базовые ---
    seed: int = 42
    base_model: str = "3b"
    dataset: str = "helpsteer3"
    output_dir: str = "checkpoints/soft_dpo_steer"
    resume_from: Optional[str] = None

    # --- DPO/soft-гиперпараметры ---
    alpha: float = 0.2
    label_noise_prob: float = 0.0
    use_bayes: bool = False
    batch_size: int = 8
    lr: float = 3e-5
    beta: float = 0.3
    epochs: int = 8

    # --- lambda-расписание ---
    lambda_min: float = 1.0
    lambda_schedule: str = "linear"
    lambda_full_epochs: int = 0
    p_pred_target_temperature: float = P_PRED_TARGET_TEMPERATURE
    use_chat_template: bool = USE_CHAT_TEMPLATE

    # --- capability retention (дефолты в config/base_config.py) ---
    capability_eval_dir: Optional[str] = None
    capability_eval_limit: Optional[int] = CAPABILITY_EVAL_LIMIT
    capability_eval_max_new_tokens: int = CAPABILITY_EVAL_MAX_NEW_TOKENS
    capability_eval_batch_size: int = CAPABILITY_EVAL_BATCH_SIZE
    capability_eval_max_prompt_tokens: int = CAPABILITY_EVAL_MAX_PROMPT_TOKENS
    capability_ref_cache_path: Optional[str] = None

    # --- val KL-MC ---
    val_kl_mc_max_prompts: int = DEFAULT_VAL_KL_MC_MAX_PROMPTS

    # --- val response entropy (дефолты в config/base_config.py) ---
    val_entropy_max_prompts: int = VAL_ENTROPY_MAX_PROMPTS
    val_entropy_num_samples: int = VAL_ENTROPY_NUM_SAMPLES
    val_entropy_max_new_tokens: int = VAL_ENTROPY_MAX_NEW_TOKENS
    val_entropy_prompt_batch_size: int = VAL_ENTROPY_PROMPT_BATCH_SIZE
    val_entropy_forward_chunk_size: int = VAL_ENTROPY_FORWARD_CHUNK_SIZE

    # --- resume ---
    resume_start_epoch_1based: int = 1


def main(cfg: SoftDPOConfig) -> None:
    """Soft-train + hard-validation. Все гиперпараметры в SoftDPOConfig.

    seed: для воспроизводимости; тот же seed, что в hard_dpo_steer (по умолчанию 42), даёт совпадающие начальные метрики на val.
    alpha: параметр бета-приора для p_bayes; значение 0.2 — слабый приор (α ≈ 0.2, 2α = 0.4 «псевдо-наблюдения»).
    use_bayes: если True, в loss используется p_bayes, иначе p (по умолчанию).
    base_model: "3b" | "7b" — Qwen2.5-*B-Instruct; "4b" — Qwen3-4B-Instruct-2507.
    dataset: helpsteer3 | ultrafeedback_binarized (бинарные p) | ultrafeedback_soft (p из скоров) | openbmb | hh_rlhf.
    lambda_full_epochs: k (1-based): эпохи 1..k только метки; в конце эпохи k — p_pred_teacher; с k+1 λ<1;
        при λ<1 в p_pred всегда 0.5·учитель + 0.5·σ((beta*diff)/T). 0 — старое поведение.
    resume_start_epoch_1based: см. utils.training.train_dpo (--epochs = полный план, --start-epoch = с какой эпохи).
    """
    if cfg.dataset not in DATASET_CHOICES:
        raise ValueError(
            f"dataset должен быть один из {DATASET_CHOICES}, получено: {cfg.dataset!r}"
        )
    set_seed(cfg.seed)
    model_name = BASE_MODEL_CHOICES[cfg.base_model]
    if cfg.dataset == "helpsteer3":
        print("Загружаю HelpSteer3-Preference...")
        train_soft_ds, val_hard_ds, hard_train_size = build_helpsteer3_soft_datasets(
            alpha=cfg.alpha,
            label_noise_prob=cfg.label_noise_prob,
            seed=cfg.seed,
        )
    elif cfg.dataset == "ultrafeedback_binarized":
        print("Загружаю UltraFeedback Binarized (бинарные метки chosen>rejected)...")
        train_soft_ds, val_hard_ds, hard_train_size = build_ultrafeedback_binarized_soft_datasets(
            alpha=cfg.alpha,
            label_noise_prob=cfg.label_noise_prob,
            seed=cfg.seed,
        )
    elif cfg.dataset == "ultrafeedback_soft":
        print("Загружаю UltraFeedback (мягкие метки по score_chosen/score_rejected)...")
        train_soft_ds, val_hard_ds, hard_train_size = build_ultrafeedback_score_soft_datasets(
            alpha=cfg.alpha,
            label_noise_prob=cfg.label_noise_prob,
            seed=cfg.seed,
        )
    elif cfg.dataset == "hh_rlhf":
        print("Загружаю PKU processed HH-RLHF (soft train, hard val)...")
        train_soft_ds, val_hard_ds, hard_train_size = build_hh_rlhf_soft_steer_datasets(alpha=cfg.alpha)
    else:  # openbmb
        print("Загружаю openbmb/UltraFeedback (soft) + val ultrafeedback_binarized...")
        train_soft_ds, val_hard_ds, hard_train_size = build_openbmb_soft_datasets(alpha=cfg.alpha)
    prob_type = "p_bayes" if cfg.use_bayes else "p"
    print(f"Model: {model_name}, Dataset: {cfg.dataset}")
    print(
        f"Train soft size: {len(train_soft_ds)}, val hard size: {len(val_hard_ds)}, "
        f"hard train size: {hard_train_size}, alpha={cfg.alpha}, target_prob={prob_type}"
    )

    if cfg.resume_from:
        print(f"Загружаю модель из чекпоинта: {cfg.resume_from} (база {model_name})")
    else:
        print(f"Загружаю модель и токенайзер: {model_name} (LoRA)")
    tokenizer, policy_model, ref_model, device = load_models_and_tokenizer(
        model_name, use_lora=True, lora_r=16, lora_alpha=32, resume_from=cfg.resume_from
    )

    def log_fn(msg: str) -> None:
        # Лог-строки идут в stdout, чтобы не смешиваться с tqdm-прогрессами (stderr).
        # Это даёт чистое `>run.log` с только осмысленными строками.
        print(msg, flush=True, file=sys.stdout)

    mode = "bayes" if cfg.use_bayes else "soft"
    num_steps_override = (
        cfg.epochs * ((hard_train_size + cfg.batch_size - 1) // cfg.batch_size)
        if hard_train_size
        else None
    )
    print(f"Начинаю обучение {mode.upper()}-DPO (train {mode}, validation hard)...")
    train_dpo(
        train_soft_ds,
        val_hard_ds,
        tokenizer,
        policy_model,
        ref_model,
        device,
        mode=mode,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        beta=cfg.beta,
        alpha=cfg.alpha,
        output_dir=cfg.output_dir,
        num_training_steps_override=num_steps_override,
        dataset_name=cfg.dataset,
        model_name=model_name,
        lambda_min=cfg.lambda_min,
        lambda_schedule=cfg.lambda_schedule,
        lambda_full_epochs=cfg.lambda_full_epochs,
        p_pred_target_temperature=cfg.p_pred_target_temperature,
        seed=cfg.seed,
        label_noise_prob=cfg.label_noise_prob,
        use_chat_template=cfg.use_chat_template,
        log=log_fn,
        capability_eval_dir=cfg.capability_eval_dir,
        capability_eval_limit=cfg.capability_eval_limit,
        capability_eval_max_new_tokens=cfg.capability_eval_max_new_tokens,
        capability_eval_batch_size=cfg.capability_eval_batch_size,
        capability_eval_max_prompt_tokens=cfg.capability_eval_max_prompt_tokens,
        capability_ref_cache_path=cfg.capability_ref_cache_path,
        val_kl_mc_max_prompts=cfg.val_kl_mc_max_prompts,
        val_entropy_max_prompts=cfg.val_entropy_max_prompts,
        val_entropy_num_samples=cfg.val_entropy_num_samples,
        val_entropy_max_new_tokens=cfg.val_entropy_max_new_tokens,
        val_entropy_prompt_batch_size=cfg.val_entropy_prompt_batch_size,
        val_entropy_forward_chunk_size=cfg.val_entropy_forward_chunk_size,
        resume_start_epoch_1based=cfg.resume_start_epoch_1based,
        resume_checkpoint_dir=cfg.resume_from,
    )


def _lambda_min_type(x: str) -> float:
    v = float(x)
    if not 0.0 <= v <= 1.0:
        raise ValueError(f"--lambda-min must be in [0, 1], got {v}")
    return v


def _parse_cli_to_config() -> SoftDPOConfig:
    """Разбирает CLI и возвращает SoftDPOConfig. Значения CLI имеют приоритет над
    дефолтами dataclass'а; поля без флагов остаются со значениями из config/base_config.py.
    """
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
    parser.add_argument("--alpha", type=float, default=0.2, help="Параметр бета-приора для p_bayes; имеет смысл только при --use-bayes (по умолчанию 0.2 — слабый приор)")
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
    parser.add_argument("--batch-size", "-b", type=int, default=8, help="Размер батча для train и validation (по умолчанию: 8).")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate (по умолчанию: 3e-5).")
    parser.add_argument("--beta", type=float, default=0.3, help="Параметр beta для DPO loss (по умолчанию: 0.3).")
    parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=8,
        help=(
            "Всего эпох в плане (λ и LR по шкале 1..epochs). При --start-epoch 1 — столько же шагов обучения; "
            "при продолжении (--start-epoch>1) обучаются эпохи start..epochs (нужно epochs >= start-epoch)."
        ),
    )
    parser.add_argument(
        "--start-epoch",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Первая эпоха этого запуска (1-based); должна быть <= --epochs. "
            "Веса --resume — после эпохи N-1 (после epochs/epoch_003 укажите 4). "
            "Число эпох в этом запуске: epochs - start + 1. "
            "При --resume и N>1 в начало train.log в --output-dir переносится история "
            "старого train.log (рядом с чекпоинтом) до эпохи N, если граница найдена в логе. "
            "При --lambda-full-epochs k и N=k+1 до цикла эпох восстанавливается p_pred_teacher "
            "с загруженных весов (как фиксация учителя в конце эпохи k)."
        ),
    )
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
    parser.add_argument(
        "--lambda-full-epochs",
        type=int,
        default=0,
        help=(
            "k (1-based): эпохи 1..k с lambda=1; в конце эпохи k фиксируется учитель; с эпохи k+1 λ<1. "
            "При λ<1: p_pred = 0.5*teacher + 0.5*σ((beta*diff)/T) на всех хвостовых шагах. "
            "0 — как раньше (см. train_dpo)."
        ),
    )
    parser.add_argument(
        "--capability-eval-dir",
        type=str,
        default=None,
        help="Каталог eval_datasets: на каждой валидации лог capability retention (gold).",
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
    return SoftDPOConfig(
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
        lambda_full_epochs=args.lambda_full_epochs,
        capability_eval_dir=args.capability_eval_dir,
        capability_ref_cache_path=args.capability_ref_cache_path,
        val_kl_mc_max_prompts=args.val_kl_mc_max_prompts,
        resume_start_epoch_1based=args.start_epoch,
    )


if __name__ == "__main__":
    main(_parse_cli_to_config())
