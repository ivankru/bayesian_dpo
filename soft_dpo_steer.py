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
    """Single source of truth for soft/bayes-DPO run hyperparameters.

    CLI-overridable fields have defaults here. Some fields
    (p_pred_target_temperature, val_entropy_*, capability_eval_*) are not exposed on the CLI —
    they come from config.base_config and change with config commits,
    not launch flags (to keep experiments comparable).
    """

    # --- basic ---
    seed: int = 42
    base_model: str = "3b"
    dataset: str = "helpsteer3"
    output_dir: str = "checkpoints/soft_dpo_steer"
    resume_from: Optional[str] = None

    # --- DPO / soft hyperparameters ---
    alpha: float = 0.2
    label_noise_prob: float = 0.0
    use_bayes: bool = False
    batch_size: int = 8
    lr: float = 3e-5
    optimizer_name: str = "AdamW"
    grad_clip_norm: float = 0.0
    beta: float = 0.3
    epochs: int = 8

    # --- lambda schedule ---
    lambda_min: float = 1.0
    lambda_schedule: str = "linear"
    lambda_full_epochs: int = 0
    p_pred_target_temperature: float = P_PRED_TARGET_TEMPERATURE
    soft_loss_type: str = "approximation"
    use_chat_template: bool = USE_CHAT_TEMPLATE

    # --- capability retention (defaults in config/base_config.py) ---
    capability_eval_dir: Optional[str] = None
    capability_eval_limit: Optional[int] = CAPABILITY_EVAL_LIMIT
    capability_eval_max_new_tokens: int = CAPABILITY_EVAL_MAX_NEW_TOKENS
    capability_eval_batch_size: int = CAPABILITY_EVAL_BATCH_SIZE
    capability_eval_max_prompt_tokens: int = CAPABILITY_EVAL_MAX_PROMPT_TOKENS
    capability_ref_cache_path: Optional[str] = None

    # --- val KL-MC ---
    val_kl_mc_max_prompts: int = DEFAULT_VAL_KL_MC_MAX_PROMPTS

    # --- val response entropy (defaults in config/base_config.py) ---
    val_entropy_max_prompts: int = VAL_ENTROPY_MAX_PROMPTS
    val_entropy_num_samples: int = VAL_ENTROPY_NUM_SAMPLES
    val_entropy_max_new_tokens: int = VAL_ENTROPY_MAX_NEW_TOKENS
    val_entropy_prompt_batch_size: int = VAL_ENTROPY_PROMPT_BATCH_SIZE
    val_entropy_forward_chunk_size: int = VAL_ENTROPY_FORWARD_CHUNK_SIZE

    # --- resume ---
    resume_start_epoch_1based: int = 1


def main(cfg: SoftDPOConfig) -> None:
    """Soft train + hard validation. All hyperparameters live in SoftDPOConfig.

    seed: reproducibility; same default as hard_dpo_steer (42) matches initial val metrics.
    alpha: beta-prior strength for p_bayes; 0.2 is a weak prior (α ≈ 0.2, 2α = 0.4 pseudo-counts).
    use_bayes: if True, loss uses p_bayes; else p (default).
    base_model: "3b" | "7b" — Qwen2.5-*B-Instruct; "4b" — Qwen3-4B-Instruct-2507.
    dataset: helpsteer3 | ultrafeedback_binarized (binary p) | ultrafeedback_soft (p from scores) | openbmb | hh_rlhf.
    lambda_full_epochs: k (1-based): epochs 1..k labels only; end of epoch k fixes p_pred_teacher; from k+1 λ<1;
        at λ<1, p_pred is always 0.5·teacher + 0.5·σ((beta*diff)/T). 0 — legacy behavior.
    resume_start_epoch_1based: see utils.training.train_dpo (--epochs = full plan, --start-epoch = first epoch this run).
    """
    if cfg.dataset not in DATASET_CHOICES:
        raise ValueError(
            f"dataset must be one of {DATASET_CHOICES}, got: {cfg.dataset!r}"
        )
    set_seed(cfg.seed)
    model_name = BASE_MODEL_CHOICES[cfg.base_model]
    if cfg.dataset == "helpsteer3":
        print("Loading HelpSteer3-Preference...")
        train_soft_ds, val_hard_ds, hard_train_size = build_helpsteer3_soft_datasets(
            alpha=cfg.alpha,
            label_noise_prob=cfg.label_noise_prob,
            seed=cfg.seed,
        )
    elif cfg.dataset == "ultrafeedback_binarized":
        print("Loading UltraFeedback Binarized (binary labels chosen>rejected)...")
        train_soft_ds, val_hard_ds, hard_train_size = build_ultrafeedback_binarized_soft_datasets(
            alpha=cfg.alpha,
            label_noise_prob=cfg.label_noise_prob,
            seed=cfg.seed,
        )
    elif cfg.dataset == "ultrafeedback_soft":
        print("Loading UltraFeedback (soft labels from score_chosen/score_rejected)...")
        train_soft_ds, val_hard_ds, hard_train_size = build_ultrafeedback_score_soft_datasets(
            alpha=cfg.alpha,
            label_noise_prob=cfg.label_noise_prob,
            seed=cfg.seed,
        )
    elif cfg.dataset == "hh_rlhf":
        print("Loading PKU processed HH-RLHF (soft train, hard val)...")
        train_soft_ds, val_hard_ds, hard_train_size = build_hh_rlhf_soft_steer_datasets(alpha=cfg.alpha)
    else:  # openbmb
        print("Loading openbmb/UltraFeedback (soft) + val ultrafeedback_binarized...")
        train_soft_ds, val_hard_ds, hard_train_size = build_openbmb_soft_datasets(alpha=cfg.alpha)
    prob_type = "p_bayes" if cfg.use_bayes else "p"
    print(f"Model: {model_name}, Dataset: {cfg.dataset}")
    print(
        f"Train soft size: {len(train_soft_ds)}, val hard size: {len(val_hard_ds)}, "
        f"hard train size: {hard_train_size}, alpha={cfg.alpha}, target_prob={prob_type}"
    )

    if cfg.resume_from:
        print(f"Loading model from checkpoint: {cfg.resume_from} (base {model_name})")
    else:
        print(f"Loading model and tokenizer: {model_name} (LoRA)")
    tokenizer, policy_model, ref_model, device = load_models_and_tokenizer(
        model_name, use_lora=True, lora_r=16, lora_alpha=32, resume_from=cfg.resume_from
    )

    def log_fn(msg: str) -> None:
        # Log lines go to stdout so they do not mix with tqdm (stderr).
        # Keeps a clean `>run.log` with meaningful lines only.
        print(msg, flush=True, file=sys.stdout)

    mode = "bayes" if cfg.use_bayes else "soft"
    num_steps_override = (
        cfg.epochs * ((hard_train_size + cfg.batch_size - 1) // cfg.batch_size)
        if hard_train_size
        else None
    )
    print(f"Starting {mode.upper()}-DPO training (train {mode}, validation hard)...")
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
        optimizer_name=cfg.optimizer_name,
        grad_clip_norm=cfg.grad_clip_norm,
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
        soft_loss_type=cfg.soft_loss_type,
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


def _optimizer_type(x: str) -> str:
    v = str(x).strip().lower()
    if v == "adamw":
        return "AdamW"
    if v == "sgd":
        return "SGD"
    raise ValueError(f"--optimizer must be one of: AdamW, SGD; got {x!r}")


def _parse_cli_to_config() -> SoftDPOConfig:
    """Parse CLI into SoftDPOConfig. CLI values override dataclass defaults;
    fields without flags keep values from config/base_config.py.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Soft-DPO (train soft, validation hard): HelpSteer3; UltraFeedback binarized or score-soft; "
            "openbmb; HH-RLHF (PKU)."
        )
    )
    parser.add_argument(
        "--resume", "-r",
        type=str,
        default=None,
        help="Checkpoint path to resume from (e.g. checkpoints/soft_dpo_steer/best)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility (match hard_dpo_steer to compare runs)")
    parser.add_argument(
        "--label-noise-prob",
        type=float,
        default=0.0,
        help=(
            "Train label noise: for binary datasets (HelpSteer3, ultrafeedback_binarized) — "
            "flip p 0↔1 with given probability; for ultrafeedback_soft — replace p with 1−p "
            "(and recompute p_bayes) with the same probability."
        ),
    )
    parser.add_argument("--alpha", type=float, default=0.2, help="Beta-prior strength for p_bayes; only matters with --use-bayes (default 0.2 — weak prior)")
    parser.add_argument("--use-bayes", action="store_true", help="Use p_bayes instead of p as target probability (default: p)")
    parser.add_argument("--output-dir", "-o", type=str, default="checkpoints/soft_dpo_steer", help="Directory for checkpoints and train.log (use a new folder per run)")
    parser.add_argument(
        "--base-model",
        type=str,
        choices=list(BASE_MODEL_CHOICES.keys()),
        default="3b",
        help="Base model: 3b/7b — Qwen2.5-Instruct; 4b — Qwen3-4B-Instruct-2507. Default: 3b.",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="helpsteer3",
        choices=list(DATASET_CHOICES),
        help=(
            "Dataset: helpsteer3; ultrafeedback_binarized (hard chosen>rejected, p∈{0,1}); "
            "ultrafeedback_soft (p=sigmoid(Δscore)); openbmb (soft); hh_rlhf (PKU processed)."
        ),
    )
    parser.add_argument("--batch-size", "-b", type=int, default=8, help="Batch size for train and validation (default: 8).")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate (default: 3e-5).")
    parser.add_argument(
        "--optimizer",
        type=_optimizer_type,
        default="AdamW",
        help="Policy optimizer (case-insensitive): AdamW (default) or SGD.",
    )
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=0.0,
        help="Max norm for clip_grad_norm_; 0 disables clipping (default: 0).",
    )
    parser.add_argument("--beta", type=float, default=0.3, help="DPO beta parameter (default: 0.3).")
    parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=8,
        help=(
            "Total planned epochs (λ and LR on scale 1..epochs). With --start-epoch 1, training steps match full plan; "
            "when resuming (--start-epoch>1), trains epochs start..epochs (require epochs >= start-epoch)."
        ),
    )
    parser.add_argument(
        "--start-epoch",
        type=int,
        default=1,
        metavar="N",
        help=(
            "First epoch of this run (1-based); must be <= --epochs. "
            "--resume weights are after epoch N-1 (after epochs/epoch_003 pass 4). "
            "Epochs in this run: epochs - start + 1. "
            "With --resume and N>1, prior train.log (next to checkpoint) is prepended to train.log in --output-dir "
            "through epoch N if a boundary is found in the log. "
            "With --lambda-full-epochs k and N=k+1, p_pred_teacher is restored from loaded weights "
            "before the epoch loop (as teacher fix at end of epoch k)."
        ),
    )
    parser.add_argument(
        "--lambda-min",
        type=_lambda_min_type,
        default=1.0,
        help="Minimum lambda_label per epoch [0, 1]; 1.0 = dataset labels only (default: 1.0).",
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
            "k (1-based): epochs 1..k have lambda=1; end of epoch k fixes teacher; from epoch k+1 λ<1. "
            "At λ<1: p_pred = 0.5*teacher + 0.5*σ((beta*diff)/T) on all tail steps. "
            "0 — legacy (see train_dpo)."
        ),
    )
    parser.add_argument(
        "--soft-loss-type",
        type=str,
        choices=["classic", "approximation", "centered_softplus"],
        default="classic",
        help=(
            "Soft train loss variant: classic=soft_dpo_classic_loss; "
            "approximation=soft_dpo_approximation_loss (scaled old_loss/beta, small-beta approximation); "
            "centered_softplus=soft_dpo_centered_softplus_loss."
        ),
    )
    parser.add_argument(
        "--capability-eval-dir",
        type=str,
        default=None,
        help="eval_datasets directory: log capability retention (gold) on each validation.",
    )
    parser.add_argument(
        "--capability-ref-cache-path",
        type=str,
        default=None,
        help="Explicit path to JSON cache of ref answers for capability retention (optional).",
    )
    parser.add_argument(
        "--val-kl-mc-max-prompts",
        type=int,
        default=DEFAULT_VAL_KL_MC_MAX_PROMPTS,
        help=(
            "MC forward KL(π‖ref) on val: first N prompts; 0 disables "
            f"(default {DEFAULT_VAL_KL_MC_MAX_PROMPTS})."
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
        optimizer_name=args.optimizer,
        grad_clip_norm=args.grad_clip_norm,
        beta=args.beta,
        epochs=args.epochs,
        lambda_min=args.lambda_min,
        lambda_schedule=args.lambda_schedule,
        lambda_full_epochs=args.lambda_full_epochs,
        soft_loss_type=args.soft_loss_type,
        capability_eval_dir=args.capability_eval_dir,
        capability_ref_cache_path=args.capability_ref_cache_path,
        val_kl_mc_max_prompts=args.val_kl_mc_max_prompts,
        resume_start_epoch_1based=args.start_epoch,
    )


if __name__ == "__main__":
    main(_parse_cli_to_config())
