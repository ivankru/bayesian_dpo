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
from utils.config import BASE_MODEL_CHOICES, BASE_MODEL_HELP, DPO_STEER_HARD_DATASET_CHOICES as DATASET_CHOICES
from utils.seed import set_seed
from utils.datasets import (
    build_dpo_datasets,
    build_dpo_datasets_hh_rlhf,
    build_dpo_datasets_orca_dpo,
    build_dpo_datasets_ultrafeedback,
)
from utils.models import load_models_and_tokenizer
from utils.training import DEFAULT_VAL_KL_MC_MAX_PROMPTS, ensure_output_dir_lock, train_dpo


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
    grad_clip_norm: float = 0.0,
    optimizer_name: str = "AdamW",
    save_epoch_checkpoints: bool = True,
    probe_margins: bool = False,
    probe_size: int = 256,
    probe_every: int = 100,
    probe_seed: int = 0,
):
    """
    resume_from: checkpoint path (e.g. "checkpoints/hard_dpo_steer/best").
    If set, policy and tokenizer load from checkpoint and training continues from those weights.
    seed: reproducibility; same seed in hard_dpo_steer and soft_steer matches initial val metrics.
    output_dir: directory for checkpoints and train.log.
    dataset: "helpsteer3" | "ultrafeedback_binarized" | "hh_rlhf" | "orca_dpo"
        (Intel/orca_dpo_pairs).
    base_model: "3b" | "7b" — Qwen2.5-*B-Instruct; "4b" — Qwen3-4B-Instruct-2507;
        "3.8b" — microsoft/Phi-4-mini-instruct.
    batch_size: batch size for train and validation.
    lambda_min: unused in hard mode (kept for CLI parity with soft_dpo_steer).
    use_chat_template: log p via apply_chat_template (default in config.base_config).
    capability_eval_dir: if set, eval_datasets (gold) on each validation; see train_dpo.
    resume_start_epoch_1based: see utils.training.train_dpo (--epochs = full plan, --start-epoch).
    """
    if dataset not in DATASET_CHOICES:
        raise ValueError(f"dataset must be one of {DATASET_CHOICES}, got: {dataset!r}")
    os.makedirs(output_dir, exist_ok=True)
    try:
        ensure_output_dir_lock(output_dir)
    except RuntimeError as e:
        print(e, file=sys.stderr, flush=True)
        sys.exit(1)
    set_seed(seed)
    if dataset == "helpsteer3":
        print("Loading HelpSteer3-Preference...")
        train_ds, val_ds = build_dpo_datasets()
    elif dataset == "ultrafeedback_binarized":
        print("Loading UltraFeedback Binarized...")
        train_ds, val_ds = build_dpo_datasets_ultrafeedback()
    elif dataset == "hh_rlhf":
        print("Loading PKU processed HH-RLHF...")
        train_ds, val_ds = build_dpo_datasets_hh_rlhf()
    elif dataset == "orca_dpo":
        print("Loading Intel/orca_dpo_pairs...")
        train_ds, val_ds = build_dpo_datasets_orca_dpo()
    else:
        raise ValueError(f"dataset must be one of {DATASET_CHOICES}, got: {dataset!r}")
    model_name = BASE_MODEL_CHOICES[base_model]
    print(f"Model: {model_name}, Dataset: {dataset}, train size: {len(train_ds)}, val size: {len(val_ds)}")
    if resume_from:
        print(f"Loading model from checkpoint: {resume_from} (base {model_name})")
    else:
        print(f"Loading model and tokenizer: {model_name} (LoRA)")
    tokenizer, policy_model, ref_model, device = load_models_and_tokenizer(
        model_name, use_lora=True, lora_r=16, lora_alpha=32, resume_from=resume_from
    )

    def log_fn(msg: str) -> None:
        # Log lines go to stdout so they do not mix with tqdm (stderr).
        # Keeps a clean `>run.log` with meaningful lines only.
        print(msg, flush=True, file=sys.stdout)

    print("Starting DPO (hard) training...")
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
        grad_clip_norm=grad_clip_norm,
        optimizer_name=optimizer_name,
        save_epoch_checkpoints=save_epoch_checkpoints,
        probe_margins=probe_margins,
        probe_size=probe_size,
        probe_every=probe_every,
        probe_seed=probe_seed,
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Hard DPO: HelpSteer3, UltraFeedback Binarized, HH-RLHF, or Orca DPO pairs."
    )
    parser.add_argument(
        "--resume", "-r",
        type=str,
        default=None,
        help="Checkpoint path to resume from (e.g. checkpoints/hard_dpo_steer/best)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility (default 42)")
    parser.add_argument("--output-dir", "-o", type=str, default="checkpoints/hard_dpo_steer", help="Directory for checkpoints and train.log (use a new folder per run)")
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="helpsteer3",
        choices=list(DATASET_CHOICES),
        help="Dataset: helpsteer3, ultrafeedback_binarized, hh_rlhf, or orca_dpo (Intel/orca_dpo_pairs).",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        choices=list(BASE_MODEL_CHOICES.keys()),
        default="3b",
        help=BASE_MODEL_HELP + " Default: 3b.",
    )
    parser.add_argument("--batch-size", "-b", type=int, default=8, help="Batch size for train and validation (default: 8).")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate (default: 2e-5).")
    parser.add_argument(
        "--optimizer",
        type=_optimizer_type,
        default="AdamW",
        help="Policy optimizer (case-insensitive): AdamW (default) or SGD.",
    )
    parser.add_argument("--beta", type=float, default=0.2, help="DPO beta parameter (default: 0.2).")
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=0.0,
        help="Max norm for clip_grad_norm_; 0 disables clipping (default: 0).",
    )
    parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=8,
        help="Total planned epochs (LR on scale 1..epochs). With --start-epoch>1, trains epochs start..epochs.",
    )
    parser.add_argument(
        "--start-epoch",
        type=int,
        default=1,
        metavar="N",
        help=(
            "First epoch of run (1-based), <= --epochs; --resume weights after epoch N-1. "
            "Epochs in this run: epochs - start + 1. "
            "With --resume and N>1, prior train.log (next to checkpoint) is prepended to train.log in --output-dir "
            "through epoch N if a boundary is found in the log."
        ),
    )
    parser.add_argument(
        "--no-epoch-checkpoints",
        action="store_true",
        help=(
            "Do not save epochs/epoch_XXX after each full epoch. "
            "best/ is still written when val NLL improves."
        ),
    )
    parser.add_argument(
        "--lambda-min",
        type=_lambda_min_type,
        default=1.0,
        help="No effect in hard mode; shared flag with soft_dpo_steer [0, 1] (default: 1.0).",
    )
    parser.add_argument(
        "--capability-eval-dir",
        type=str,
        default=None,
        help="eval_datasets directory (knowledge/*.jsonl, reasoning/*.jsonl): log retention on each validation.",
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
    parser.add_argument(
        "--probe-margins",
        action="store_true",
        help=(
            "Every --probe-every steps, log Δ on a fixed val subset "
            "(--probe-size pairs, --probe-seed; not the training seed). "
            "Caches frozen ref logps (2 policy forwards/snapshot). Writes "
            "probe_margins.jsonl plus full-val Δ at each epoch val. "
            "No extra 100-pair set."
        ),
    )
    parser.add_argument(
        "--probe-size",
        type=int,
        default=256,
        help="Fixed val probe size when --probe-margins is set (default: 256).",
    )
    parser.add_argument(
        "--probe-every",
        type=int,
        default=100,
        help="Probe cadence in optimizer steps when --probe-margins is set (default: 100).",
    )
    parser.add_argument(
        "--probe-seed",
        type=int,
        default=0,
        help="RNG seed for the probe subset; independent of --seed (default: 0).",
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
        optimizer_name=args.optimizer,
        grad_clip_norm=args.grad_clip_norm,
        epochs=args.epochs,
        lambda_min=args.lambda_min,
        capability_eval_dir=args.capability_eval_dir,
        capability_ref_cache_path=args.capability_ref_cache_path,
        val_kl_mc_max_prompts=args.val_kl_mc_max_prompts,
        resume_start_epoch_1based=args.start_epoch,
        save_epoch_checkpoints=not args.no_epoch_checkpoints,
        probe_margins=args.probe_margins,
        probe_size=args.probe_size,
        probe_every=args.probe_every,
        probe_seed=args.probe_seed,
    )
