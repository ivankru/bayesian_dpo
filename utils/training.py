# -*- coding: utf-8 -*-
"""
Общий цикл обучения одной эпохи для DPO и универсальная функция train_dpo (режимы hard / soft / bayes).
"""
import os
from typing import Any, Callable, Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from utils.config import MAX_FULL_LEN, MAX_PROMPT_LEN
from utils.loss import hard_dpo_loss, soft_dpo_loss
from utils.metrics import eval_pairwise_accuracy, eval_pairwise_nll

DPO_MODE_CHOICES = ("hard", "soft", "bayes")


def collate_fn_hard(examples: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    return {
        "prompt": [e["prompt"] for e in examples],
        "chosen": [e["chosen"] for e in examples],
        "rejected": [e["rejected"] for e in examples],
    }


def collate_fn_soft(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "prompt": [e["prompt"] for e in examples],
        "resp1": [e["resp1"] for e in examples],
        "resp2": [e["resp2"] for e in examples],
        "p": [e["p"] for e in examples],
        "p_bayes": [e["p_bayes"] for e in examples],
    }


def train_one_epoch_dpo(
    train_loader,
    tokenizer,
    policy_model,
    ref_model,
    device: str,
    loss_fn: Callable[..., Any],
    optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    log=print,
    **loss_kwargs: Any,
) -> int:
    """
    Одна эпоха DPO. loss_fn(batch, tokenizer, policy_model, ref_model, device, **loss_kwargs)
    должна возвращать (loss, kl_approx).
    """
    policy_model.train()
    running_loss = 0.0
    running_kl = 0.0
    log_interval = 100

    for batch in train_loader:
        optimizer.zero_grad(set_to_none=True)
        loss, kl_batch = loss_fn(
            batch, tokenizer, policy_model, ref_model, device, **loss_kwargs
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        running_kl += kl_batch
        global_step += 1

        if global_step % 1000 == 0:
            log(f"  step {global_step} lr={optimizer.param_groups[0]['lr']:.2e}")
        if global_step % log_interval == 0:
            n = log_interval
            log(
                f"[epoch {epoch+1}] step {global_step} train_loss={running_loss / n:.4f} kl_pi_ref={running_kl / n:.4f}"
            )
            running_loss = 0.0
            running_kl = 0.0

    return global_step


def train_dpo(
    train_ds: Dataset,
    val_ds: Dataset,
    tokenizer,
    policy_model,
    ref_model,
    device: str,
    mode: str = "hard",
    epochs: int = 1,
    batch_size: int = 8,
    lr: float = 5e-6,
    beta: float = 0.2,
    alpha: float = 1.0,
    output_dir: str = "checkpoints/dpo",
    num_training_steps_override: Optional[int] = None,
    dataset_name: Optional[str] = None,
    model_name: Optional[str] = None,
    lambda_min: float = 1.0,
    seed: int = 42,
    log=print,
):
    """
    Универсальный цикл DPO: hard, soft или bayes.

    mode: "hard" — train и val в формате chosen/rejected, loss = hard_dpo_loss.
          "soft" — train в формате resp1, resp2, p, p_bayes; val в формате chosen/rejected; train loss = soft_dpo_loss(use_bayes=False).
          "bayes" — как soft, но train loss = soft_dpo_loss(use_bayes=True).

    val_ds всегда в формате chosen/rejected; валидация по hard DPO loss, NLL, accuracy.
    num_training_steps_override: для soft/bayes можно задать число шагов (например по hard train size) для выравнивания LR schedule.
    lambda_min: для soft/bayes — нижняя граница lambda_label по эпохам (смешивание с p_pred); при 1.0 поведение как раньше.
    seed: фиксирует shuffle train DataLoader (torch.Generator + num_workers=0).
    """
    if mode not in DPO_MODE_CHOICES:
        raise ValueError(f"mode должен быть один из {DPO_MODE_CHOICES}, получено: {mode!r}")
    if not 0.0 <= lambda_min <= 1.0:
        raise ValueError(f"lambda_min must be in [0, 1], got {lambda_min!r}")

    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "train.log")

    def log_msg(msg: str) -> None:
        log(msg)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    use_bayes = mode == "bayes"
    if mode == "hard":
        train_collate = collate_fn_hard
        train_loss_fn = hard_dpo_loss
        loss_kwargs = {"beta": beta}
        mode_label = "Hard DPO"
    else:
        train_collate = collate_fn_soft
        train_loss_fn = soft_dpo_loss
        loss_kwargs = {"beta": beta, "use_bayes": use_bayes}
        mode_label = "Bayes DPO" if use_bayes else "Soft DPO"

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_collate,
        num_workers=0,
        generator=g,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_hard,
        num_workers=0,
    )

    num_training_steps = num_training_steps_override or epochs * len(train_loader)
    if num_training_steps_override is not None:
        log_msg(f"LR schedule: num_training_steps={num_training_steps} (override)")

    log_msg(f"=== {mode_label} ===")
    log_msg(f"Model: {model_name or 'N/A'}, Dataset: {dataset_name or 'N/A'}, train size: {len(train_ds)}, val size: {len(val_ds)}")
    log_msg(
        f"Старт train_dpo: mode={mode}, beta={beta}, lr={lr}, batch_size={batch_size}, epochs={epochs}, lambda_min={lambda_min}, seed={seed}"
    )
    log_msg(f"MAX_PROMPT_LEN={MAX_PROMPT_LEN}, MAX_FULL_LEN={MAX_FULL_LEN}")

    # Начальная валидация (hard)
    policy_model.eval()
    init_dpo_sum = 0.0
    init_kl_sum = 0.0
    init_n = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="init DPO loss", leave=False):
            loss, kl_b = hard_dpo_loss(batch, tokenizer, policy_model, ref_model, device, beta=beta)
            n = len(batch["prompt"])
            init_dpo_sum += loss.item() * n
            init_kl_sum += kl_b * n
            init_n += n
    init_dpo = init_dpo_sum / max(1, init_n)
    init_kl = init_kl_sum / max(1, init_n)
    init_nll = eval_pairwise_nll(val_loader, tokenizer, policy_model, device, beta=1.0, desc="init pairwise NLL")
    init_acc = eval_pairwise_accuracy(val_loader, tokenizer, policy_model, device, desc="init pairwise acc")

    log_msg("=== Initial (before training) ===")
    log_msg(f"validation DPO loss   : {init_dpo:.4f}")
    log_msg(f"validation KL(π||ref) : {init_kl:.4f}")
    log_msg(f"validation pair NLL   : {init_nll:.4f}")
    log_msg(f"validation pair acc   : {init_acc:.4f}")

    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(10, num_training_steps // 20),
        num_training_steps=num_training_steps,
    )

    global_step = 0
    best_val_nll = float("inf")
    for epoch in range(epochs):
        if mode == "hard":
            epoch_loss_kwargs = loss_kwargs
        else:
            progress_epoch = epoch / (epochs - 1) if epochs > 1 else 1.0
            lambda_label_epoch = max(lambda_min, 1.0 - progress_epoch)
            epoch_loss_kwargs = {**loss_kwargs, "lambda_label": lambda_label_epoch}
            log_msg(f"Epoch {epoch + 1}/{epochs}: lambda_label={lambda_label_epoch:.6f}")

        global_step = train_one_epoch_dpo(
            train_loader,
            tokenizer,
            policy_model,
            ref_model,
            device,
            train_loss_fn,
            optimizer,
            scheduler,
            epoch,
            global_step,
            log=log_msg,
            **epoch_loss_kwargs,
        )

        policy_model.eval()
        val_dpo_sum = 0.0
        val_kl_sum = 0.0
        val_n = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"val DPO ep{epoch+1}", leave=False):
                loss, kl_b = hard_dpo_loss(batch, tokenizer, policy_model, ref_model, device, beta=beta)
                n = len(batch["prompt"])
                val_dpo_sum += loss.item() * n
                val_kl_sum += kl_b * n
                val_n += n
        val_dpo = val_dpo_sum / max(1, val_n)
        val_kl = val_kl_sum / max(1, val_n)
        val_nll = eval_pairwise_nll(val_loader, tokenizer, policy_model, device, beta=1.0, desc=f"val NLL ep{epoch+1}")
        val_acc = eval_pairwise_accuracy(val_loader, tokenizer, policy_model, device, desc=f"val acc ep{epoch+1}")

        log_msg(f"=== Epoch {epoch+1} ===")
        log_msg(f"validation DPO loss   : {val_dpo:.4f}")
        log_msg(f"validation KL(π||ref) : {val_kl:.4f}")
        log_msg(f"validation pair NLL   : {val_nll:.4f}")
        log_msg(f"validation pair acc   : {val_acc:.4f}")

        if val_nll < best_val_nll:
            best_val_nll = val_nll
            ckpt_dir = os.path.join(output_dir, "best")
            os.makedirs(ckpt_dir, exist_ok=True)
            tokenizer.save_pretrained(ckpt_dir)
            policy_model.save_pretrained(ckpt_dir)
            log_msg(f"New best NLL {val_nll:.4f} -> checkpoint saved: {ckpt_dir}")
