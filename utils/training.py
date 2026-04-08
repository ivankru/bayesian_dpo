# -*- coding: utf-8 -*-
"""
Общий цикл обучения одной эпохи для DPO и универсальная функция train_dpo (режимы hard / soft / bayes).
"""
import math
import os
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, Optional

import mlflow
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from utils.config import MAX_FULL_LEN, MAX_PROMPT_LEN
from utils.datasets import precompute_p_pred_cached
from utils.loss import hard_dpo_loss, soft_dpo_loss
from utils.metrics import eval_pairwise_accuracy, eval_pairwise_nll
from utils.val_distributions import compute_val_delta_distributions
from utils.val_kl_mc import estimate_val_kl_mc

DPO_MODE_CHOICES = ("hard", "soft", "bayes")

@contextmanager
def _mlflow_training_context(
    enabled: bool,
    experiment: str,
    run_name: Optional[str],
    tracking_uri: Optional[str],
    params: Dict[str, Any],
    log_path: str,
) -> Iterator[None]:
    if not enabled:
        yield
        return
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    mlflow.start_run(run_name=run_name)
    try:
        to_log = {k: str(v) for k, v in params.items() if v is not None}
        mlflow.log_params(to_log)
        yield
    finally:
        if os.path.isfile(log_path):
            try:
                mlflow.log_artifact(log_path)
            except OSError:
                pass
        mlflow.end_run()


def collate_fn_hard(examples: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    return {
        "prompt": [e["prompt"] for e in examples],
        "chosen": [e["chosen"] for e in examples],
        "rejected": [e["rejected"] for e in examples],
    }


def collate_fn_soft(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "prompt": [e["prompt"] for e in examples],
        "resp1": [e["resp1"] for e in examples],
        "resp2": [e["resp2"] for e in examples],
        "p": [e["p"] for e in examples],
        "p_bayes": [e["p_bayes"] for e in examples],
    }
    if examples and "p_pred_cached" in examples[0]:
        out["p_pred_cached"] = [e["p_pred_cached"] for e in examples]
    return out


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
    use_mlflow: bool = False,
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
            lr_cur = optimizer.param_groups[0]["lr"]
            log(f"  step {global_step} lr={lr_cur:.2e}")
            if use_mlflow:
                mlflow.log_metric("lr", lr_cur, step=global_step)
        if global_step % log_interval == 0:
            n = log_interval
            log(
                f"[epoch {epoch+1}] step {global_step} train_loss={running_loss / n:.4f} kl_pi_ref={running_kl / n:.4f}"
            )
            if use_mlflow:
                mlflow.log_metric("train_loss", running_loss / n, step=global_step)
                mlflow.log_metric("train_kl_pi_ref", running_kl / n, step=global_step)
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
    lambda_schedule: str = "linear",
    seed: int = 42,
    label_noise_prob: Optional[float] = None,
    use_chat_template: bool = False,
    log=print,
    use_mlflow: bool = False,
    mlflow_experiment: str = "bayesian_dpo",
    mlflow_run_name: Optional[str] = None,
    mlflow_tracking_uri: Optional[str] = None,
    val_kl_mc_max_prompts: Optional[int] = None,
    val_kl_mc_num_samples: int = 4,
    val_kl_mc_max_new_tokens: int = 128,
    val_kl_mc_prompt_batch_size: int = 6,
    val_distributions_max_batches: Optional[int] = None,
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
    label_noise_prob: вероятность шума меток при сборке soft train (--label-noise-prob); для hard не задаётся (в логе N/A).
    use_chat_template: если True, get_logps использует tokenizer.apply_chat_template (Qwen-Instruct); иначе plain prompt\\nresponse.
    use_mlflow: логировать параметры, метрики и train.log в MLflow (tracking URI из mlflow_tracking_uri или окружения по умолчанию).
    val_kl_mc_max_prompts: если задано (>0), в конце каждой эпохи считается MC-оценка KL(π‖ref) по сэмплам π_θ
          на первых N промптах val (см. utils.val_kl_mc.estimate_val_kl_mc); лог: val_kl_mc, метрика MLflow при use_mlflow.
    val_kl_mc_num_samples: число независимых генераций на промпт для MC.
    val_distributions_max_batches: если задано (>0), после основных val-метрик считаются распределения
        delta_theta, delta_ref, diff на первых N батчах val; лог, MLflow, np.savez_compressed в output_dir.
    """
    if mode not in DPO_MODE_CHOICES:
        raise ValueError(f"mode должен быть один из {DPO_MODE_CHOICES}, получено: {mode!r}")
    if not 0.0 <= lambda_min <= 1.0:
        raise ValueError(f"lambda_min must be in [0, 1], got {lambda_min!r}")
    if lambda_schedule not in ("linear", "cosine"):
        raise ValueError(
            f"lambda_schedule must be one of ('linear', 'cosine'), got {lambda_schedule!r}"
        )

    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "train.log")

    def log_msg(msg: str) -> None:
        log(msg)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    mlflow_param_dict: Dict[str, Any] = {
        "mode": mode,
        "beta": beta,
        "lr": lr,
        "batch_size": batch_size,
        "epochs": epochs,
        "lambda_min": lambda_min,
        "lambda_schedule": lambda_schedule,
        "seed": seed,
        "dataset_name": dataset_name,
        "model_name": model_name,
        "output_dir": output_dir,
        "alpha": alpha,
        "label_noise_prob": label_noise_prob,
        "use_chat_template": use_chat_template,
        "num_training_steps_override": num_training_steps_override,
        "val_kl_mc_max_prompts": val_kl_mc_max_prompts,
        "val_kl_mc_num_samples": val_kl_mc_num_samples,
        "val_kl_mc_max_new_tokens": val_kl_mc_max_new_tokens,
        "val_distributions_max_batches": val_distributions_max_batches,
    }

    with _mlflow_training_context(
        use_mlflow,
        mlflow_experiment,
        mlflow_run_name,
        mlflow_tracking_uri,
        mlflow_param_dict,
        log_path,
    ):
        use_bayes = mode == "bayes"
        if mode == "hard":
            train_collate = collate_fn_hard
            train_loss_fn = hard_dpo_loss
            loss_kwargs = {"beta": beta, "use_chat_template": use_chat_template}
            mode_label = "Hard DPO"
        else:
            train_collate = collate_fn_soft
            train_loss_fn = soft_dpo_loss
            loss_kwargs = {"beta": beta, "use_bayes": use_bayes, "use_chat_template": use_chat_template}
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
        if use_mlflow:
            mlflow.log_param("num_training_steps", num_training_steps)
        if num_training_steps_override is not None:
            log_msg(f"LR schedule: num_training_steps={num_training_steps} (override)")

        log_msg(f"=== {mode_label} ===")
        log_msg(f"Model: {model_name or 'N/A'}, Dataset: {dataset_name or 'N/A'}, train size: {len(train_ds)}, val size: {len(val_ds)}")
        _lnp = label_noise_prob if label_noise_prob is not None else "N/A"
        log_msg(
            f"Старт train_dpo: mode={mode}, beta={beta}, lr={lr}, batch_size={batch_size}, epochs={epochs}, lambda_min={lambda_min}, lambda_schedule={lambda_schedule}, label_noise_prob={_lnp}, seed={seed}"
        )
        log_msg(f"MAX_PROMPT_LEN={MAX_PROMPT_LEN}, MAX_FULL_LEN={MAX_FULL_LEN}, use_chat_template={use_chat_template}")

        # Начальная валидация (hard)
        policy_model.eval()
        init_dpo_sum = 0.0
        init_kl_sum = 0.0
        init_n = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="init DPO loss", leave=False):
                loss, kl_b = hard_dpo_loss(
                    batch,
                    tokenizer,
                    policy_model,
                    ref_model,
                    device,
                    beta=beta,
                    use_chat_template=use_chat_template,
                )
                n = len(batch["prompt"])
                init_dpo_sum += loss.item() * n
                init_kl_sum += kl_b * n
                init_n += n
        init_dpo = init_dpo_sum / max(1, init_n)
        init_kl = init_kl_sum / max(1, init_n)
        init_nll = eval_pairwise_nll(
            val_loader,
            tokenizer,
            policy_model,
            device,
            beta=1.0,
            use_chat_template=use_chat_template,
            desc="init pairwise NLL",
        )
        init_acc = eval_pairwise_accuracy(
            val_loader,
            tokenizer,
            policy_model,
            device,
            use_chat_template=use_chat_template,
            desc="init pairwise acc",
        )

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
                # progress_epoch = epoch / (epochs - 1) if epochs > 1 else 1.0
                # lambda_label_epoch = max(lambda_min, 1.0 - progress_epoch)
                if epochs > 1:
                    progress_epoch = epoch / (epochs - 1)  # 0 ... 1
                else:
                    progress_epoch = 1.0

                if lambda_schedule == "linear":
                    lambda_label_epoch = 1.0 - (1.0 - lambda_min) * progress_epoch
                else:  # cosine
                    lambda_label_epoch = (
                        lambda_min
                        + (1.0 - lambda_min)
                        * (1.0 + math.cos(math.pi * progress_epoch))
                        / 2.0
                    )

                epoch_loss_kwargs = {**loss_kwargs, "lambda_label": lambda_label_epoch}
                log_msg(f"Epoch {epoch + 1}/{epochs}, lambda_label={lambda_label_epoch:.6f}")

                if lambda_label_epoch < 1.0:
                    train_ds = precompute_p_pred_cached(
                        train_ds,
                        tokenizer,
                        policy_model,
                        ref_model,
                        device=device,
                        beta=beta,
                        use_chat_template=use_chat_template,
                        batch_size=batch_size,
                        collate_fn=train_collate,
                    )
                    train_loader = DataLoader(
                        train_ds,
                        batch_size=batch_size,
                        shuffle=True,
                        collate_fn=train_collate,
                        num_workers=0,
                        generator=g,
                    )

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
                    loss, kl_b = hard_dpo_loss(
                        batch,
                        tokenizer,
                        policy_model,
                        ref_model,
                        device,
                        beta=beta,
                        use_chat_template=use_chat_template,
                    )
                    n = len(batch["prompt"])
                    val_dpo_sum += loss.item() * n
                    val_kl_sum += kl_b * n
                    val_n += n
            val_dpo = val_dpo_sum / max(1, val_n)
            val_kl = val_kl_sum / max(1, val_n)
            val_nll = eval_pairwise_nll(
                val_loader,
                tokenizer,
                policy_model,
                device,
                beta=1.0,
                use_chat_template=use_chat_template,
                desc=f"val NLL ep{epoch+1}",
            )
            val_acc = eval_pairwise_accuracy(
                val_loader,
                tokenizer,
                policy_model,
                device,
                use_chat_template=use_chat_template,
                desc=f"val acc ep{epoch+1}",
            )

            log_msg(f"=== Epoch {epoch+1} ===")
            log_msg(f"validation DPO loss   : {val_dpo:.4f}")
            log_msg(f"validation KL(π||ref) : {val_kl:.4f}")
            log_msg(f"validation pair NLL   : {val_nll:.4f}")
            log_msg(f"validation pair acc   : {val_acc:.4f}")

            if (
                val_distributions_max_batches is not None
                and val_distributions_max_batches > 0
                and len(val_ds) > 0
            ):
                dist = compute_val_delta_distributions(
                    policy_model,
                    ref_model,
                    tokenizer,
                    val_loader,
                    device,
                    use_chat_template=use_chat_template,
                    max_batches=val_distributions_max_batches,
                )
                dt = dist["delta_theta"]
                dr = dist["delta_ref"]
                margin = dist["diff"]

                def _val_dist_stats_line(label: str, arr: np.ndarray) -> str:
                    if arr.size == 0:
                        return f"{label}: (no samples)"
                    mean = float(np.mean(arr))
                    std = float(np.std(arr))
                    med = float(np.median(arr))
                    p5 = float(np.percentile(arr, 5))
                    p95 = float(np.percentile(arr, 95))
                    return (
                        f"{label}: mean={mean:.2f} std={std:.2f} median={med:.2f} "
                        f"p5={p5:.2f} p95={p95:.2f}"
                    )

                log_msg(_val_dist_stats_line("val_delta_theta  ", dt))
                log_msg(_val_dist_stats_line("val_delta_ref    ", dr))
                log_msg(_val_dist_stats_line("val_diff (margin)", margin))

                if use_mlflow and margin.size > 0:
                    mlflow.log_metric(
                        "val_delta_theta_mean", float(np.mean(dt)), step=global_step
                    )
                    mlflow.log_metric(
                        "val_delta_ref_mean", float(np.mean(dr)), step=global_step
                    )
                    mlflow.log_metric(
                        "val_diff_mean", float(np.mean(margin)), step=global_step
                    )
                    mlflow.log_metric(
                        "val_diff_std", float(np.std(margin)), step=global_step
                    )

                npz_path = os.path.join(
                    output_dir, f"val_distributions_epoch{epoch + 1}.npz"
                )
                np.savez_compressed(
                    npz_path,
                    delta_theta=dt,
                    delta_ref=dr,
                    diff=margin,
                )

            if (
                val_kl_mc_max_prompts is not None
                and val_kl_mc_max_prompts > 0
                and len(val_ds) > 0
            ):
                n_mc = min(int(val_kl_mc_max_prompts), len(val_ds))
                mc_prompts = val_ds.select(range(n_mc))["prompt"]
                val_kl_mc = estimate_val_kl_mc(
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
                log_msg(
                    f"validation KL_MC (π‖ref, MC samples from policy, {n_mc} prompts × {val_kl_mc_num_samples}) : {val_kl_mc:.4f}"
                )
                if use_mlflow:
                    mlflow.log_metric("val_kl_mc", val_kl_mc, step=global_step)

            if val_nll < best_val_nll:
                best_val_nll = val_nll
                ckpt_dir = os.path.join(output_dir, "best")
                os.makedirs(ckpt_dir, exist_ok=True)
                tokenizer.save_pretrained(ckpt_dir)
                policy_model.save_pretrained(ckpt_dir)
                log_msg(f"New best NLL {val_nll:.4f} -> checkpoint saved: {ckpt_dir}")
