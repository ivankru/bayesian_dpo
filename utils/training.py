# -*- coding: utf-8 -*-
"""
Общий цикл обучения одной эпохи для DPO и универсальная функция train_dpo (режимы hard / soft / bayes).
"""
import json
import math
import os
import re
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import mlflow
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import Dataset
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from config.base_config import (
    CAPABILITY_EVAL_BATCH_SIZE,
    CAPABILITY_EVAL_LIMIT,
    CAPABILITY_EVAL_MAX_NEW_TOKENS,
    CAPABILITY_EVAL_MAX_PROMPT_TOKENS,
    LOG_INTERVAL,
    LR_ALIGN_LOG_INTERVAL,
    MAX_FULL_LEN,
    MAX_PROMPT_LEN,
    P_PRED_TARGET_TEMPERATURE,
    USE_CHAT_TEMPLATE,
    VAL_ENTROPY_FORWARD_CHUNK_SIZE,
    VAL_ENTROPY_MAX_NEW_TOKENS,
    VAL_ENTROPY_MAX_PROMPTS,
    VAL_ENTROPY_NUM_SAMPLES,
    VAL_ENTROPY_PROMPT_BATCH_SIZE,
    VAL_KL_MC_NUM_SAMPLES,
)
from utils.datasets import precompute_p_pred_cached, precompute_p_pred_teacher
from utils.loss import hard_dpo_loss, soft_dpo_loss
from utils.metrics import (
    EvalRow,
    aggregate_anchor_alignment_window,
    build_ref_cache_metadata,
    estimate_val_response_entropy,
    estimate_val_kl_mc,
    eval_pairwise_accuracy,
    eval_pairwise_nll,
    format_anchor_alignment_log,
    format_capability_retention_log_lines,
    load_eval_rows,
    load_ref_texts_cache_if_compatible,
    log_mlflow_capability_metrics,
    run_retention_eval_pair,
    save_ref_texts_cache,
)
from utils.val_distributions import compute_val_delta_distributions

DPO_MODE_CHOICES = ("hard", "soft", "bayes")

# MC forward KL(π‖ref) по сэмплам с π_θ; min(N, len(val)) промптов. 0 в val_kl_mc_max_prompts — отключить.
DEFAULT_VAL_KL_MC_MAX_PROMPTS = 256

# Soft/Bayes-ADPO: половинная эпоха — валидация и обновление lambda / p_pred_cached.
ULTRAFB_MID_EPOCH_DATASETS = frozenset(
    {"openbmb", "ultrafeedback_binarized", "ultrafeedback_soft", "hh_rlhf"}
)


def infer_run_root_from_checkpoint_dir(resume_checkpoint_dir: str) -> Optional[Path]:
    """
    Корень прогона (родительская папка с train.log): .../best, .../epochs/epoch_XXX, иначе parent.
    """
    p = Path(resume_checkpoint_dir).expanduser()
    try:
        p = p.resolve()
    except OSError:
        p = Path(resume_checkpoint_dir).expanduser()
    if not p.exists():
        return None
    name = p.name
    if name == "best":
        return p.parent
    if p.parent.name == "epochs" and re.match(r"^epoch_\d{3}$", name):
        return p.parent.parent
    # Корень прогона (output_dir): адаптер лежит в best/, train.log рядом
    if (p / "best" / "adapter_config.json").is_file():
        return p
    return p.parent


def slice_train_log_lines_before_resume_start_epoch(
    lines: List[str],
    mode: str,
    resume_start_epoch_1based: int,
) -> List[str]:
    """
    Строки train.log до начала эпохи resume_start_epoch_1based (1-based).
    Soft/bayes: обрезка перед строкой «=== Epoch S ===» или «=== Epoch S/E ===» (новый формат).
    Hard: обрезка перед первой строкой «[epoch S]» (обучение эпохи S; формат не менялся).
    """
    s = int(resume_start_epoch_1based)
    if s <= 1:
        return []
    if mode in ("soft", "bayes"):
        # Принимаем оба формата: старый "=== Epoch S ===" и новый "=== Epoch S/E ===".
        boundary = re.compile(rf"^\s*=== Epoch {s}(?:/\d+)? ===\s*$")
        for i, line in enumerate(lines):
            if boundary.match(line):
                return lines[:i]
        return []
    if mode == "hard":
        boundary = re.compile(rf"^\[epoch {int(s)}\]")
        for i, line in enumerate(lines):
            if boundary.match(line):
                return lines[:i]
        return []
    return []


def _lambda_label_at_progress(
    progress: float, lambda_min: float, lambda_schedule: str
) -> float:
    """progress in [0, 1]: как в основном цикле эпох (linear / cosine)."""
    progress = max(0.0, min(1.0, float(progress)))
    if lambda_schedule == "linear":
        return 1.0 - (1.0 - lambda_min) * progress
    return lambda_min + (1.0 - lambda_min) * (1.0 + math.cos(math.pi * progress)) / 2.0


def _lambda_schedule_progress(
    epoch_idx_0: int,
    epochs: int,
    lambda_full_epochs: int,
    mid_frac: float,
) -> float:
    """
    Доля [0, 1] для аргумента _lambda_label_at_progress. Формула симметрична
    «эпоха-к-эпохе» между режимами k=0 и k>0: при одинаковых (epochs, epoch_idx_0)
    начало каждой эпохи даёт один и тот же progress в обоих режимах, если
    трактовать первую эпоху при k=0 как warmup-по-меткам, а при k>0 — как явный
    warmup длиной k эпох. Это нужно для двух вещей:
      (a) честного сравнения k=0 vs k>0 при одинаковом total epochs;
      (b) чтобы resume сразу после эпохи k (resume_start_epoch_1based == k+1)
          стартовал с λ<1 с первого же шага, а не тратил ещё одну эпоху на λ=1.

    lambda_full_epochs == 0 (без warmup-анкора):
        progress = (epoch_idx_0 + mid_frac) / (epochs - 1).
        Первая эпоха (idx=0, mid=0): progress=0 → λ=1 — неявный warmup-по-меткам;
        последняя (idx=epochs-1, mid=0): progress=1 → λ=lambda_min.

    lambda_full_epochs == k > 0 (warmup-анкор: эпохи 1..k только метки, в конце
    эпохи k фиксируется p_pred_teacher):
        warmup 1..k → progress = 0 → λ = 1;
        хвост k+1..epochs (decay_epochs = epochs - k эпох):
            rel = (epoch_idx_0 - k) + mid_frac;
            progress = (rel + 1) / decay_epochs.
        Начало первой хвостовой (rel=0): progress = 1/decay → λ<1 сразу.
        Конец последней (rel=decay-1, mid=0 на ней же): progress = 1 → λ=lambda_min.

    Проверка симметрии «эпоха-к-эпохе» на epochs=5:
        k=0: epoch starts → [0, 0.25, 0.5, 0.75, 1.0]
        k=1: epoch starts → [warmup=0, 0.25, 0.5, 0.75, 1.0]
        k=2: epoch starts → [warmup=0, warmup=0, 1/3, 2/3, 1.0]
    Расписание хвоста укладывается в decay_epochs эпох так, что первая хвостовая
    уже делает активный шаг (а не дублирует warmup).

    Edge cases:
      - decay_epochs <= 0 (k >= epochs): хвоста нет, всё warmup → progress=0.
      - decay_epochs == 1 (единственная хвостовая эпоха): progress = 1 сразу
        (λ=lambda_min на всей этой эпохе).
      - epochs <= 1: тривиально, 0.5 / 1.0 по mid_frac.

    Расписание детерминировано по (epoch_idx_0, epochs, k, mid_frac) — resume с
    любого resume_start_epoch_1based даёт ту же λ, что и непрерывный прогон при
    тех же параметрах. Для resume ровно на первой хвостовой эпохе
    (resume_start_epoch_1based == k+1) p_pred_teacher пересчитывается по
    ЗАГРУЖЕННЫМ весам (см. train_dpo) — эквивалентно концу эпохи k в непрерывном
    прогоне, и λ<1 включается с первого шага после загрузки.

    mid_frac: 0 — начало эпохи; 0.5 — середина (для mid-epoch валидации).
    """
    if epochs <= 1:
        return 0.5 if mid_frac > 0 else 1.0
    if lambda_full_epochs <= 0:
        return min(1.0, (epoch_idx_0 + mid_frac) / (epochs - 1))
    f = int(lambda_full_epochs)
    if epoch_idx_0 < f:
        return 0.0
    decay_epochs = epochs - f
    if decay_epochs <= 0:
        return 0.0
    if decay_epochs == 1:
        return 1.0
    rel = (epoch_idx_0 - f) + mid_frac
    return min(1.0, max(0.0, (rel + 1) / decay_epochs))


def _val_resp_entropy_vocab_nats_max(tokenizer, policy_model) -> Tuple[int, float]:
    """
    V и log(V) в натах: верхняя граница энтропии одного шага при равномерном softmax
    по полному словарю (как в estimate_val_response_entropy).
    """
    cfg = getattr(policy_model, "config", None)
    v = getattr(cfg, "vocab_size", None) if cfg is not None else None
    if not isinstance(v, int) or v <= 0:
        v = getattr(tokenizer, "vocab_size", None)
    if not isinstance(v, int) or v <= 0:
        v = len(tokenizer)
    v = int(v)
    log_v = math.log(float(v)) if v > 1 else float("nan")
    return v, log_v


def _log_val_response_entropy_two_lines(
    log_msg: Callable[..., None],
    ent_stats: Dict[str, float],
    tokenizer,
    policy_model,
    *,
    l_tokens: int,
    n_prompts: int,
    num_samples: int,
) -> None:
    """Две строки в лог: абсолютные наты (с max = log V) и те же статистики в % от max."""
    v, log_v = _val_resp_entropy_vocab_nats_max(tokenizer, policy_model)
    hdr = (
        "validation response entropy "
        f"(L={l_tokens}, {n_prompts} prompts × {num_samples})"
    )
    m = float(ent_stats["mean"])
    med = float(ent_stats["median"])
    p10 = float(ent_stats["p10"])
    p90 = float(ent_stats["p90"])
    if math.isfinite(log_v) and log_v > 0:
        inv_pct = 100.0 / log_v
        log_msg(
            f"{hdr} — abs (nats): mean={m:.4f} median={med:.4f} p10={p10:.4f} p90={p90:.4f} "
            f"(max uniform = log V = {log_v:.4f}, V={v})"
        )
        log_msg(
            f"{hdr} — % of max: mean={m * inv_pct:.2f}% median={med * inv_pct:.2f}% "
            f"p10={p10 * inv_pct:.2f}% p90={p90 * inv_pct:.2f}%"
        )
    else:
        log_msg(
            f"{hdr} : mean={m:.4f} median={med:.4f} p10={p10:.4f} p90={p90:.4f}"
        )


def _fmt_seconds(seconds: float) -> str:
    s = max(0.0, float(seconds))
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = s - h * 3600 - m * 60
    return f"{h}h {m:02d}m {sec:05.2f}s"


def _make_shuffled_train_loader(
    ds,
    collate_fn,
    batch_size: int,
    generator: torch.Generator,
) -> DataLoader:
    """Тренировочный DataLoader с shuffle через переданный torch.Generator.

    Единая точка сборки: все перезапуски после precompute_p_pred_* используют
    один и тот же `generator` (продолжает серию рандома от seed), num_workers=0
    для детерминированного порядка батчей.
    """
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        generator=generator,
    )


def _make_ordered_loader(
    ds,
    collate_fn,
    batch_size: int,
) -> DataLoader:
    """DataLoader без shuffle для валидации и для фиксированных split'ов эпохи."""
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )


def _build_loss_spec(
    mode: str,
    beta: float,
    use_chat_template: bool,
    p_pred_target_temperature: float,
) -> Tuple[Callable, Callable, Dict[str, Any], str]:
    """По режиму возвращает (train_collate, train_loss_fn, loss_kwargs, mode_label).

    Чистая функция; mode уже должен быть провалидирован против DPO_MODE_CHOICES.
    """
    if mode == "hard":
        return (
            collate_fn_hard,
            hard_dpo_loss,
            {"beta": beta, "use_chat_template": use_chat_template},
            "Hard DPO",
        )
    use_bayes = mode == "bayes"
    return (
        collate_fn_soft,
        soft_dpo_loss,
        {
            "beta": beta,
            "use_bayes": use_bayes,
            "use_chat_template": use_chat_template,
            "p_pred_target_temperature": p_pred_target_temperature,
        },
        "Bayes DPO" if use_bayes else "Soft DPO",
    )


def _epoch_lambda_and_loss_kw(
    g0: int,
    epochs: int,
    lambda_full_epochs: int,
    lambda_min: float,
    lambda_schedule: str,
    has_teacher_column: bool,
    base_loss_kwargs: Dict[str, Any],
) -> Tuple[Dict[str, Any], float, bool, float]:
    """Собирает loss_kwargs для начала эпохи (g0, 1-based = g0+1) soft/bayes-режима.

    Возвращает (epoch_loss_kw, lambda_label_epoch, has_teacher_anchor, teacher_blend_w).
    Используется вне hard-ветки.
    """
    progress_epoch = _lambda_schedule_progress(
        g0, epochs, lambda_full_epochs, 0.0
    )
    lambda_label_epoch = _lambda_label_at_progress(
        progress_epoch, lambda_min, lambda_schedule
    )
    has_teacher_anchor = lambda_full_epochs > 0 and has_teacher_column
    teacher_blend_w = 0.5 if has_teacher_anchor else 0.0
    epoch_loss_kw = {
        **base_loss_kwargs,
        "lambda_label": lambda_label_epoch,
        "p_pred_teacher_blend": teacher_blend_w,
    }
    return epoch_loss_kw, lambda_label_epoch, has_teacher_anchor, teacher_blend_w


def _validate_train_dpo_args(
    mode: str,
    epochs: int,
    lambda_min: float,
    lambda_schedule: str,
    lambda_full_epochs: int,
    p_pred_target_temperature: float,
    resume_start_epoch_1based: int,
    resume_rewarmup_steps: int,
    resume_rewarmup_lr_floor: float,
) -> None:
    """Валидация аргументов train_dpo. Единая точка отказа с понятным сообщением."""
    if mode not in DPO_MODE_CHOICES:
        raise ValueError(f"mode должен быть один из {DPO_MODE_CHOICES}, получено: {mode!r}")
    if not 0.0 <= lambda_min <= 1.0:
        raise ValueError(f"lambda_min must be in [0, 1], got {lambda_min!r}")
    if lambda_schedule not in ("linear", "cosine"):
        raise ValueError(
            f"lambda_schedule must be one of ('linear', 'cosine'), got {lambda_schedule!r}"
        )
    if lambda_full_epochs < 0:
        raise ValueError(f"lambda_full_epochs must be >= 0, got {lambda_full_epochs!r}")
    if p_pred_target_temperature <= 0:
        raise ValueError(
            f"p_pred_target_temperature must be > 0, got {p_pred_target_temperature!r}"
        )
    if resume_start_epoch_1based < 1:
        raise ValueError(
            f"resume_start_epoch_1based must be >= 1, got {resume_start_epoch_1based!r}"
        )
    if resume_rewarmup_steps < 0:
        raise ValueError(
            f"resume_rewarmup_steps must be >= 0, got {resume_rewarmup_steps!r}"
        )
    if not 0.0 <= resume_rewarmup_lr_floor <= 1.0:
        raise ValueError(
            f"resume_rewarmup_lr_floor must be in [0, 1], got {resume_rewarmup_lr_floor!r}"
        )
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs!r}")
    if resume_start_epoch_1based > epochs:
        raise ValueError(
            f"resume_start_epoch_1based={resume_start_epoch_1based} must be <= epochs={epochs}"
        )


def _gpu_peak_memory_gb(device: torch.device) -> Optional[float]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    try:
        idx = device.index if device.index is not None else torch.cuda.current_device()
        bytes_peak = torch.cuda.max_memory_allocated(idx)
    except Exception:
        return None
    return float(bytes_peak) / (1024.0**3)


def _reset_cuda_peak_memory_stats(device: torch.device) -> None:
    if device.type != "cuda" or not torch.cuda.is_available():
        return
    try:
        idx = device.index if device.index is not None else torch.cuda.current_device()
        torch.cuda.reset_peak_memory_stats(idx)
    except Exception:
        pass


def _fmt_mem_gb(mem_gb: Optional[float]) -> str:
    if mem_gb is None:
        return "n/a"
    return f"{mem_gb:.2f} GB"


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
    if examples and "p_pred_teacher" in examples[0]:
        out["p_pred_teacher"] = [e["p_pred_teacher"] for e in examples]
    return out


def train_one_epoch_dpo(
    train_loader_box: List[DataLoader],
    tokenizer,
    policy_model,
    ref_model,
    device: str,
    loss_fn: Callable[..., Any],
    optimizer,
    scheduler,
    epoch_1based: int,
    global_step: int,
    loss_kw: Dict[str, Any],
    log=print,
    use_mlflow: bool = False,
    mid_epoch_hook: Optional[Callable[[int], None]] = None,
) -> int:
    """
    Одна эпоха DPO. loss_fn(..., **loss_kw) возвращает (loss, kl_approx).
    loss_kw изменяется in-place (например lambda_label после mid_epoch_hook).

    train_loader_box:
      - длиной 1 — единственный DataLoader на всю эпоху (mid_epoch_hook игнорируется);
      - длиной 2 — [first_half_loader, second_half_placeholder]: цикл обходит первый
        лоадер, потом один раз вызывает mid_epoch_hook(global_step) (который обязан
        положить второй DataLoader в train_loader_box[1]), и дальше обходит второй.
        Эта схема даёт эпохе 100%-ное покрытие (непересекающиеся выборки).

    epoch_1based: номер эпохи (1-based) только для строк лога train.
    """
    policy_model.train()
    running_loss = 0.0
    running_kl = 0.0
    log_interval = int(LOG_INTERVAL)
    # Строка lr и агрегированные align-метрики с той же периодичностью (накопление за весь интервал).
    lr_align_log_interval = int(LR_ALIGN_LOG_INTERVAL)
    align_gap_parts: List[np.ndarray] = []
    align_ts_parts: List[np.ndarray] = []

    def flush_align_log() -> None:
        if not align_ts_parts:
            return
        align_m = aggregate_anchor_alignment_window(align_gap_parts, align_ts_parts)
        log(format_anchor_alignment_log(align_m))
        if use_mlflow:
            for k, v in align_m.items():
                if v == v:  # not NaN
                    mlflow.log_metric(f"train_{k}", v, step=global_step)
        align_gap_parts.clear()
        align_ts_parts.clear()

    def process_batch(batch) -> None:
        nonlocal global_step, running_loss, running_kl
        optimizer.zero_grad(set_to_none=True)
        out = loss_fn(batch, tokenizer, policy_model, ref_model, device, **loss_kw)
        if len(out) == 3:
            loss, kl_batch, soft_diag = out
            if isinstance(soft_diag, dict):
                ts = soft_diag.get("target_shift")
                if ts is not None and ts.size:
                    align_ts_parts.append(ts)
                ga = soft_diag.get("gap_abs")
                if ga is not None and ga.size:
                    align_gap_parts.append(ga)
        else:
            loss, kl_batch = out
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        running_kl += kl_batch
        global_step += 1

        if global_step % lr_align_log_interval == 0:
            lr_cur = optimizer.param_groups[0]["lr"]
            log(f"[epoch {epoch_1based} step {global_step}] lr={lr_cur:.2e}")
            if use_mlflow:
                mlflow.log_metric("lr", lr_cur, step=global_step)
            flush_align_log()
        if global_step % log_interval == 0:
            n = log_interval
            log(
                f"[epoch {epoch_1based} step {global_step}] "
                f"train_loss={running_loss / n:.4f} "
                f"train_logp_gap_mean={running_kl / n:.4f}"
            )
            if use_mlflow:
                mlflow.log_metric("train_loss", running_loss / n, step=global_step)
                mlflow.log_metric("train_logp_gap_mean", running_kl / n, step=global_step)
            running_loss = 0.0
            running_kl = 0.0

    split_mid = mid_epoch_hook is not None and len(train_loader_box) == 2

    if not split_mid:
        loader = train_loader_box[0]
        for batch in loader:
            process_batch(batch)
        flush_align_log()
        return global_step

    first_loader = train_loader_box[0]
    for batch in first_loader:
        process_batch(batch)

    mid_epoch_hook(global_step)

    second_loader = train_loader_box[1]
    if second_loader is None:
        raise RuntimeError(
            "mid_epoch_hook должен положить второй DataLoader в train_loader_box[1]"
        )
    for batch in second_loader:
        process_batch(batch)

    flush_align_log()
    return global_step


def train_dpo(
    train_ds: Dataset,
    val_ds: Dataset,
    tokenizer,
    policy_model,
    ref_model,
    device: str | torch.device,
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
    lambda_full_epochs: int = 0,
    p_pred_target_temperature: float = P_PRED_TARGET_TEMPERATURE,
    seed: int = 42,
    label_noise_prob: Optional[float] = None,
    use_chat_template: bool = USE_CHAT_TEMPLATE,
    log=print,
    use_mlflow: bool = False,
    mlflow_experiment: str = "bayesian_dpo",
    mlflow_run_name: Optional[str] = None,
    mlflow_tracking_uri: Optional[str] = None,
    val_kl_mc_max_prompts: int = DEFAULT_VAL_KL_MC_MAX_PROMPTS,
    val_kl_mc_num_samples: int = VAL_KL_MC_NUM_SAMPLES,
    val_kl_mc_max_new_tokens: int = 128,
    val_kl_mc_prompt_batch_size: int = 6,
    val_entropy_max_prompts: int = VAL_ENTROPY_MAX_PROMPTS,
    val_entropy_num_samples: int = VAL_ENTROPY_NUM_SAMPLES,
    val_entropy_max_new_tokens: int = VAL_ENTROPY_MAX_NEW_TOKENS,
    # Узкие дефолты под одну A100 80GB вместе с ref + KL_MC: шире — риск OOM на forward по полной длине.
    val_entropy_prompt_batch_size: int = VAL_ENTROPY_PROMPT_BATCH_SIZE,
    val_entropy_forward_chunk_size: int = VAL_ENTROPY_FORWARD_CHUNK_SIZE,
    val_distributions_max_batches: Optional[int] = None,
    capability_eval_dir: Optional[str] = None,
    capability_eval_limit: Optional[int] = CAPABILITY_EVAL_LIMIT,
    capability_eval_max_new_tokens: int = CAPABILITY_EVAL_MAX_NEW_TOKENS,
    capability_eval_batch_size: int = CAPABILITY_EVAL_BATCH_SIZE,
    capability_eval_max_prompt_tokens: int = CAPABILITY_EVAL_MAX_PROMPT_TOKENS,
    capability_ref_cache_path: Optional[str] = None,
    resume_start_epoch_1based: int = 1,
    resume_checkpoint_dir: Optional[str] = None,
    resume_rewarmup_steps: int = 50,
    resume_rewarmup_lr_floor: float = 0.0,
):
    """
    Универсальный цикл DPO: hard, soft или bayes.

    mode: "hard" — train и val в формате chosen/rejected, loss = hard_dpo_loss.
          "soft" — train в формате resp1, resp2, p, p_bayes; val в формате chosen/rejected; train loss = soft_dpo_loss(use_bayes=False).
          "bayes" — как soft, но train loss = soft_dpo_loss(use_bayes=True).

    epochs: общее число эпох в плане (для λ и linear LR по шкале 1..epochs). Цикл обучения —
        эпохи resume_start_epoch_1based, …, epochs (включительно). Старт с начала: resume_start_epoch_1based=1.
    val_ds всегда в формате chosen/rejected; валидация по hard DPO loss, NLL, accuracy.
    num_training_steps_override: для soft/bayes можно задать число шагов (например по hard train size) для выравнивания LR schedule.
    lambda_min: для soft/bayes — нижняя граница lambda_label по эпохам (смешивание с p_pred); при 1.0 поведение как раньше.
    lambda_full_epochs: для soft/bayes — k (1-based): эпохи 1..k только метки (λ=1); в конце эпохи k фиксируется
        p_pred_teacher (σ(beta*diff) без T). С эпохи k+1 λ<1 сразу, по расписанию на хвосте (decay=epochs-k
        эпох): progress первой хвостовой = 1/decay, последней = 1. Это согласовано по эпохам с режимом
        lambda_full_epochs=0 (эпоха n при k=0 ≡ эпоха n при k>0 в смысле progress, если первую эпоху при
        k=0 считать неявным warmup-по-меткам). Полезное следствие: при resume с
        resume_start_epoch_1based == k+1 (загрузка весов конца эпохи k) λ<1 включается с первого шага
        после загрузки — без лишней эпохи на λ=1. Пока в train_ds есть p_pred_teacher, в p_pred при λ<1
        всегда w=0.5: 0.5*p_pred_teacher + 0.5*σ((beta*diff)/T).
        0 — без warmup-анкора: расписание λ от первой эпохи, кэш p_pred_cached пересчитывается каждый шаг.
    p_pred_target_temperature: T>0 для σ((beta*diff)/T) в якорном режиме (см. utils.loss.soft_dpo_loss); при lambda_full_epochs=0 не используется.
    seed: фиксирует shuffle train DataLoader (torch.Generator + num_workers=0).
    label_noise_prob: вероятность шума меток при сборке soft train (--label-noise-prob); для hard не задаётся (в логе N/A).
    use_chat_template: если True, get_logps использует tokenizer.apply_chat_template (Qwen-Instruct); иначе plain prompt\\nresponse (дефолт: config.base_config.USE_CHAT_TEMPLATE).
    use_mlflow: логировать параметры, метрики и train.log в MLflow (tracking URI из mlflow_tracking_uri или окружения по умолчанию).
    val_kl_mc_max_prompts: если >0 (по умолчанию DEFAULT_VAL_KL_MC_MAX_PROMPTS), в конце каждой эпохи считается MC-оценка KL(π‖ref) по сэмплам π_θ
          на первых min(N, len(val)) промптах val (см. utils.metrics.estimate_val_kl_mc); лог: val_kl_mc, метрика MLflow при use_mlflow. 0 — отключить.
    val_kl_mc_num_samples: число независимых генераций на промпт для MC.
    val_entropy_max_prompts: если >0, в конце каждой эпохи считается средняя токенная энтропия ответов policy
          по первым min(L, T_resp) токенам и агрегируется по prompt'ам; 0 — отключить.
    val_entropy_num_samples: число независимых генераций на prompt для оценки энтропии.
    val_entropy_max_new_tokens: L, ограничение на первые токены ответа для энтропии.
    val_entropy_prompt_batch_size: сколько val-промптов за один generate (× num_samples параллельных цепочек).
    val_entropy_forward_chunk_size: микробатч для полного forward по сгенерированным seq (снижает пик VRAM).
    val_distributions_max_batches: если задано (>0), после основных val-метрик считаются распределения
        delta_theta, delta_ref, diff на первых N батчах val; лог, MLflow, np.savez_compressed в output_dir.
    capability_eval_dir: если задан (каталог с knowledge/*.jsonl и reasoning/*.jsonl), на каждой валидации
        и в начале (epoch init) считается удержание возможностей: ref vs policy по gold; лог + JSON в output_dir;
        ответы ref кэшируются после первой генерации. MLflow: val_cap_*.
    capability_eval_limit: обрезка числа примеров (первые N в порядке файлов).
    capability_eval_max_new_tokens / capability_eval_batch_size / capability_eval_max_prompt_tokens: генерация.
    capability_ref_cache_path: путь к JSON-кэшу ref ответов для retention.
        Если не задан, используется {capability_eval_dir}/ref_cache/<safe_model_name>_ref_texts.json.
    resume_rewarmup_steps: при resume (g0_start>0) первые N шагов после возобновления ещё раз
        плавно разгоняют lr (дополнительный множитель поверх основного расписания) — от
        resume_rewarmup_lr_floor до 1.0 линейно. Оптимизатор пересоздаётся «с нуля» при
        каждом resume (moments=0), поэтому полный lr на первом же шаге даёт большие
        неустойчивые апдейты — этот ре-warmup сглаживает эффект. 0 отключает ре-warmup.
    resume_rewarmup_lr_floor: минимальная доля lr в момент возобновления (0.0 — старт
        буквально с нуля за N шагов; 0.05 — с 5% сразу).
    resume_start_epoch_1based: первая эпоха этого запуска (1-based, как в логах и epochs/epoch_XXX).
        Должно быть 1 <= resume_start_epoch_1based <= epochs. Веса из чекпоинта — после эпохи N-1
        (например после epoch_003 задайте 4).         При N>1 первая валидация — полный val как после эпохи (N-1): тот же заголовок/теги,
        что в конце эпохи; best_val_nll инициализируется этим NLL (склейка с предыдущим train.log).
    resume_checkpoint_dir: путь к чекпоинту как при --resume (best или epochs/epoch_XXX); рядом ищется train.log.
        При resume_start_epoch_1based>1 и непустом префиксе до эпохи S строки дописываются в начало train.log
        в output_dir (перед логом текущего запуска).
        Для soft/bayes с lambda_full_epochs=k>0: если resume_start_epoch_1based==k+1 (первый хвостовой шаг),
        до цикла эпох вызывается precompute_p_pred_teacher по загруженным весам — как фиксация учителя
        в конце эпохи k в непрерывном прогоне (при отсутствии столбца p_pred_teacher в train_ds).
    Для soft/bayes и датасетов openbmb, ultrafeedback_binarized, ultrafeedback_soft, hh_rlhf при epochs>=2:
        после первой половины батчей эпохи — валидация с меткой «0.5», «1.5», …; затем lambda_label
        по расписанию для позиции k.5 (с учётом lambda_full_epochs) и при необходимости пересчёт p_pred_cached
        для второй половины эпохи; в якорном режиме (p_pred_teacher) пересчёт кэша не делается.
    """
    _validate_train_dpo_args(
        mode=mode,
        epochs=epochs,
        lambda_min=lambda_min,
        lambda_schedule=lambda_schedule,
        lambda_full_epochs=lambda_full_epochs,
        p_pred_target_temperature=p_pred_target_temperature,
        resume_start_epoch_1based=resume_start_epoch_1based,
        resume_rewarmup_steps=resume_rewarmup_steps,
        resume_rewarmup_lr_floor=resume_rewarmup_lr_floor,
    )
    g0_start = resume_start_epoch_1based - 1

    if not isinstance(device, torch.device):
        device = torch.device(device)

    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "train.log")

    prior_train_log_lines: List[str] = []
    prior_train_log_src: Optional[str] = None
    if (
        resume_start_epoch_1based > 1
        and resume_checkpoint_dir
        and str(resume_checkpoint_dir).strip()
    ):
        root = infer_run_root_from_checkpoint_dir(str(resume_checkpoint_dir))
        if root is not None:
            cand = root / "train.log"
            prior_train_log_src = str(cand)
            if cand.is_file():
                try:
                    with open(cand, "r", encoding="utf-8", errors="replace") as rf:
                        raw_lines = rf.readlines()
                    prior_train_log_lines = slice_train_log_lines_before_resume_start_epoch(
                        raw_lines, mode, resume_start_epoch_1based
                    )
                    if raw_lines and not prior_train_log_lines:
                        print(
                            "train.log: не удалось вырезать историю до "
                            f"эпохи {resume_start_epoch_1based} "
                            f"({mode}: ожидается граница эпохи в логе) в {prior_train_log_src}; "
                            "пропуск переноса.",
                            file=sys.stderr,
                        )
                except OSError as e:
                    print(
                        f"train.log: не удалось прочитать {cand}: {e}",
                        file=sys.stderr,
                    )

    def log_msg(msg: str) -> None:
        log(msg)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    if prior_train_log_lines:
        sep = (
            f"\n--- train.log resumed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
            f"(epochs 1..{resume_start_epoch_1based - 1} из предыдущего train.log: {prior_train_log_src}) ---\n"
        )
        with open(log_path, "w", encoding="utf-8") as wf:
            wf.writelines(prior_train_log_lines)
            wf.write(sep)

    mlflow_param_dict: Dict[str, Any] = {
        "mode": mode,
        "beta": beta,
        "lr": lr,
        "batch_size": batch_size,
        "epochs": epochs,
        "lambda_min": lambda_min,
        "lambda_schedule": lambda_schedule,
        "lambda_full_epochs": lambda_full_epochs,
        "p_pred_target_temperature": p_pred_target_temperature,
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
        "val_entropy_max_prompts": val_entropy_max_prompts,
        "val_entropy_num_samples": val_entropy_num_samples,
        "val_entropy_max_new_tokens": val_entropy_max_new_tokens,
        "val_entropy_prompt_batch_size": val_entropy_prompt_batch_size,
        "val_entropy_forward_chunk_size": val_entropy_forward_chunk_size,
        "val_distributions_max_batches": val_distributions_max_batches,
        "capability_eval_dir": capability_eval_dir,
        "capability_eval_limit": capability_eval_limit,
        "capability_eval_max_new_tokens": capability_eval_max_new_tokens,
        "capability_eval_batch_size": capability_eval_batch_size,
        "capability_eval_max_prompt_tokens": capability_eval_max_prompt_tokens,
        "resume_start_epoch_1based": resume_start_epoch_1based,
        "resume_checkpoint_dir": resume_checkpoint_dir,
        "resume_rewarmup_steps": resume_rewarmup_steps,
        "resume_rewarmup_lr_floor": resume_rewarmup_lr_floor,
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
        train_collate, train_loss_fn, loss_kwargs, mode_label = _build_loss_spec(
            mode=mode,
            beta=beta,
            use_chat_template=use_chat_template,
            p_pred_target_temperature=p_pred_target_temperature,
        )

        g = torch.Generator()
        g.manual_seed(seed)

        train_loader = _make_shuffled_train_loader(
            train_ds, train_collate, batch_size, g
        )
        val_loader = _make_ordered_loader(val_ds, collate_fn_hard, batch_size)

        cap_rows: Optional[List[EvalRow]] = None
        cap_ref_cache: List[Optional[List[str]]] = [None]
        cap_ref_cache_path_obj: Optional[Path] = None
        cap_ref_cache_meta: Optional[Dict[str, Any]] = None

        def _safe_tokenizer_name_or_path() -> str:
            v = getattr(tokenizer, "name_or_path", None)
            if isinstance(v, str) and v.strip():
                return v
            return "N/A"

        def _safe_ref_model_revision() -> str:
            cfg = getattr(ref_model, "config", None)
            if cfg is None:
                return "N/A"
            rev = getattr(cfg, "_commit_hash", None)
            if isinstance(rev, str) and rev.strip():
                return rev
            rev2 = getattr(cfg, "revision", None)
            if isinstance(rev2, str) and rev2.strip():
                return rev2
            return "N/A"

        if capability_eval_dir:
            eval_p = Path(capability_eval_dir).expanduser().resolve()
            if eval_p.is_dir():
                try:
                    cap_rows = load_eval_rows(eval_p)
                    if capability_eval_limit is not None and capability_eval_limit > 0:
                        cap_rows = cap_rows[: int(capability_eval_limit)]
                    if not cap_rows:
                        log_msg(
                            f"capability_eval_dir={eval_p}: примеры не найдены, retention пропущен."
                        )
                        cap_rows = None
                    else:
                        safe_model = (model_name or "unknown_model").replace("/", "__")
                        if capability_ref_cache_path:
                            cap_ref_cache_path_obj = Path(capability_ref_cache_path).expanduser().resolve()
                        else:
                            cap_ref_cache_path_obj = (
                                eval_p / "ref_cache" / f"{safe_model}_ref_texts.json"
                            )
                        cap_ref_cache_meta = build_ref_cache_metadata(
                            cap_rows,
                            model_name=model_name or "N/A",
                            tokenizer_name_or_path=_safe_tokenizer_name_or_path(),
                            ref_model_revision=_safe_ref_model_revision(),
                            max_new_tokens=capability_eval_max_new_tokens,
                            max_prompt_tokens=capability_eval_max_prompt_tokens,
                            use_chat_template=use_chat_template,
                        )
                        loaded, reason = load_ref_texts_cache_if_compatible(
                            cap_ref_cache_path_obj, cap_ref_cache_meta
                        )
                        if loaded is not None:
                            cap_ref_cache[0] = loaded
                            log_msg(
                                f"Capability retention: loaded ref cache ({len(loaded)} responses) from {cap_ref_cache_path_obj} [{reason}]"
                            )
                        else:
                            log_msg(
                                f"Capability retention: ref cache miss at {cap_ref_cache_path_obj} [{reason}]"
                            )
                        log_msg(
                            f"Capability retention: {len(cap_rows)} примеров из {eval_p} "
                            f"(ref кэшируется после первой генерации)."
                        )
                except Exception as e:
                    log_msg(f"Capability retention: ошибка загрузки eval: {e}")
                    cap_rows = None
            else:
                log_msg(f"Capability retention: каталог не найден {eval_p}, пропуск.")

        def _epoch_tag_for_files(epoch_display: str) -> str:
            return epoch_display.replace(".", "_")

        def _run_capability_retention(epoch_display: str, step_m: int) -> None:
            if cap_rows is None:
                return
            try:
                desc_ref = (
                    f"cap_ret ref [ep {epoch_display}]"
                    if cap_ref_cache[0] is None
                    else None
                )
                summary, cap_ref_cache[0] = run_retention_eval_pair(
                    tokenizer,
                    ref_model,
                    policy_model,
                    device,
                    cap_rows,
                    cap_ref_cache[0],
                    capability_eval_max_new_tokens,
                    capability_eval_batch_size,
                    capability_eval_max_prompt_tokens,
                    desc_ref=desc_ref,
                    desc_pol=f"cap_ret policy [ep {epoch_display}]",
                )
            except Exception as e:
                log_msg(f"Capability retention: ошибка генерации/скоринга: {e}")
                return
            if (
                cap_ref_cache_path_obj is not None
                and cap_ref_cache_meta is not None
                and cap_ref_cache[0] is not None
            ):
                try:
                    save_ref_texts_cache(
                        cap_ref_cache_path_obj, cap_ref_cache_meta, cap_ref_cache[0]
                    )
                except Exception as err:
                    log_msg(
                        f"Capability retention: failed to save ref cache {cap_ref_cache_path_obj}: {err}"
                    )
                else:
                    log_msg(
                        f"Capability retention: ref cache saved to {cap_ref_cache_path_obj}"
                    )
            for line in format_capability_retention_log_lines(summary, epoch_display):
                log_msg(line)
            tag = _epoch_tag_for_files(epoch_display)
            cap_ret_dir = os.path.join(output_dir, "capability_retention")
            os.makedirs(cap_ret_dir, exist_ok=True)
            cap_json = os.path.join(
                cap_ret_dir, f"capability_retention_epoch{tag}.json"
            )
            try:
                with open(cap_json, "w", encoding="utf-8") as f:
                    json.dump(summary, f, ensure_ascii=False, indent=2)
            except OSError as err:
                log_msg(f"Capability retention: не удалось записать {cap_json}: {err}")
            if use_mlflow:
                log_mlflow_capability_metrics(
                    summary,
                    step_m,
                    lambda n, v, s: mlflow.log_metric(n, v, step=s),
                )

        def _run_validation(
            epoch_display: str,
            mlflow_step: Optional[int] = None,
            training_seconds: Optional[float] = None,
        ) -> float:
            """epoch_display: '1', '0.5', '1.5', ... для логов и имён артефактов. Возвращает val NLL."""
            tag = _epoch_tag_for_files(epoch_display)
            step_m = global_step if mlflow_step is None else mlflow_step
            t_validation_total_start = perf_counter()
            if device.type == "cuda" and torch.cuda.is_available():
                try:
                    idx = device.index if device.index is not None else torch.cuda.current_device()
                    torch.cuda.reset_peak_memory_stats(idx)
                except Exception:
                    pass
            policy_model.eval()
            val_dpo_sum = 0.0
            val_kl_sum = 0.0
            val_n = 0
            t_validation_core_start = perf_counter()
            with torch.no_grad():
                for batch in tqdm(
                    val_loader, desc=f"val DPO [ep {epoch_display}]", leave=False
                ):
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
                desc=f"val NLL [ep {epoch_display}]",
            )
            val_acc = eval_pairwise_accuracy(
                val_loader,
                tokenizer,
                policy_model,
                device,
                use_chat_template=use_chat_template,
                desc=f"val acc [ep {epoch_display}]",
            )
            validation_core_seconds = perf_counter() - t_validation_core_start

            log_msg("")
            log_msg(f"=== Validation, epoch {epoch_display} ===")
            log_msg("")
            log_msg(f"validation DPO loss   : {val_dpo:.4f}")
            log_msg(f"validation logp_gap_mean : {val_kl:.4f}")
            log_msg(f"validation pair NLL   : {val_nll:.4f}")
            log_msg(f"validation pair acc   : {100 * val_acc:.2f}%")

            if use_mlflow:
                mlflow.log_metric("val_dpo_loss", val_dpo, step=step_m)
                mlflow.log_metric("logp_gap_mean", val_kl, step=step_m)
                mlflow.log_metric("val_pair_nll", val_nll, step=step_m)
                mlflow.log_metric("val_pair_acc", val_acc, step=step_m)
                try:
                    ef = float(epoch_display)
                except ValueError:
                    ef = float("nan")
                if not math.isnan(ef):
                    mlflow.log_metric("epoch_float", ef, step=step_m)

            if (
                val_distributions_max_batches is not None
                and val_distributions_max_batches > 0
                and len(val_ds) > 0
            ):
                # Обёрнуто в try/except: это «nice to have» диагностика
                # (распределения margin'ов), OOM/NaN здесь не должен валить
                # всю эпоху — основные метрики уже посчитаны выше.
                try:
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
                            "val_delta_theta_mean", float(np.mean(dt)), step=step_m
                        )
                        mlflow.log_metric(
                            "val_delta_ref_mean", float(np.mean(dr)), step=step_m
                        )
                        mlflow.log_metric(
                            "val_diff_mean", float(np.mean(margin)), step=step_m
                        )
                        mlflow.log_metric(
                            "val_diff_std", float(np.std(margin)), step=step_m
                        )

                    npz_path = os.path.join(
                        output_dir, f"val_distributions_epoch{tag}.npz"
                    )
                    np.savez_compressed(
                        npz_path,
                        delta_theta=dt,
                        delta_ref=dr,
                        diff=margin,
                    )
                except Exception as e:
                    log_msg(
                        "validation delta distributions: FAILED "
                        f"({type(e).__name__}: {e}); continuing without margin-distribution metrics"
                    )

            if val_kl_mc_max_prompts > 0 and len(val_ds) > 0:
                n_mc = min(int(val_kl_mc_max_prompts), len(val_ds))
                log_msg(
                    f"validation KL_MC: computing (first {n_mc} val prompts × {val_kl_mc_num_samples} samples)..."
                )
                t_mc_kl_start = perf_counter()
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
                    log_msg(
                        f"validation KL_MC (π‖ref, MC samples from policy, {n_mc} prompts × "
                        f"{val_kl_mc_num_samples}): per_seq={val_kl_mc_per_seq:.4f}, "
                        f"per_token={val_kl_mc_per_token:.6f} (total_tokens={n_tokens_mc})"
                    )
                    if use_mlflow:
                        # per-seq оставляем под старым именем для совместимости графиков;
                        # per-token — основная метрика для кросс-прогонного сравнения.
                        mlflow.log_metric("val_kl_mc", val_kl_mc_per_seq, step=step_m)
                        mlflow.log_metric(
                            "val_kl_mc_per_seq", val_kl_mc_per_seq, step=step_m
                        )
                        mlflow.log_metric(
                            "val_kl_mc_per_token", val_kl_mc_per_token, step=step_m
                        )
                except Exception as e:
                    log_msg(
                        f"validation KL_MC: FAILED ({type(e).__name__}: {e}); continuing without val_kl_mc metric"
                    )
                mc_kl_seconds = perf_counter() - t_mc_kl_start
            elif val_kl_mc_max_prompts <= 0:
                log_msg("validation KL_MC: skipped (val_kl_mc_max_prompts<=0).")
                mc_kl_seconds = 0.0
            elif len(val_ds) == 0:
                log_msg("validation KL_MC: skipped (empty val_ds).")
                mc_kl_seconds = 0.0

            if val_entropy_max_prompts > 0 and len(val_ds) > 0:
                n_ent = min(int(val_entropy_max_prompts), len(val_ds))
                log_msg(
                    "validation response entropy: computing "
                    f"(first {n_ent} val prompts × {val_entropy_num_samples} samples, "
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
                    _log_val_response_entropy_two_lines(
                        log_msg,
                        ent_stats,
                        tokenizer,
                        policy_model,
                        l_tokens=val_entropy_max_new_tokens,
                        n_prompts=n_ent,
                        num_samples=val_entropy_num_samples,
                    )
                    if use_mlflow:
                        mlflow.log_metric("val_resp_entropy_mean", ent_stats["mean"], step=step_m)
                        mlflow.log_metric(
                            "val_resp_entropy_median", ent_stats["median"], step=step_m
                        )
                        mlflow.log_metric("val_resp_entropy_p10", ent_stats["p10"], step=step_m)
                        mlflow.log_metric("val_resp_entropy_p90", ent_stats["p90"], step=step_m)
                except Exception as e:
                    log_msg(
                        "validation response entropy: FAILED "
                        f"({type(e).__name__}: {e}); continuing without response entropy metric"
                    )
            elif val_entropy_max_prompts <= 0:
                log_msg("validation response entropy: skipped (val_entropy_max_prompts<=0).")
            elif len(val_ds) == 0:
                log_msg("validation response entropy: skipped (empty val_ds).")

            t_capability_start = perf_counter()
            _run_capability_retention(epoch_display, step_m)
            capability_seconds = perf_counter() - t_capability_start
            validation_total_seconds = perf_counter() - t_validation_total_start
            validation_peak_mem_gb = _gpu_peak_memory_gb(device)

            timing_parts = [f"validation={_fmt_seconds(validation_core_seconds)}"]
            if training_seconds is not None:
                timing_parts.insert(0, f"training={_fmt_seconds(training_seconds)}")
            timing_parts.append(f"mc_kl={_fmt_seconds(mc_kl_seconds)}")
            timing_parts.append(
                f"capability_retention={_fmt_seconds(capability_seconds)}"
            )
            timing_parts.append(f"validation_total={_fmt_seconds(validation_total_seconds)}")
            log_msg("timings: " + ", ".join(timing_parts))
            # Пик на обучении логируется сразу после train_one_epoch_dpo (отдельная строка).
            log_msg(f"gpu_mem_peak: validation={_fmt_mem_gb(validation_peak_mem_gb)}")
            log_msg("")
            return val_nll

        use_mid_epoch_val = (
            mode != "hard"
            and epochs >= 2
            and dataset_name in ULTRAFB_MID_EPOCH_DATASETS
        )

        actual_steps_per_epoch = len(train_loader)
        total_actual_steps = epochs * actual_steps_per_epoch
        if num_training_steps_override is not None:
            num_training_steps = num_training_steps_override
        else:
            num_training_steps = total_actual_steps

        steps_per_schedule_epoch = max(1, num_training_steps // max(1, epochs))
        if num_training_steps % epochs != 0:
            log_msg(
                "Предупреждение: num_training_steps не делится на epochs "
                "нацело; стартовый сдвиг LR: "
                f"{g0_start} * floor({num_training_steps}/{epochs})."
            )
        start_global_step = g0_start * steps_per_schedule_epoch
        if start_global_step >= num_training_steps:
            raise ValueError(
                f"resume_start_epoch_1based={resume_start_epoch_1based} даёт "
                f"start_global_step={start_global_step} >= num_training_steps={num_training_steps}"
            )

        # Явно логируем плановое и фактическое число шагов, чтобы LR-расписание
        # визуально соответствовало реальной длине прогона (частая ловушка при
        # num_training_steps_override по hard_train_size: если soft-train больше,
        # scheduler уедет в lr=0 раньше конца обучения; если меньше — lr не дойдёт до 0).
        steps_delta = total_actual_steps - num_training_steps
        steps_delta_pct = (
            100.0 * steps_delta / max(1, num_training_steps)
        )
        log_msg(
            f"LR schedule: num_training_steps={num_training_steps}"
            + (" (override)" if num_training_steps_override is not None else " (auto: epochs*len(train_loader))")
        )
        log_msg(
            f"Actual steps: epochs={epochs} × len(train_loader)={actual_steps_per_epoch} "
            f"= {total_actual_steps}; delta(actual-planned)={steps_delta:+d} "
            f"({steps_delta_pct:+.2f}%)"
        )
        if steps_delta > 0:
            log_msg(
                "Предупреждение: фактических шагов БОЛЬШЕ планового num_training_steps — "
                f"последние {steps_delta} шагов пройдут с lr=0 (линейный спад уже достиг нуля)."
            )
        elif steps_delta < 0:
            final_lr_frac = 1.0 - (total_actual_steps / max(1, num_training_steps))
            log_msg(
                "Предупреждение: фактических шагов МЕНЬШЕ планового num_training_steps — "
                f"к концу прогона lr не опустится до нуля (останется ≈ {final_lr_frac:.2%} от lr max)."
            )

        if use_mlflow:
            mlflow.log_param("num_training_steps", num_training_steps)
            mlflow.log_param("total_actual_steps", total_actual_steps)
            mlflow.log_param("actual_steps_per_epoch", actual_steps_per_epoch)
            mlflow.log_param("steps_delta_actual_minus_planned", steps_delta)

        log_msg(f"=== {mode_label} ===")
        log_msg(f"Run started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log_msg(f"Model: {model_name or 'N/A'}, Dataset: {dataset_name or 'N/A'}, train size: {len(train_ds)}, val size: {len(val_ds)}")
        _lnp = label_noise_prob if label_noise_prob is not None else "N/A"
        log_msg(
            f"Старт train_dpo: mode={mode}, beta={beta}, lr={lr}, batch_size={batch_size}, "
            f"epochs_total={epochs}, epochs_this_run={epochs - g0_start}, "
            f"resume_start_epoch_1based={resume_start_epoch_1based}, "
            f"lambda_min={lambda_min}, lambda_schedule={lambda_schedule}, lambda_full_epochs={lambda_full_epochs}, "
            f"p_pred_target_temperature={p_pred_target_temperature}, label_noise_prob={_lnp}, seed={seed}"
        )
        log_msg(f"MAX_PROMPT_LEN={MAX_PROMPT_LEN}, MAX_FULL_LEN={MAX_FULL_LEN}, use_chat_template={use_chat_template}")
        log_msg(
            f"val_KL_MC: max_prompts={val_kl_mc_max_prompts}, samples_per_prompt={val_kl_mc_num_samples}, "
            f"max_new_tokens={val_kl_mc_max_new_tokens} (max_prompts=0 disables MC-KL after each epoch val)"
        )
        log_msg(
            f"val_response_entropy: max_prompts={val_entropy_max_prompts}, "
            f"samples_per_prompt={val_entropy_num_samples}, max_new_tokens={val_entropy_max_new_tokens}, "
            f"prompt_batch_size={val_entropy_prompt_batch_size}, "
            f"forward_chunk_size={val_entropy_forward_chunk_size} "
            "(max_prompts=0 disables response entropy after each epoch val)"
        )

        # Начальная валидация: при --start-epoch>1 — как конец эпохи (start-1), полный val как после эпохи.
        pre_val_epoch_done = g0_start  # завершённые эпохи 1..g0_start; веса = после эпохи pre_val_epoch_done

        checkpoint_val_nll: Optional[float] = None
        policy_model.eval()
        if g0_start == 0:
            log_msg("")
            log_msg("=== Initial (before training), epoch 0 ===")
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
            log_msg("")
            log_msg(f"validation DPO loss   : {init_dpo:.4f}")
            log_msg(f"validation logp_gap_mean : {init_kl:.4f}")
            log_msg(f"validation pair NLL   : {init_nll:.4f}")
            log_msg(f"validation pair acc   : {100 * init_acc:.2f}%")
            if val_entropy_max_prompts > 0 and len(val_ds) > 0:
                n_ent = min(int(val_entropy_max_prompts), len(val_ds))
                log_msg(
                    "validation response entropy: computing "
                    f"(first {n_ent} val prompts × {val_entropy_num_samples} samples, "
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
                    _log_val_response_entropy_two_lines(
                        log_msg,
                        ent_stats,
                        tokenizer,
                        policy_model,
                        l_tokens=val_entropy_max_new_tokens,
                        n_prompts=n_ent,
                        num_samples=val_entropy_num_samples,
                    )
                    if use_mlflow:
                        step_m = start_global_step
                        mlflow.log_metric(
                            "val_resp_entropy_mean", ent_stats["mean"], step=step_m
                        )
                        mlflow.log_metric(
                            "val_resp_entropy_median", ent_stats["median"], step=step_m
                        )
                        mlflow.log_metric(
                            "val_resp_entropy_p10", ent_stats["p10"], step=step_m
                        )
                        mlflow.log_metric(
                            "val_resp_entropy_p90", ent_stats["p90"], step=step_m
                        )
                except Exception as e:
                    log_msg(
                        "validation response entropy: FAILED "
                        f"({type(e).__name__}: {e}); continuing without response entropy metric"
                    )
            elif val_entropy_max_prompts <= 0:
                log_msg("validation response entropy: skipped (val_entropy_max_prompts<=0).")
            elif len(val_ds) == 0:
                log_msg("validation response entropy: skipped (empty val_ds).")
            _run_capability_retention("init", 0)
        else:
            checkpoint_val_nll = _run_validation(
                str(pre_val_epoch_done),
                mlflow_step=start_global_step,
            )

        optimizer = torch.optim.AdamW(policy_model.parameters(), lr=lr)
        base_warmup_steps = max(10, num_training_steps // 20)
        do_resume_rewarmup = start_global_step > 0 and resume_rewarmup_steps > 0

        def _lr_lambda(current_step: int) -> float:
            # Основное расписание: линейный warmup до base_warmup_steps, затем линейный
            # спад до 0 к num_training_steps — идентично get_linear_schedule_with_warmup.
            if current_step < base_warmup_steps:
                base = current_step / max(1, base_warmup_steps)
            else:
                progress = (current_step - base_warmup_steps) / max(
                    1, num_training_steps - base_warmup_steps
                )
                base = max(0.0, 1.0 - progress)
            # Поверх основного — локальный ре-warmup: в течение первых
            # resume_rewarmup_steps шагов после возобновления дополнительный
            # множитель линейно растёт от resume_rewarmup_lr_floor до 1.0.
            # Цель — дать AdamW собрать first/second moments без гигантских
            # обновлений на первом же шаге (optimizer state не сохраняется).
            if do_resume_rewarmup:
                rel = current_step - start_global_step
                if 0 <= rel < resume_rewarmup_steps:
                    ramp = resume_rewarmup_lr_floor + (
                        1.0 - resume_rewarmup_lr_floor
                    ) * (rel / resume_rewarmup_steps)
                else:
                    ramp = 1.0
            else:
                ramp = 1.0
            return base * ramp

        scheduler = LambdaLR(optimizer, lr_lambda=_lr_lambda)
        for _ in range(start_global_step):
            scheduler.step()

        if do_resume_rewarmup:
            log_msg(
                f"Resume rewarmup: первые {resume_rewarmup_steps} шагов после "
                f"start_global_step={start_global_step} — дополнительный линейный "
                f"множитель lr от {resume_rewarmup_lr_floor:g} до 1.0 поверх основного "
                "расписания (компенсирует обнулённые moments AdamW при resume)."
            )

        # Якорный режим: в непрерывном прогоне p_pred_teacher появляется в конце эпохи k (lambda_full_epochs).
        # Старт с эпохи (k+1) без этого столбца — считаем учителя по текущим весам (конец эпохи k), как там же.
        if (
            mode != "hard"
            and lambda_full_epochs > 0
            and resume_start_epoch_1based == lambda_full_epochs + 1
            and "p_pred_teacher" not in train_ds.column_names
        ):
            _reset_cuda_peak_memory_stats(device)
            train_ds = precompute_p_pred_teacher(
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
            log_msg(
                "gpu_mem_peak: "
                f"precompute_p_pred_teacher={_fmt_mem_gb(_gpu_peak_memory_gb(device))}"
            )
            if "p_pred_cached" in train_ds.column_names:
                train_ds = train_ds.remove_columns(["p_pred_cached"])
            train_loader = _make_shuffled_train_loader(
                train_ds, train_collate, batch_size, g
            )
            log_msg(
                f"p_pred_teacher: зафиксирован по загруженным весам "
                f"(resume_start_epoch_1based={resume_start_epoch_1based} == lambda_full_epochs+1={lambda_full_epochs + 1}; "
                f"эквивалентно концу эпохи {lambda_full_epochs} в непрерывном прогоне). "
                f"Дальше как в непрерывном прогоне: на всех хвостовых эпохах (λ<1) "
                f"p_pred = 0.5·p_teacher + 0.5·σ((β·diff)/T)."
            )

        best_val_nll = (
            float("inf") if checkpoint_val_nll is None else float(checkpoint_val_nll)
        )
        global_step = start_global_step
        for g0 in range(g0_start, epochs):
            log_msg("")
            log_msg(f"=== Epoch {g0 + 1}/{epochs} ===")
            if mode == "hard":
                epoch_loss_kw = dict(loss_kwargs)
                mid_hook: Optional[Callable[[int], None]] = None
            else:
                (
                    epoch_loss_kw,
                    lambda_label_epoch,
                    has_teacher_anchor,
                    teacher_blend_w,
                ) = _epoch_lambda_and_loss_kw(
                    g0=g0,
                    epochs=epochs,
                    lambda_full_epochs=lambda_full_epochs,
                    lambda_min=lambda_min,
                    lambda_schedule=lambda_schedule,
                    has_teacher_column="p_pred_teacher" in train_ds.column_names,
                    base_loss_kwargs=loss_kwargs,
                )
                log_msg(
                    f"[epoch {g0 + 1}/{epochs}] lambda_label={lambda_label_epoch:.6f}"
                    + (" (teacher_anchor)" if has_teacher_anchor else "")
                    + (
                        f", p_pred_teacher_blend={teacher_blend_w}"
                        if has_teacher_anchor
                        else ""
                    )
                )

                if lambda_label_epoch < 1.0 and not has_teacher_anchor:
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
                    train_loader = _make_shuffled_train_loader(
                        train_ds, train_collate, batch_size, g
                    )

                mid_hook = None
                if use_mid_epoch_val and len(train_loader) >= 2:
                    # Делим эпоху на две непересекающиеся половины по примерам:
                    # генерируем одну пермутацию индексов train_ds, режем её в точке
                    # first_count_examples = n_first_batches * batch_size, и дальше
                    # перечисляем select(idx_first) и select(idx_second). Порядок
                    # строк в train_ds сохраняется при add_column/remove_columns/map
                    # в datasets, поэтому idx_second остаётся валидной выборкой
                    # «оставшихся» примеров даже если mid_hook пересчитает
                    # p_pred_cached. Итог: за эпоху каждый пример используется ровно
                    # один раз, без пропусков и дубликатов.
                    n_examples = len(train_ds)
                    perm_t = torch.randperm(n_examples, generator=g).tolist()
                    num_batches_total = (n_examples + batch_size - 1) // batch_size
                    n_first_batches = num_batches_total // 2
                    first_count_examples = min(
                        n_first_batches * batch_size, n_examples
                    )
                    idx_first = perm_t[:first_count_examples]
                    idx_second = perm_t[first_count_examples:]

                    first_ds = train_ds.select(idx_first)
                    first_loader_local = _make_ordered_loader(
                        first_ds, train_collate, batch_size
                    )
                    train_loader_box: List[Optional[DataLoader]] = [
                        first_loader_local,
                        None,
                    ]

                    def mid_hook(gs: int) -> None:
                        nonlocal train_ds, epoch_loss_kw
                        mid_epoch_display = f"{g0 + 0.5:.1f}"
                        # Mid-epoch валидация — диагностика динамики, не критерий
                        # save-best: падение здесь (OOM на KL_MC, сеть отвалилась
                        # при cap-retention и т.п.) не должно ронять оставшиеся
                        # полэпохи обучения. Ошибка логируется, обучение продолжается.
                        try:
                            _run_validation(mid_epoch_display, mlflow_step=gs)
                        except Exception as e:
                            log_msg(
                                f"[epoch {mid_epoch_display}/{epochs}] mid-epoch validation FAILED "
                                f"({type(e).__name__}: {e}); продолжаем вторую половину эпохи "
                                "без mid-epoch метрик."
                            )
                        prog_m = _lambda_schedule_progress(
                            g0, epochs, lambda_full_epochs, 0.5
                        )
                        lambda_mid = _lambda_label_at_progress(
                            prog_m, lambda_min, lambda_schedule
                        )
                        log_msg(
                            f"[epoch {mid_epoch_display}/{epochs}] "
                            f"lambda_label={lambda_mid:.6f} (2nd half)"
                        )
                        teacher_here = (
                            lambda_full_epochs > 0
                            and "p_pred_teacher" in train_ds.column_names
                        )
                        tw = 0.5 if teacher_here else 0.0
                        epoch_loss_kw.clear()
                        epoch_loss_kw.update(
                            {
                                **loss_kwargs,
                                "lambda_label": lambda_mid,
                                "p_pred_teacher_blend": tw,
                            }
                        )
                        if lambda_mid < 1.0 and not teacher_here:
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
                        second_ds = train_ds.select(idx_second)
                        train_loader_box[1] = _make_ordered_loader(
                            second_ds, train_collate, batch_size
                        )
                        policy_model.train()
                else:
                    train_loader_box = [train_loader]

            if mode == "hard":
                train_loader_box = [train_loader]

            if device.type == "cuda" and torch.cuda.is_available():
                try:
                    idx = device.index if device.index is not None else torch.cuda.current_device()
                    torch.cuda.reset_peak_memory_stats(idx)
                except Exception:
                    pass
            t_training_start = perf_counter()
            global_step = train_one_epoch_dpo(
                train_loader_box,
                tokenizer,
                policy_model,
                ref_model,
                device,
                train_loss_fn,
                optimizer,
                scheduler,
                g0 + 1,
                global_step,
                loss_kw=epoch_loss_kw,
                log=log_msg,
                use_mlflow=use_mlflow,
                mid_epoch_hook=mid_hook if mode != "hard" else None,
            )
            training_seconds = perf_counter() - t_training_start
            training_peak_mem_gb = _gpu_peak_memory_gb(device)
            log_msg(f"gpu_mem_peak: training={_fmt_mem_gb(training_peak_mem_gb)}")
            # В split-режиме train_loader_box держит две непересекающиеся половины
            # текущей эпохи, поэтому забирать из него «основной» train_loader нельзя.
            # Переменная train_loader будет пересоздана в начале следующей эпохи при
            # любой ветке, которая реально по ней итерирует (precompute_p_pred_cached
            # в начале эпохи или teacher_anchor-переход ниже), а до тех пор она
            # используется только для оценки len(train_loader) (число батчей полного
            # датасета), которое не меняется в mid_hook.

            if (
                mode != "hard"
                and lambda_full_epochs > 0
                and (g0 + 1) == lambda_full_epochs
            ):
                _reset_cuda_peak_memory_stats(device)
                train_ds = precompute_p_pred_teacher(
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
                log_msg(
                    "gpu_mem_peak: "
                    f"precompute_p_pred_teacher={_fmt_mem_gb(_gpu_peak_memory_gb(device))}"
                )
                if "p_pred_cached" in train_ds.column_names:
                    train_ds = train_ds.remove_columns(["p_pred_cached"])
                train_loader = _make_shuffled_train_loader(
                    train_ds, train_collate, batch_size, g
                )
                log_msg(
                    f"p_pred_teacher: зафиксирован в конце эпохи {g0 + 1} (1-based k={lambda_full_epochs}); "
                    f"с эпохи {g0 + 2} λ<1 по расписанию; при λ<1 p_pred_teacher_blend=0.5 на всех хвостовых шагах."
                )

            policy_model.eval()
            val_nll = _run_validation(
                str(g0 + 1),
                training_seconds=training_seconds,
            )

            if val_nll < best_val_nll:
                best_val_nll = val_nll
                ckpt_dir = os.path.join(output_dir, "best")
                os.makedirs(ckpt_dir, exist_ok=True)
                tokenizer.save_pretrained(ckpt_dir)
                policy_model.save_pretrained(ckpt_dir)
                log_msg(f"New best NLL {val_nll:.4f} -> checkpoint saved: {ckpt_dir}")

            epoch_ckpt_dir = os.path.join(
                output_dir, "epochs", f"epoch_{g0 + 1:03d}"
            )
            os.makedirs(epoch_ckpt_dir, exist_ok=True)
            tokenizer.save_pretrained(epoch_ckpt_dir)
            policy_model.save_pretrained(epoch_ckpt_dir)
            log_msg(
                f"[epoch {g0 + 1}/{epochs}] checkpoint (full epoch only): {epoch_ckpt_dir}"
            )
