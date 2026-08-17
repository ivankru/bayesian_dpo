#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backfill validation DPO margin (val_diff) for best checkpoints.

Loads best LoRA weights, runs the same val_diff (margin) metric as train_dpo /
classic_dpo, and writes results to val_diff_best.log next to train.log.

By default runs are processed **sequentially** on a single GPU (--gpu 0).
Pass --gpus 0,1,2,3 to run up to that many jobs in parallel (one checkpoint per GPU).

Run config (dataset, model, batch_size, seed, use_chat_template) is parsed from
train.log when present; CLI flags override.

Examples:
  # sequential (default): one run after another on GPU 0
  python val_diff_batch.py \\
      checkpoints/hsteer/3b/hard_hsteer_lr2e5/epoch_6_seed42 \\
      checkpoints/hsteer/3b/hard_hsteer_lr2e5/epoch_6_seed43

  # sequential on GPU 1
  python val_diff_batch.py -f checkpoints/.../classic --gpu 1 --skip-existing

  # parallel on 4 GPUs (round-robin assignment)
  python val_diff_batch.py -f checkpoints/.../classic --gpus 0,1,2,3 --skip-existing
"""
from __future__ import annotations

import argparse
import gc
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_WORKER_FLAG = "--_worker"
DEFAULT_WORKER_TIMEOUT_SEC = 7200

# Pin GPU before torch init. Parallel multi-GPU workers set CUDA_VISIBLE_DEVICES in subprocess.
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--gpu", type=int, default=0)
_pre.add_argument("--gpus", type=str, default=None)
_pre_args, _ = _pre.parse_known_args()
if _pre_args.gpus:
    _gpu_ids = [int(x.strip()) for x in _pre_args.gpus.split(",") if x.strip()]
    if len(_gpu_ids) == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(_gpu_ids[0])
elif _WORKER_FLAG not in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_pre_args.gpu)

import torch
from torch.utils.data import DataLoader

from config.base_config import USE_CHAT_TEMPLATE
from utils.config import BASE_MODEL_CHOICES, DPO_STEER_SOFT_DATASET_CHOICES
from utils.datasets import (
    build_dpo_datasets,
    build_dpo_datasets_hh_rlhf,
    build_dpo_datasets_ultrafeedback,
    build_openbmb_soft_datasets,
    build_ultrafeedback_score_soft_datasets,
)
from utils.models import load_models_and_tokenizer, resolve_peft_adapter_dir
from utils.training import collate_fn_hard, infer_run_root_from_checkpoint_dir
from utils.val_distributions import compute_val_margin_stats, log_val_diff_stats

REPO_ROOT = Path(__file__).resolve().parent
VAL_DIFF_BEST_LOG = "val_diff_best.log"

ALL_DATASET_CHOICES = tuple(
    sorted(set(DPO_STEER_SOFT_DATASET_CHOICES) | {"helpsteer3", "ultrafeedback_binarized", "hh_rlhf"})
)


@dataclass
class RunConfig:
    dataset: str
    model_name: str
    batch_size: int
    seed: int
    use_chat_template: bool


@dataclass(frozen=True)
class EvalKwargs:
    dataset: Optional[str]
    base_model: Optional[str]
    batch_size: Optional[int]
    use_chat_template: Optional[bool]
    skip_existing: bool
    max_batches: Optional[int]


def parse_train_log(train_log: Path) -> RunConfig:
    """Extract run settings from train.log (train_dpo or classic_dpo format)."""
    text = train_log.read_text(encoding="utf-8", errors="replace")

    model_name: Optional[str] = None
    dataset: Optional[str] = None
    batch_size: Optional[int] = None

    m_classic = re.search(
        r"^Model: (.+?), Dataset: ([^,]+), train size: \d+, val size: (\d+)\s*$",
        text,
        re.M,
    )
    if m_classic:
        model_name = m_classic.group(1).strip()
        dataset = m_classic.group(2).strip()
    else:
        m_model = re.search(r"^Model: (.+?)\s*$", text, re.M)
        if m_model:
            model_name = m_model.group(1).strip()
        m_dataset = re.search(
            r"^Dataset: ([^,]+), train size: \d+, val size: (\d+), batch_size=(\d+)\s*$",
            text,
            re.M,
        )
        if m_dataset:
            dataset = m_dataset.group(1).strip()
            batch_size = int(m_dataset.group(3))

    if batch_size is None:
        m_bs = re.search(r"batch_size=(\d+)", text)
        if m_bs:
            batch_size = int(m_bs.group(1))

    seed = 42
    m_seed = re.search(r"(?:^|\b)seed=(\d+)", text, re.M)
    if m_seed:
        seed = int(m_seed.group(1))

    use_chat_template = USE_CHAT_TEMPLATE
    m_chat = re.search(r"use_chat_template=(True|False)", text)
    if m_chat:
        use_chat_template = m_chat.group(1) == "True"

    missing = []
    if not model_name:
        missing.append("model")
    if not dataset:
        missing.append("dataset")
    if batch_size is None:
        missing.append("batch_size")
    if missing:
        raise ValueError(
            f"Could not parse {', '.join(missing)} from {train_log}. "
            "Pass --dataset, --base-model, and/or --batch-size."
        )

    if dataset not in ALL_DATASET_CHOICES:
        raise ValueError(
            f"Unknown dataset {dataset!r} in {train_log}; "
            f"expected one of {ALL_DATASET_CHOICES}"
        )

    return RunConfig(
        dataset=dataset,
        model_name=model_name,
        batch_size=batch_size,
        seed=seed,
        use_chat_template=use_chat_template,
    )


def load_val_dataset(dataset: str, seed: int):
    if dataset == "helpsteer3":
        _, val_ds = build_dpo_datasets()
    elif dataset == "ultrafeedback_binarized":
        _, val_ds = build_dpo_datasets_ultrafeedback()
    elif dataset == "hh_rlhf":
        _, val_ds = build_dpo_datasets_hh_rlhf()
    elif dataset == "ultrafeedback_soft":
        _, val_ds, _ = build_ultrafeedback_score_soft_datasets(alpha=0.2, seed=seed)
    elif dataset == "openbmb":
        _, val_ds, _ = build_openbmb_soft_datasets(alpha=0.2)
    else:
        raise ValueError(f"Unsupported dataset: {dataset!r}")
    return val_ds


def resolve_run_paths(path: str) -> Tuple[Path, Path, Path]:
    """
    Return (run_root, adapter_dir, train_log_path) for a run dir or checkpoint path.
    """
    adapter_dir = Path(resolve_peft_adapter_dir(path)).resolve()
    run_root = infer_run_root_from_checkpoint_dir(str(adapter_dir))
    if run_root is None:
        run_root = adapter_dir.parent
    else:
        run_root = run_root.resolve()
    train_log = run_root / "train.log"
    return run_root, adapter_dir, train_log


def _has_valid_val_diff_log(path: Path) -> bool:
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8", errors="replace")
    return "val_diff (margin): mean=" in text or "validation val_diff_mean" in text


def eval_val_diff_for_checkpoint(
    run_path: str,
    *,
    dataset: Optional[str] = None,
    base_model: Optional[str] = None,
    batch_size: Optional[int] = None,
    use_chat_template: Optional[bool] = None,
    skip_existing: bool = False,
    max_batches: Optional[int] = None,
    log: Callable[[str], None] = print,
) -> Optional[Path]:
    run_root, adapter_dir, train_log = resolve_run_paths(run_path)
    out_path = run_root / VAL_DIFF_BEST_LOG

    if skip_existing and _has_valid_val_diff_log(out_path):
        log(f"skip existing: {out_path}")
        return out_path

    if train_log.is_file():
        cfg = parse_train_log(train_log)
    else:
        if not dataset or not base_model:
            raise FileNotFoundError(
                f"No train.log at {train_log}; pass --dataset and --base-model."
            )
        cfg = RunConfig(
            dataset=dataset,
            model_name=BASE_MODEL_CHOICES[base_model],
            batch_size=batch_size or 8,
            seed=42,
            use_chat_template=USE_CHAT_TEMPLATE if use_chat_template is None else use_chat_template,
        )

    if dataset is not None:
        cfg.dataset = dataset
    if base_model is not None:
        cfg.model_name = BASE_MODEL_CHOICES[base_model]
    if batch_size is not None:
        cfg.batch_size = batch_size
    if use_chat_template is not None:
        cfg.use_chat_template = use_chat_template

    log(f"=== {run_root.name} ===")
    log(f"checkpoint: {adapter_dir}")
    log(
        f"dataset={cfg.dataset}, model={cfg.model_name}, batch_size={cfg.batch_size}, "
        f"use_chat_template={cfg.use_chat_template}"
    )

    val_ds = load_val_dataset(cfg.dataset, cfg.seed)
    log(f"val size: {len(val_ds)}")

    tokenizer, policy_model, ref_model, device = load_models_and_tokenizer(
        cfg.model_name,
        use_lora=True,
        resume_from=str(adapter_dir),
        share_ref_with_policy=True,
    )
    policy_model.eval()
    if hasattr(ref_model, "eval"):
        ref_model.eval()

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_fn_hard,
        num_workers=0,
    )

    stats = compute_val_margin_stats(
        policy_model,
        ref_model,
        tokenizer,
        val_loader,
        device,
        use_chat_template=cfg.use_chat_template,
        max_batches=max_batches,
    )

    lines: List[str] = []

    def capture(msg: str) -> None:
        lines.append(msg)
        log(msg)

    capture("")
    capture("=== val_diff (best checkpoint) ===")
    capture(f"computed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    capture(f"checkpoint: {adapter_dir}")
    capture(f"dataset: {cfg.dataset}, model: {cfg.model_name}, val size: {len(val_ds)}")
    capture(f"batch_size: {cfg.batch_size}, use_chat_template: {cfg.use_chat_template}")
    if max_batches is not None:
        capture(f"max_batches: {max_batches} (subset)")
    log_val_diff_stats(stats, capture)

    content = "\n".join(lines) + "\n"
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    tmp_path.write_text(content, encoding="utf-8")
    tmp_path.replace(out_path)
    log(f"wrote {out_path}")

    del policy_model, ref_model, tokenizer, val_loader, val_ds
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return out_path


def collect_paths_from_folder(folder: Path) -> List[str]:
    """Collect run dirs with best/ adapters (same layout as alpaca_eval_batch)."""
    folder = folder.resolve()
    paths: List[str] = []
    for child in sorted(folder.iterdir()):
        if not child.is_dir():
            continue
        if (child / "best" / "adapter_config.json").is_file():
            paths.append(str(child))
        elif (child / "adapter_config.json").is_file():
            paths.append(str(child))
    return paths


def dedupe_paths(paths: Sequence[str]) -> List[str]:
    seen = set()
    unique_paths: List[str] = []
    for p in paths:
        key = str(Path(p).expanduser().resolve())
        if key not in seen:
            seen.add(key)
            unique_paths.append(p)
    return unique_paths


def _forwarded_argv(args: argparse.Namespace) -> List[str]:
    """CLI flags passed to per-job worker subprocesses (no paths / folder / gpus)."""
    out: List[str] = []
    if args.dataset is not None:
        out.extend(["--dataset", args.dataset])
    if args.base_model is not None:
        out.extend(["--base-model", args.base_model])
    if args.batch_size is not None:
        out.extend(["--batch-size", str(args.batch_size)])
    if args.use_chat_template is not None:
        out.extend(
            ["--use-chat-template" if args.use_chat_template else "--no-use-chat-template"]
        )
    if args.skip_existing:
        out.append("--skip-existing")
    if args.max_batches is not None:
        out.extend(["--max-batches", str(args.max_batches)])
    if getattr(args, "worker_timeout", None) is not None:
        out.extend(["--worker-timeout", str(args.worker_timeout)])
    return out


def run_worker_subprocess(
    path: str,
    gpu: int,
    forwarded: Sequence[str],
    worker_timeout_sec: int,
) -> Tuple[str, int]:
    """Run one checkpoint in an isolated process on the given GPU."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "val_diff_batch.py"),
        _WORKER_FLAG,
        "--gpu",
        str(gpu),
        path,
        *forwarded,
    ]
    print(f"[GPU {gpu}] start: {path}", flush=True)
    started = time.perf_counter()
    try:
        proc = subprocess.run(cmd, env=env, timeout=worker_timeout_sec)
        elapsed = time.perf_counter() - started
        print(
            f"[GPU {gpu}] done in {elapsed:.0f}s exit={proc.returncode}: {path}",
            flush=True,
        )
        return path, proc.returncode
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - started
        print(
            f"[GPU {gpu}] TIMEOUT after {elapsed:.0f}s (limit={worker_timeout_sec}s): {path}",
            file=sys.stderr,
            flush=True,
        )
        return path, 124


def run_sequential(
    paths: Sequence[str],
    eval_kwargs: EvalKwargs,
) -> int:
    errors = 0
    for i, path in enumerate(paths, start=1):
        print(f"\n[{i}/{len(paths)}] {path}", flush=True)
        try:
            eval_val_diff_for_checkpoint(
                path,
                dataset=eval_kwargs.dataset,
                base_model=eval_kwargs.base_model,
                batch_size=eval_kwargs.batch_size,
                use_chat_template=eval_kwargs.use_chat_template,
                skip_existing=eval_kwargs.skip_existing,
                max_batches=eval_kwargs.max_batches,
                log=lambda msg: print(msg, flush=True),
            )
        except Exception as exc:
            errors += 1
            print(f"ERROR: {path}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    return errors


def run_parallel(
    paths: Sequence[str],
    gpus: Sequence[int],
    forwarded: Sequence[str],
    worker_timeout_sec: int,
) -> int:
    max_workers = min(len(gpus), len(paths))
    errors = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                run_worker_subprocess,
                path,
                gpus[i % len(gpus)],
                forwarded,
                worker_timeout_sec,
            ): path
            for i, path in enumerate(paths)
        }
        for fut in as_completed(futures):
            path = futures[fut]
            try:
                _, code = fut.result()
            except Exception as exc:
                errors += 1
                print(
                    f"ERROR: {path}: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            if code != 0:
                errors += 1
                print(f"ERROR: {path}: worker exited with code {code}", file=sys.stderr, flush=True)
    return errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute val_diff (margin) on best checkpoints and write val_diff_best.log. "
            "Default: sequential on --gpu (one run after another)."
        )
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="Run directories (with best/) or checkpoint paths.",
    )
    parser.add_argument(
        "-f",
        "--folder",
        type=str,
        default=None,
        help="Discover run subdirs under this folder (each with best/).",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU id for sequential mode (default: 0). Ignored when --gpus is set.",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help=(
            "Comma-separated GPU ids for parallel jobs (e.g. 0,1,2,3). "
            "If omitted, runs are processed sequentially on --gpu."
        ),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=ALL_DATASET_CHOICES,
        help="Override dataset from train.log.",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        choices=tuple(BASE_MODEL_CHOICES.keys()),
        help="Override base model from train.log.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override eval batch size from train.log (default: parsed or 8).",
    )
    parser.add_argument(
        "--use-chat-template",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override use_chat_template from train.log.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=f"Skip runs that already have a valid {VAL_DIFF_BEST_LOG}.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Evaluate only the first N val batches (debug / smoke test).",
    )
    parser.add_argument(
        "--worker-timeout",
        type=int,
        default=DEFAULT_WORKER_TIMEOUT_SEC,
        help=f"Per-run timeout in parallel mode (default: {DEFAULT_WORKER_TIMEOUT_SEC}s).",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    is_worker = _WORKER_FLAG in argv
    if is_worker:
        argv = [a for a in argv if a != _WORKER_FLAG]

    parser = build_parser()
    args = parser.parse_args(argv)

    paths: List[str] = list(args.paths)
    if args.folder:
        paths.extend(collect_paths_from_folder(Path(args.folder)))
    if not paths:
        parser.error("Provide at least one path or --folder.")

    unique_paths = dedupe_paths(paths)
    eval_kwargs = EvalKwargs(
        dataset=args.dataset,
        base_model=args.base_model,
        batch_size=args.batch_size,
        use_chat_template=args.use_chat_template,
        skip_existing=args.skip_existing,
        max_batches=args.max_batches,
    )

    # Worker subprocess: always one path, sequential eval inside the process.
    if is_worker:
        if len(unique_paths) != 1:
            parser.error("Worker mode expects exactly one path.")
        try:
            eval_val_diff_for_checkpoint(
                unique_paths[0],
                dataset=eval_kwargs.dataset,
                base_model=eval_kwargs.base_model,
                batch_size=eval_kwargs.batch_size,
                use_chat_template=eval_kwargs.use_chat_template,
                skip_existing=eval_kwargs.skip_existing,
                max_batches=eval_kwargs.max_batches,
                log=lambda msg: print(msg, flush=True),
            )
        except Exception as exc:
            print(
                f"ERROR: {unique_paths[0]}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            return 1
        return 0

    gpus: Optional[List[int]] = None
    if args.gpus:
        gpus = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]
        if not gpus:
            parser.error("--gpus must list at least one GPU id.")

    print(f"Runs: {len(unique_paths)}", flush=True)
    if gpus and len(gpus) > 1:
        print(
            f"Mode: parallel on GPUs {gpus} (worker_timeout={args.worker_timeout}s)",
            flush=True,
        )
        errors = run_parallel(
            unique_paths,
            gpus,
            _forwarded_argv(args),
            args.worker_timeout,
        )
    else:
        gpu = gpus[0] if gpus else args.gpu
        print(f"Mode: sequential on GPU {gpu}", flush=True)
        errors = run_sequential(unique_paths, eval_kwargs)

    if errors:
        print(f"\nFinished with {errors} error(s) out of {len(unique_paths)} run(s).", flush=True)
        return 1
    print(f"\nDone: {len(unique_paths)} run(s).", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
