#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Google IFEval (Instruction Following Eval): verifiable instruction-following metric
without an LLM judge — only rules from the official benchmark.

Source code and data (Apache-2.0):
  https://github.com/google-research/google-research/tree/master/instruction_following_eval

Delivery: on first run, required .py files and input_data.jsonl are downloaded to
  ~/.cache/soft_dpo/ifeval/
(master branch from raw.githubusercontent.com; pin a full google-research commit in
IFEVAL_GITHUB_REV below for reproducibility).

Eval dependencies (as in that directory's official requirements.txt): see comments in
requirements.py (absl-py, langdetect, nltk, immutabledict).

Examples:
  # Base 7B, no LoRA
  python ifeval_run.py --base-only --base-model 7b --output ifeval2/base_7b

  # UltraFeedback soft-DPO + LoRA checkpoint (7B base); artifacts default to <checkpoint>/ifeval/
  python ifeval_run.py -c checkpoints/ultrafb/soft_ultrafb_lr4e5_beta01/best \\
      --base-model 7b -o ifeval2/soft_ultrafb_lr4e5_beta01
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, Optional, Tuple

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

from alpaca_eval_judge import (
    BASE_MODEL_CHOICES,
    BASE_MODEL_HELP,
    generate_responses,
    is_cuda_device,
    load_base_model,
    load_candidate_model,
)

# Pin revision (set full google-research commit SHA for a fully pinned checkout).
IFEVAL_GITHUB_REV = "master"
IFEVAL_REPO_PREFIX = (
    f"https://raw.githubusercontent.com/google-research/google-research/"
    f"{IFEVAL_GITHUB_REV}/instruction_following_eval"
)

IFEVAL_PY_FILES = (
    "evaluation_lib.py",
    "instructions.py",
    "instructions_registry.py",
    "instructions_util.py",
)
IFEVAL_DATA_REL = "data/input_data.jsonl"

DEFAULT_CACHE_ROOT = Path.home() / ".cache" / "soft_dpo" / "ifeval"


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=120) as resp:
            data = resp.read()
    except urllib.error.URLError as e:
        raise RuntimeError(f"Failed to download {url}: {e}") from e
    dest.write_bytes(data)


def ensure_ifeval_sources(cache_root: Optional[Path] = None) -> Tuple[Path, Path]:
    """
    Places the instruction_following_eval package and data under cache_root.
    Returns (pkg_parent, path_to_input_jsonl); add pkg_parent to sys.path.

    Marker `.ifeval_sources_ok` stores IFEVAL_REPO_PREFIX so that when the revision
    (IFEVAL_GITHUB_REV) changes, the cache is invalidated and files are re-downloaded.
    """
    root = (cache_root or DEFAULT_CACHE_ROOT).resolve()
    pkg_parent = root / "pkg"
    pkg_dir = pkg_parent / "instruction_following_eval"
    data_path = root / "data" / "input_data.jsonl"

    marker = root / ".ifeval_sources_ok"
    cached_prefix = ""
    if marker.is_file():
        try:
            cached_prefix = marker.read_text(encoding="utf-8").strip()
        except OSError:
            cached_prefix = ""
    sources_complete = (
        cached_prefix == IFEVAL_REPO_PREFIX
        and all((pkg_dir / name).is_file() for name in IFEVAL_PY_FILES)
        and data_path.is_file()
    )
    if sources_complete:
        return pkg_parent, data_path

    pkg_dir.mkdir(parents=True, exist_ok=True)
    for name in IFEVAL_PY_FILES:
        _download(f"{IFEVAL_REPO_PREFIX}/{name}", pkg_dir / name)
    init_py = pkg_dir / "__init__.py"
    if not init_py.is_file():
        init_py.write_text("# namespace package marker for IFEval\n", encoding="utf-8")

    _download(f"{IFEVAL_REPO_PREFIX}/{IFEVAL_DATA_REL}", data_path)
    marker.write_text(IFEVAL_REPO_PREFIX + "\n", encoding="utf-8")
    return pkg_parent, data_path


def _ensure_nltk(cache_root: Optional[Path] = None) -> None:
    """
    Ensures NLTK tokenizers (punkt_tab or punkt) are available.
    If HOME is read-only, download to cache_root/nltk_data and register in nltk.data.path.
    """
    import nltk

    for resource in ("punkt_tab", "punkt"):
        try:
            nltk.data.find(f"tokenizers/{resource}")
            return
        except LookupError:
            pass

    download_dir: Optional[str] = None
    if cache_root is not None:
        nltk_dir = (cache_root / "nltk_data").resolve()
        nltk_dir.mkdir(parents=True, exist_ok=True)
        if str(nltk_dir) not in nltk.data.path:
            nltk.data.path.insert(0, str(nltk_dir))
        download_dir = str(nltk_dir)

    for resource in ("punkt_tab", "punkt"):
        try:
            if download_dir is not None:
                nltk.download(resource, quiet=True, download_dir=download_dir)
            else:
                nltk.download(resource, quiet=True)
        except Exception:
            continue
    for resource in ("punkt_tab", "punkt"):
        try:
            nltk.data.find(f"tokenizers/{resource}")
            return
        except LookupError:
            pass
    raise RuntimeError(
        "NLTK: tokenizers/punkt_tab or punkt not found. "
        "Install nltk and run: python3 -c \"import nltk; nltk.download('punkt_tab')\""
    )


def import_evaluation_lib(pkg_parent: Path, cache_root: Optional[Path] = None):
    """Import official evaluation_lib after adding pkg_parent to the path."""
    parent = str(pkg_parent.resolve())
    if parent not in sys.path:
        sys.path.insert(0, parent)

    try:
        import absl.logging

        absl.logging.set_verbosity(absl.logging.ERROR)
    except ImportError:
        pass

    _ensure_nltk(cache_root=cache_root)

    try:
        from instruction_following_eval import evaluation_lib  # noqa: WPS433
    except ImportError as e:
        raise ImportError(
            "IFEval requires dependencies from official instruction_following_eval/requirements.txt. "
            "Install: python3 -m pip install absl-py langdetect nltk immutabledict"
        ) from e

    return evaluation_lib


def load_ifeval_inputs_simple(
    jsonl_path: str,
    pkg_parent: Path,
    max_evals: Optional[int] = None,
    cache_root: Optional[Path] = None,
):
    evaluation_lib = import_evaluation_lib(pkg_parent, cache_root=cache_root)
    inputs = evaluation_lib.read_prompt_list(jsonl_path)
    if max_evals is not None:
        inputs = inputs[:max_evals]
    return evaluation_lib, inputs


def aggregate_metrics(outputs) -> Dict[str, Any]:
    """Same aggregates as print_report in evaluation_lib (prompt / instruction level + by type)."""
    prompt_total = 0
    prompt_correct = 0
    instruction_total = 0
    instruction_correct = 0
    tier0_total: Dict[str, int] = collections.defaultdict(int)
    tier0_correct: Dict[str, int] = collections.defaultdict(int)

    for example in outputs:
        follow_instruction_list = example.follow_instruction_list
        instruction_id_list = example.instruction_id_list
        prompt_total += 1
        if all(follow_instruction_list):
            prompt_correct += 1
        instruction_total += len(instruction_id_list)
        instruction_correct += sum(follow_instruction_list)
        for instruction_id, followed in zip(instruction_id_list, follow_instruction_list):
            tier0 = instruction_id.split(":")[0]
            tier0_total[tier0] += 1
            if followed:
                tier0_correct[tier0] += 1

    return {
        "prompt_level_accuracy": prompt_correct / prompt_total if prompt_total else 0.0,
        "instruction_level_accuracy": (
            instruction_correct / instruction_total if instruction_total else 0.0
        ),
        "prompt_total": prompt_total,
        "prompt_correct": prompt_correct,
        "instruction_total": instruction_total,
        "instruction_correct": instruction_correct,
        "by_constraint_prefix": {
            k: tier0_correct[k] / tier0_total[k] for k in sorted(tier0_total.keys())
        },
    }


def default_output_dir(
    base_only: bool, base_model_key: str, checkpoint: Optional[str]
) -> str:
    if base_only:
        return f"ifeval2/base_{base_model_key}"
    return os.path.join(checkpoint or ".", "ifeval")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="IFEval: rule-based instruction-following (Google IFEval)"
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=str,
        default=None,
        help="Checkpoint (LoRA or full model). Not used with --base-only.",
    )
    parser.add_argument(
        "--base-only",
        action="store_true",
        help="Evaluate base model only without LoRA.",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        choices=list(BASE_MODEL_CHOICES.keys()),
        default="3b",
        help=BASE_MODEL_HELP,
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Artifact directory (default: ifeval2/base_* or <checkpoint>/ifeval).",
    )
    parser.add_argument(
        "--data",
        "-d",
        type=str,
        default=None,
        help="Local IFEval input_data.jsonl; otherwise from cache ~/.cache/soft_dpo/ifeval/.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="IFEval cache root (default: ~/.cache/soft_dpo/ifeval).",
    )
    parser.add_argument(
        "--max-evals",
        type=int,
        default=None,
        help="Max number of prompts (debugging).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="Max new tokens (IFEval often needs long completions).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Generation batch size.",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Sampling (temperature 0.6 as in alpaca_eval_judge).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cuda / cpu (default: cuda if available).",
    )
    parser.add_argument(
        "--skip-loose",
        action="store_true",
        help="Skip loose metrics (upper bound with relaxed checks).",
    )
    parser.add_argument(
        "--no-system-prompt",
        action="store_true",
        help=(
            "Do not add system='You are a helpful assistant.' to the chat template. "
            "Useful for IFEval: with the default system message, models sometimes add "
            "a preamble ('Sure! Here's...') that breaks positional constraints "
            "(first_word, startswith, response_language, etc.)."
        ),
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help=(
            "Custom system prompt (overrides default). Ignored with --no-system-prompt."
        ),
    )
    args = parser.parse_args()

    if args.base_only and args.checkpoint:
        parser.error("Cannot use --checkpoint and --base-only together.")
    if not args.base_only and not args.checkpoint:
        parser.error("Provide --checkpoint or --base-only.")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    base_model_id = BASE_MODEL_CHOICES[args.base_model]

    cache_root = Path(args.cache_dir).expanduser() if args.cache_dir else DEFAULT_CACHE_ROOT
    pkg_parent, default_data = ensure_ifeval_sources(cache_root)
    data_path = args.data or str(default_data)
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"IFEval data file not found: {data_path}")

    out_dir = args.output or default_output_dir(
        args.base_only, args.base_model, args.checkpoint
    )
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "eval.log")

    def log(msg: str) -> None:
        print(msg, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    open(log_path, "w", encoding="utf-8").close()
    run_started_at = datetime.now()
    run_started_perf = perf_counter()

    model_name_for_log = base_model_id if args.base_only else str(args.checkpoint)

    if args.no_system_prompt:
        system_prompt: Optional[str] = None
    elif args.system_prompt is not None:
        system_prompt = args.system_prompt
    else:
        system_prompt = "You are a helpful assistant."

    log("=== IFEval (Google Instruction Following Eval, rule-based) ===")
    log(f"Run started at: {run_started_at.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Source: {IFEVAL_REPO_PREFIX}")
    log(f"Model: {model_name_for_log}" + (" (base, no LoRA)" if args.base_only else ""))
    log(f"Base (for LoRA): {base_model_id}")
    log(f"Data: {data_path}")
    log(f"Device: {device}")
    log(f"max_new_tokens={args.max_new_tokens}, batch_size={args.batch_size}, do_sample={args.do_sample}")
    log(
        "system_prompt: "
        + ("<none>" if system_prompt is None else repr(system_prompt))
    )

    evaluation_lib, inputs = load_ifeval_inputs_simple(
        data_path, pkg_parent, args.max_evals, cache_root=cache_root
    )
    n = len(inputs)
    log(f"Examples: {n}")

    if args.base_only:
        log(f"Loading base model: {base_model_id}...")
        tokenizer, model, _ = load_base_model(base_model=base_model_id, device=device)
    else:
        log(f"Loading checkpoint (base {base_model_id})...")
        tokenizer, model, _ = load_candidate_model(
            args.checkpoint, base_model=base_model_id, device=device
        )

    prompts = [inp.prompt for inp in inputs]
    log("Generating responses...")
    responses = generate_responses(
        tokenizer,
        model,
        device,
        prompts,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        do_sample=args.do_sample,
        system_prompt=system_prompt,
    )

    del model
    if is_cuda_device(device):
        torch.cuda.empty_cache()

    prompt_to_response = {inp.prompt: responses[i] for i, inp in enumerate(inputs)}

    responses_jsonl = os.path.join(out_dir, "responses.jsonl")
    with open(responses_jsonl, "w", encoding="utf-8") as f:
        for inp, resp in zip(inputs, responses):
            rec = {
                "key": inp.key,
                "instruction_id_list": inp.instruction_id_list,
                "kwargs": inp.kwargs,
                "prompt": inp.prompt,
                "response": resp,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    log(f"Saved {responses_jsonl}")

    candidate_json = os.path.join(out_dir, "candidate_outputs.json")
    with open(candidate_json, "w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "instruction_id_list": inp.instruction_id_list,
                    "prompt": inp.prompt,
                    "response": responses[i],
                    "key": inp.key,
                }
                for i, inp in enumerate(inputs)
            ],
            f,
            ensure_ascii=False,
            indent=2,
        )
    log(f"Saved {candidate_json}")

    metrics_summary: Dict[str, Any] = {}

    def run_mode(
        label: str,
        test_fn: Callable,
        out_suffix: str,
    ) -> None:
        outputs = [test_fn(inp, prompt_to_response) for inp in inputs]
        out_jsonl = os.path.join(out_dir, f"eval_results_{out_suffix}.jsonl")
        evaluation_lib.write_outputs(out_jsonl, outputs)
        log(f"Saved {out_jsonl}")
        m = aggregate_metrics(outputs)
        metrics_summary[label] = m
        log("")
        log(f"--- {label} ---")
        log(
            f"prompt-level accuracy (all constraints per prompt): {m['prompt_level_accuracy']:.4f}"
        )
        log(
            f"instruction-level accuracy (per constraint): {m['instruction_level_accuracy']:.4f}"
        )
        log("Per constraint type (prefix before ':', official tier0 bucket):")
        for prefix, acc in m["by_constraint_prefix"].items():
            log(f"  {prefix}: {acc:.4f}")

    log("")
    log("Evaluating (strict, official test_instruction_following_strict)...")
    run_mode("strict", evaluation_lib.test_instruction_following_strict, "strict")

    if not args.skip_loose:
        log("")
        log("Evaluating (loose, official test_instruction_following_loose)...")
        run_mode("loose", evaluation_lib.test_instruction_following_loose, "loose")

    metrics_path = os.path.join(out_dir, "ifeval_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2, ensure_ascii=False)
    log(f"Saved {metrics_path}")
    log(f"Done. Artifacts in {out_dir}")
    run_finished_at = datetime.now()
    run_duration_sec = perf_counter() - run_started_perf
    run_duration_hours = run_duration_sec / 3600.0
    log(f"Run finished at: {run_finished_at.strftime('%Y-%m-%d %H:%M:%S')}")
    log("Run status: SUCCESS")
    log(f"Run duration: {run_duration_hours:.2f}h")


if __name__ == "__main__":
    main()
