#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch AlpacaEval 2.0 for a folder of seed checkpoints (best/ weights).

Discovers subfolders with LoRA adapters, runs alpaca_eval_judge.py --alpaca2 in
parallel (one job per GPU), then prints a summary table with per-seed metrics,
mean and std across seeds.

Example (4 seeds on GPUs 0–3, Qwen3-4B base):
  python alpaca_eval_batch.py \\
      -f checkpoints/hsteer/4b/compar_sgd_stand_cetered_sftpl_lr2e4/classic \\
      --base-model 4b

Parent folder with experiment groups (classic/, centered_softplus/):
  python alpaca_eval_batch.py \\
      -f checkpoints/hsteer/4b/compar_sgd_stand_cetered_sftpl_lr2e4 \\
      --base-model 4b
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from alpaca_eval_lock import clear_stale_eval_lock, clear_stale_eval_locks_in_tree
from utils.config import BASE_MODEL_CHOICES, BASE_MODEL_HELP

REPO_ROOT = Path(__file__).resolve().parent
JUDGE_SCRIPT = REPO_ROOT / "alpaca_eval_judge.py"

SUMMARY_METRICS = (
    ("win_rate", "Win rate"),
    ("length_controlled_win_rate", "LC win rate"),
    ("tie_rate", "Tie rate"),
    ("loss_rate", "Loss rate"),
)


@dataclass(frozen=True)
class RunJob:
    group: str
    run_name: str
    checkpoint: Path


def _has_adapter(checkpoint_dir: Path) -> bool:
    return (checkpoint_dir / "adapter_config.json").is_file()


def find_best_checkpoints(root: Path) -> List[Tuple[str, Path]]:
    """Return [(subdir_name, best_checkpoint_path), ...] sorted by name."""
    root = root.resolve()
    found: List[Tuple[str, Path]] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        best = child / "best"
        if best.is_dir() and _has_adapter(best):
            found.append((child.name, best))
        elif _has_adapter(child):
            found.append((child.name, child))
    return found


def discover_experiment_groups(folder: Path) -> Dict[str, List[Tuple[str, Path]]]:
    """
    One-level: folder/seed*/best/ -> {folder.name: [...]}.
    Two-level: folder/group/seed*/best/ -> {group: [...]}.
    """
    folder = folder.resolve()
    direct = find_best_checkpoints(folder)
    if direct:
        return {folder.name: direct}

    groups: Dict[str, List[Tuple[str, Path]]] = {}
    for child in sorted(folder.iterdir()):
        if not child.is_dir():
            continue
        runs = find_best_checkpoints(child)
        if runs:
            groups[child.name] = runs
    return groups


def parse_seed_filter(seeds_arg: Optional[str]) -> Optional[set[int]]:
    if not seeds_arg:
        return None
    seeds: set[int] = set()
    for part in seeds_arg.split(","):
        part = part.strip()
        if part:
            seeds.add(int(part))
    return seeds or None


def job_matches_seed_filter(job: RunJob, seeds: set[int]) -> bool:
    for seed in seeds:
        suffix = f"seed{seed}"
        if job.run_name == suffix or job.run_name.endswith(f"_{suffix}"):
            return True
    return False


def build_jobs(
    folder: Path,
    output_root: Optional[Path],
) -> Tuple[List[RunJob], Path]:
    groups = discover_experiment_groups(folder)
    if not groups:
        raise FileNotFoundError(
            f"No checkpoints with best/ LoRA adapters found under {folder}. "
            "Expected layout: <folder>/epoch_6_seed43/best/adapter_config.json "
            "or <folder>/<group>/epoch_6_seed43/best/."
        )

    out_root = (folder / "alpaca_eval2") if output_root is None else output_root.resolve()

    jobs: List[RunJob] = []
    for group_name, runs in groups.items():
        for run_name, ckpt in runs:
            jobs.append(
                RunJob(
                    group=group_name,
                    run_name=run_name,
                    checkpoint=ckpt.resolve(),
                )
            )
    return jobs, out_root


def job_output_dir(out_root: Path, job: RunJob, multi_group: bool) -> Path:
    if multi_group:
        return out_root / job.group / job.run_name
    return out_root / job.run_name


def run_single_job(
    job: RunJob,
    gpu: int,
    out_dir: Path,
    base_model: str,
    max_evals: Optional[int],
    max_new_tokens: int,
    batch_size: int,
    do_sample: bool,
    judge_seed: int,
    skip_existing: bool,
    judge_only: bool = False,
) -> Optional[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "judge_results.json"
    lock_path = out_dir / ".eval.lock"
    candidate_path = out_dir / "candidate_outputs.json"
    if skip_existing and results_path.is_file():
        print(f"[GPU {gpu}] skip existing: {job.run_name} -> {results_path}", flush=True)
        return results_path
    if judge_only and not candidate_path.is_file():
        print(
            f"[GPU {gpu}] skip (no candidate_outputs.json): {job.run_name}",
            flush=True,
        )
        return None
    if lock_path.is_file():
        if clear_stale_eval_lock(lock_path, out_dir):
            pid = lock_path.read_text(encoding="utf-8").strip()
            print(
                f"[GPU {gpu}] skip (already running): {job.run_name} "
                f"lock={lock_path} pid={pid}",
                flush=True,
            )
            return None

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    cmd = [
        sys.executable,
        str(JUDGE_SCRIPT),
        "--alpaca2",
        "--output",
        str(out_dir),
        "--device",
        "cuda",
        "--judge-seed",
        str(judge_seed),
    ]
    if judge_only:
        cmd.append("--judge-only")
    else:
        cmd.extend(
            [
                "--checkpoint",
                str(job.checkpoint),
                "--base-model",
                base_model,
                "--max-new-tokens",
                str(max_new_tokens),
                "--batch-size",
                str(batch_size),
            ]
        )
    if skip_existing:
        cmd.append("--skip-existing")
    if not judge_only:
        if max_evals is not None:
            cmd.extend(["--max-evals", str(max_evals)])
        if do_sample:
            cmd.append("--do-sample")

    mode = "judge-only" if judge_only else "full"
    print(
        f"[GPU {gpu}] {job.group}/{job.run_name} mode={mode} out={out_dir}",
        flush=True,
    )
    result = subprocess.run(cmd, env=env, cwd=str(REPO_ROOT))
    if result.returncode == 2:
        print(
            f"[GPU {gpu}] skip (already running): {job.run_name}",
            flush=True,
        )
        return None
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd)
    if not results_path.is_file():
        raise FileNotFoundError(
            f"Expected results at {results_path} after alpaca_eval_judge run."
        )
    return results_path


def load_results(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _mean_std(values: Sequence[float]) -> Tuple[Optional[float], Optional[float]]:
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def _pct(x: Optional[float]) -> str:
    if x is None:
        return "—"
    return f"{100.0 * x:.2f}%"


def _pct_pm(mean: Optional[float], std: Optional[float]) -> str:
    if mean is None:
        return "—"
    if std is None:
        return _pct(mean)
    return f"{100.0 * mean:.2f}% ± {100.0 * std:.2f}%"


def summarize_group(
    group: str,
    rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"group": group, "runs": rows, "aggregate": {}}
    for key, _label in SUMMARY_METRICS:
        vals = [r[key] for r in rows if r.get(key) is not None]
        mean, std = _mean_std(vals)
        summary["aggregate"][key] = {"mean": mean, "std": std, "n": len(vals)}
    return summary


def print_group_table(
    group: str,
    rows: List[Dict[str, Any]],
    aggregate: Dict[str, Any],
) -> None:
    print("\n" + "=" * 72, flush=True)
    print(f"  AlpacaEval 2.0 — {group}", flush=True)
    print("=" * 72, flush=True)

    header = ["run"] + [label for _, label in SUMMARY_METRICS] + ["n"]
    col_widths = [max(len(h), 16) for h in header]

    def fmt_row(cells: List[str]) -> str:
        for i, cell in enumerate(cells):
            col_widths[i] = max(col_widths[i], len(cell))
        return "  ".join(c.ljust(col_widths[i]) for i, c in enumerate(cells))

    print(fmt_row(header), flush=True)
    print("-" * (sum(col_widths) + 2 * (len(header) - 1)), flush=True)
    for row in rows:
        cells = [row["run_name"]]
        for key, _ in SUMMARY_METRICS:
            cells.append(_pct(row.get(key)))
        cells.append(str(row.get("n", "")))
        print(fmt_row(cells), flush=True)

    mean_cells = ["mean ± std"]
    for key, _ in SUMMARY_METRICS:
        agg = aggregate.get(key, {})
        mean_cells.append(_pct_pm(agg.get("mean"), agg.get("std")))
    mean_cells.append("")
    print("-" * (sum(col_widths) + 2 * (len(header) - 1)), flush=True)
    print(fmt_row(mean_cells), flush=True)
    print("=" * 72 + "\n", flush=True)


def save_summary_csv(path: Path, summaries: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["group", "run_name", "checkpoint"] + [k for k, _ in SUMMARY_METRICS] + ["n"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            group = summary["group"]
            for row in summary["runs"]:
                writer.writerow(
                    {
                        "group": group,
                        "run_name": row["run_name"],
                        "checkpoint": row.get("checkpoint", ""),
                        "win_rate": row.get("win_rate"),
                        "length_controlled_win_rate": row.get("length_controlled_win_rate"),
                        "tie_rate": row.get("tie_rate"),
                        "loss_rate": row.get("loss_rate"),
                        "n": row.get("n"),
                    }
                )
            agg = summary["aggregate"]
            writer.writerow(
                {
                    "group": group,
                    "run_name": "mean",
                    "checkpoint": "",
                    "win_rate": agg.get("win_rate", {}).get("mean"),
                    "length_controlled_win_rate": agg.get(
                        "length_controlled_win_rate", {}
                    ).get("mean"),
                    "tie_rate": agg.get("tie_rate", {}).get("mean"),
                    "loss_rate": agg.get("loss_rate", {}).get("mean"),
                    "n": "",
                }
            )
            writer.writerow(
                {
                    "group": group,
                    "run_name": "std",
                    "checkpoint": "",
                    "win_rate": agg.get("win_rate", {}).get("std"),
                    "length_controlled_win_rate": agg.get(
                        "length_controlled_win_rate", {}
                    ).get("std"),
                    "tie_rate": agg.get("tie_rate", {}).get("std"),
                    "loss_rate": agg.get("loss_rate", {}).get("std"),
                    "n": "",
                }
            )


def rows_from_output_dir(seed_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for child in sorted(seed_root.iterdir()):
        if not child.is_dir():
            continue
        results_path = child / "judge_results.json"
        if not results_path.is_file():
            continue
        metrics = load_results(results_path)
        rows.append(
            {
                "run_name": child.name,
                "checkpoint": metrics.get("checkpoint", ""),
                "win_rate": metrics.get("win_rate"),
                "length_controlled_win_rate": metrics.get("length_controlled_win_rate"),
                "tie_rate": metrics.get("tie_rate"),
                "loss_rate": metrics.get("loss_rate"),
                "n": metrics.get("n"),
                "results_path": str(results_path),
            }
        )
    return rows


def collect_summaries_from_output(out_root: Path) -> List[Dict[str, Any]]:
    out_root = out_root.resolve()
    direct_rows = rows_from_output_dir(out_root)
    if direct_rows:
        return [summarize_group(out_root.name, direct_rows)]

    summaries: List[Dict[str, Any]] = []
    for child in sorted(out_root.iterdir()):
        if not child.is_dir():
            continue
        rows = rows_from_output_dir(child)
        if rows:
            summaries.append(summarize_group(child.name, rows))
    return summaries


def write_summaries(out_root: Path, summaries: List[Dict[str, Any]]) -> None:
    if not summaries:
        raise FileNotFoundError(
            f"No judge_results.json found under {out_root.resolve()}."
        )
    for summary in summaries:
        print_group_table(summary["group"], summary["runs"], summary["aggregate"])
    summary_json = out_root / "summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)
    print(f"Saved summary JSON: {summary_json}", flush=True)
    summary_csv = out_root / "summary.csv"
    save_summary_csv(summary_csv, summaries)
    print(f"Saved summary CSV: {summary_csv}", flush=True)


def summarize_output_tree(out_root: Path, per_group: bool = True) -> List[Dict[str, Any]]:
    """Rebuild summary.csv/json from existing judge_results.json under out_root."""
    out_root = out_root.resolve()
    direct_rows = rows_from_output_dir(out_root)
    if direct_rows:
        summaries = [summarize_group(out_root.name, direct_rows)]
        write_summaries(out_root, summaries)
        return summaries

    summaries: List[Dict[str, Any]] = []
    for child in sorted(out_root.iterdir()):
        if not child.is_dir():
            continue
        rows = rows_from_output_dir(child)
        if not rows:
            continue
        summary = summarize_group(child.name, rows)
        summaries.append(summary)
        if per_group:
            write_summaries(child, [summary])

    if not summaries:
        raise FileNotFoundError(
            f"No judge_results.json found under {out_root}."
        )
    write_summaries(out_root, summaries)
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch AlpacaEval 2.0 for seed folders (parallel on multiple GPUs)"
    )
    parser.add_argument(
        "--folder",
        "-f",
        type=str,
        default=None,
        help=(
            "Folder with seed subdirs (each containing best/) or parent with "
            "experiment groups (classic/, centered_softplus/, ...)."
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output root (default: <folder>/alpaca_eval2).",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        choices=list(BASE_MODEL_CHOICES.keys()),
        default="4b",
        help=BASE_MODEL_HELP + " Default: 4b (Qwen3-4B).",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0,1,2,3",
        help="Comma-separated GPU ids for parallel jobs (default: 0,1,2,3).",
    )
    parser.add_argument(
        "--max-evals",
        type=int,
        default=None,
        help="Max AlpacaEval examples per run (debug).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Max new tokens for candidate generation.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Generation batch size per job.",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Use sampling during candidate generation.",
    )
    parser.add_argument(
        "--judge-seed",
        type=int,
        default=0,
        help="Seed for judge answer-order randomization.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help=(
            "Comma-separated seed ids to run (e.g. 47,48,49,50). "
            "Matches run folders like epoch_6_seed47."
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip runs where judge_results.json already exists.",
    )
    parser.add_argument(
        "--judge-only",
        action="store_true",
        help=(
            "Skip generation; run judge only for seeds with candidate_outputs.json "
            "in the output directory."
        ),
    )
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help=(
            "Rebuild summary.csv/json from existing judge_results.json under "
            "--output (or --folder). Does not run eval jobs."
        ),
    )
    args = parser.parse_args()

    if args.summarize_only:
        out_arg = args.output or args.folder
        if not out_arg:
            parser.error("--summarize-only requires --output or --folder")
        summarize_output_tree(Path(out_arg).expanduser(), per_group=True)
        return

    if not args.folder:
        parser.error("--folder is required unless --summarize-only is set")

    if not JUDGE_SCRIPT.is_file():
        raise FileNotFoundError(f"alpaca_eval_judge.py not found at {JUDGE_SCRIPT}")

    folder = Path(args.folder).expanduser()
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")

    gpus = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]
    if not gpus:
        raise ValueError("--gpus must list at least one GPU id.")

    output_root = Path(args.output).expanduser() if args.output else None
    jobs, out_root = build_jobs(folder, output_root)
    seed_filter = parse_seed_filter(args.seeds)
    if seed_filter is not None:
        jobs = [job for job in jobs if job_matches_seed_filter(job, seed_filter)]
        if not jobs:
            raise ValueError(
                f"No jobs match --seeds {args.seeds!r} under {folder.resolve()}."
            )
    out_root.mkdir(parents=True, exist_ok=True)
    multi_group = len({j.group for j in jobs}) > 1

    print(f"Folder: {folder.resolve()}", flush=True)
    print(f"Output: {out_root}", flush=True)
    if seed_filter is not None:
        print(f"Seeds: {sorted(seed_filter)}", flush=True)
    print(
        f"Jobs: {len(jobs)} ({', '.join(j.group + '/' + j.run_name for j in jobs)})",
        flush=True,
    )
    print(f"GPUs: {gpus}", flush=True)
    if args.judge_only:
        print("Mode: judge-only (reuse candidate_outputs.json)", flush=True)

    n_cleared = clear_stale_eval_locks_in_tree(out_root)
    if n_cleared:
        print(f"Cleared {n_cleared} stale .eval.lock file(s) under {out_root}", flush=True)

    def _execute(job_index: Tuple[int, RunJob]) -> Tuple[RunJob, Optional[Path]]:
        idx, job = job_index
        gpu = gpus[idx % len(gpus)]
        out_dir = job_output_dir(out_root, job, multi_group)
        results_path = run_single_job(
            job=job,
            gpu=gpu,
            out_dir=out_dir,
            base_model=args.base_model,
            max_evals=args.max_evals,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            do_sample=args.do_sample,
            judge_seed=args.judge_seed,
            skip_existing=args.skip_existing,
            judge_only=args.judge_only,
        )
        return job, results_path

    max_workers = min(len(gpus), len(jobs))
    indexed_jobs = list(enumerate(jobs))
    job_results: List[Tuple[RunJob, Optional[Path]]] = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_execute, item): item for item in indexed_jobs}
        for fut in as_completed(futures):
            job, results_path = fut.result()
            job_results.append((job, results_path))
            if results_path is None:
                print(f"Skipped (in progress): {job.group}/{job.run_name}", flush=True)
            else:
                print(f"Done: {job.group}/{job.run_name}", flush=True)

    rows_by_group: Dict[str, List[Dict[str, Any]]] = {}
    for job, results_path in sorted(job_results, key=lambda x: (x[0].group, x[0].run_name)):
        if results_path is None:
            continue
        metrics = load_results(results_path)
        row = {
            "run_name": job.run_name,
            "checkpoint": str(job.checkpoint),
            "win_rate": metrics.get("win_rate"),
            "length_controlled_win_rate": metrics.get("length_controlled_win_rate"),
            "tie_rate": metrics.get("tie_rate"),
            "loss_rate": metrics.get("loss_rate"),
            "n": metrics.get("n"),
            "results_path": str(results_path),
        }
        rows_by_group.setdefault(job.group, []).append(row)

    summaries: List[Dict[str, Any]] = []
    for group in sorted(rows_by_group.keys()):
        rows = rows_by_group[group]
        summary = summarize_group(group, rows)
        summaries.append(summary)
        print_group_table(group, rows, summary["aggregate"])

    summary_json = out_root / "summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)
    print(f"Saved summary JSON: {summary_json}", flush=True)

    summary_csv = out_root / "summary.csv"
    save_summary_csv(summary_csv, summaries)
    print(f"Saved summary CSV: {summary_csv}", flush=True)


if __name__ == "__main__":
    main()
