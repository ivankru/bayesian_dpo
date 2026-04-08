#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Визуализация сохранённых val-распределений (npz из train_dpo).

Пример:
  python scripts/plot_val_distributions.py \\
    --npz_paths run/a/val_distributions_epoch1.npz run/a/val_distributions_epoch4.npz \\
    --labels epoch1 epoch4 \\
    --out_dir plots/run_a/
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _stats_row(arr: np.ndarray) -> Tuple[float, float, float, float, float]:
    if arr.size == 0:
        return (float("nan"),) * 5
    return (
        float(np.mean(arr)),
        float(np.std(arr)),
        float(np.median(arr)),
        float(np.percentile(arr, 5)),
        float(np.percentile(arr, 95)),
    )


def _series_colors(n: int) -> np.ndarray:
    if n <= 10:
        return plt.cm.tab10(np.linspace(0, 0.95, n))
    if n <= 20:
        return plt.cm.tab20(np.linspace(0, 0.95, n))
    base = plt.cm.tab20(np.linspace(0, 0.95, 20))
    return np.array([base[i % 20] for i in range(n)])


def _load_series(
    npz_paths: Sequence[str], labels: Sequence[str]
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for path, label in zip(npz_paths, labels):
        data = np.load(path)
        out.append(
            {
                "label": label,
                "path": path,
                "delta_theta": np.asarray(data["delta_theta"]),
                "delta_ref": np.asarray(data["delta_ref"]),
                "diff": np.asarray(data["diff"]),
            }
        )
    return out


def _hist_shared_bins(arrays: List[np.ndarray], n_edges: int = 61) -> np.ndarray:
    non_empty = [a for a in arrays if a.size > 0]
    if not non_empty:
        return np.linspace(-1.0, 1.0, n_edges)
    lo = min(float(a.min()) for a in non_empty)
    hi = max(float(a.max()) for a in non_empty)
    if lo == hi:
        lo -= 1.0
        hi += 1.0
    return np.linspace(lo, hi, n_edges)


def _plot_histogram_overlay(
    series: List[Dict[str, Any]],
    key: str,
    title: str,
    out_path: str,
    colors: np.ndarray,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    arrays = [np.asarray(s[key]) for s in series]
    nonempty = [a for a in arrays if a.size > 0]
    if not nonempty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return

    bins = _hist_shared_bins(arrays)
    for i, s in enumerate(series):
        arr = np.asarray(s[key])
        if arr.size == 0:
            continue
        ax.hist(
            arr,
            bins=bins,
            alpha=0.5,
            label=s["label"],
            color=colors[i % len(colors)],
            density=True,
        )
    ax.axvline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_scatter_diff_vs_delta_ref(
    series: List[Dict[str, Any]], out_path: str, colors: np.ndarray
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    any_pts = False
    for i, s in enumerate(series):
        dr = np.asarray(s["delta_ref"])
        diff = np.asarray(s["diff"])
        if dr.size == 0:
            continue
        ax.scatter(
            dr,
            diff,
            alpha=0.4,
            s=10,
            label=s["label"],
            color=colors[i % len(colors)],
            edgecolors="none",
        )
        any_pts = True
    if not any_pts:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
    ax.axvline(0.0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("delta_ref")
    ax.set_ylabel("diff (margin)")
    ax.set_title("diff vs delta_ref")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_summary_table(series: List[Dict[str, Any]], out_path: str) -> None:
    n = len(series)
    fig_h = max(3.0, 1.2 + 0.35 * n)
    fig, ax = plt.subplots(figsize=(12, fig_h))
    ax.axis("off")
    col_labels = ["", "mean", "std", "median", "p5", "p95"]
    cell_text: List[List[str]] = []
    for s in series:
        mean, std, med, p5, p95 = _stats_row(np.asarray(s["diff"]))
        cell_text.append(
            [
                s["label"],
                f"{mean:.2f}",
                f"{std:.2f}",
                f"{med:.2f}",
                f"{p5:.2f}",
                f"{p95:.2f}",
            ]
        )
    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.0, 1.5)
    ax.set_title("Summary: diff (margin) by epoch / file", pad=20)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot val_distributions .npz files.")
    parser.add_argument(
        "--npz_paths",
        nargs="+",
        required=True,
        help="Paths to val_distributions_epoch*.npz",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        required=True,
        help="Legend labels (same length as --npz_paths)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Directory for PNG outputs",
    )
    args = parser.parse_args()

    if len(args.npz_paths) != len(args.labels):
        raise SystemExit("Error: --npz_paths and --labels must have the same length.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["font.size"] = 11

    series = _load_series(args.npz_paths, args.labels)
    n = max(len(series), 1)
    colors = _series_colors(n)

    _plot_histogram_overlay(
        series,
        "delta_theta",
        "delta_theta (log π chosen − log π rejected, policy)",
        str(out_dir / "hist_delta_theta.png"),
        colors,
    )
    _plot_histogram_overlay(
        series,
        "delta_ref",
        "delta_ref (reference)",
        str(out_dir / "hist_delta_ref.png"),
        colors,
    )
    _plot_histogram_overlay(
        series,
        "diff",
        "diff (DPO margin)",
        str(out_dir / "hist_diff_margin.png"),
        colors,
    )
    _plot_scatter_diff_vs_delta_ref(
        series,
        str(out_dir / "scatter_diff_vs_delta_ref.png"),
        colors,
    )
    _plot_summary_table(series, str(out_dir / "summary_table.png"))


if __name__ == "__main__":
    main()
