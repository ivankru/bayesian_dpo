# -*- coding: utf-8 -*-
"""
Fixed-probe DPO margins for τ / ε / β_r diagnostics.

A single val subset is scored every ``probe_every`` steps. For Orca the 256
pairs are the committed ``utils/probe_indices_orca_dpo.py`` (not resampled).
Ref log-probs are cached once (ref is frozen), so each snapshot is two policy
forwards rather than four.

Sidecar ``probe_margins.jsonl`` holds the full Δ vector plus cheap scalars
(lr used on that step, H_cum = Σ η, endpoint ‖∇L‖). train.log gets a one-line
summary only.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.config import MAX_FULL_LEN, MAX_PROMPT_LEN
from utils.metrics import get_logps
from utils.probe_indices_orca_dpo import ORCA_DPO_PROBE
from utils.val_distributions import summarize_margin

# Frozen probe subsets. Orca pairs are locked; do not resample from --probe-seed.
CANONICAL_PROBES = {
    "orca_dpo": ORCA_DPO_PROBE,
}


class StepTracker:
    """Running Σ η, plus the lr and grad-norm of the last completed update."""

    __slots__ = ("H_cum", "last_lr", "last_grad_norm")

    def __init__(self) -> None:
        self.H_cum = 0.0
        self.last_lr = float("nan")
        self.last_grad_norm = float("nan")


def select_probe_indices(n_val: int, size: int, probe_seed: int) -> List[int]:
    """Sample ``min(size, n_val)`` unique val indices. Order is the probe order.

    Used only when the dataset has no canonical probe file. Orca uses the
    committed ``utils/probe_indices_orca_dpo.py`` instead.
    """
    n = int(n_val)
    k = min(int(size), n)
    if k <= 0 or n <= 0:
        return []
    rng = np.random.RandomState(int(probe_seed))
    return rng.choice(n, size=k, replace=False).astype(int).tolist()


def pair_fingerprint(example: Dict[str, Any]) -> str:
    """Stable id of one (prompt, chosen, rejected) triple."""
    h = hashlib.sha1()
    for key in ("prompt", "chosen", "rejected"):
        h.update(str(example.get(key, "")).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def fingerprint_probe_pairs(val_ds, indices: Sequence[int]) -> List[str]:
    return [pair_fingerprint(val_ds[int(i)]) for i in indices]


def _read_probe_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "indices" not in data:
        raise RuntimeError(f"probe: {path} has no 'indices'")
    return data


def _write_probe_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def _validate_indices(idx: List[int], n_val: int, path: str) -> None:
    if not idx:
        raise RuntimeError(f"probe: empty indices in {path}")
    if len(idx) != len(set(idx)):
        raise RuntimeError(f"probe: duplicate indices in {path}")
    if min(idx) < 0 or max(idx) >= int(n_val):
        raise RuntimeError(
            f"probe: index out of range in {path} "
            f"(n_val={n_val}, min={min(idx)}, max={max(idx)})"
        )


def _assert_pair_keys(val_ds, indices: Sequence[int], stored: Sequence[str], path: str) -> None:
    got = fingerprint_probe_pairs(val_ds, indices)
    if list(stored) != got:
        raise RuntimeError(
            f"probe: pair fingerprint mismatch vs {path}; "
            "val split or row order changed. Refusing to continue."
        )


def resolve_probe_indices(
    run_path: str,
    n_val: int,
    size: int,
    probe_seed: int,
    log: Callable[[str], None] = print,
    dataset_name: Optional[str] = None,
    val_ds=None,
) -> List[int]:
    """Lock the probe subset for this run.

    Order of sources:
      1. ``run_path`` if it already exists (resume) — never resample.
      2. Canonical file for ``dataset_name`` (Orca) — copy into ``run_path``.
      3. Else sample once from ``probe_seed`` and write ``run_path``.

    When ``val_ds`` is given, SHA1 fingerprints of the selected triples are
    stored and checked so a silent val-order change cannot retarget the probe.
    """
    n_val = int(n_val)
    ds_name = str(dataset_name).strip() if dataset_name else ""

    def _with_keys(idx: List[int], base: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(base)
        out["indices"] = idx
        out["n_val"] = n_val
        out["n_probe"] = len(idx)
        if val_ds is not None:
            out["pair_keys"] = fingerprint_probe_pairs(val_ds, idx)
        return out

    if os.path.isfile(run_path):
        data = _read_probe_json(run_path)
        idx = [int(i) for i in data["indices"]]
        _validate_indices(idx, n_val, run_path)
        stored = data.get("pair_keys")
        if stored and val_ds is not None:
            _assert_pair_keys(val_ds, idx, stored, run_path)
        elif val_ds is not None and not stored:
            _write_probe_json(run_path, _with_keys(idx, data))
        log(f"probe: reusing {len(idx)} frozen indices from {run_path}")
        return idx

    canonical = CANONICAL_PROBES.get(ds_name)
    if ds_name == "orca_dpo" and canonical is None:
        raise RuntimeError(
            "probe: orca_dpo requires utils/probe_indices_orca_dpo.py "
            "(pairs are frozen; will not sample from --probe-seed)"
        )
    if canonical is not None:
        src = f"utils/probe_indices_{ds_name}.py"
        data = dict(canonical)
        idx = [int(i) for i in data["indices"]]
        expected_n = int(data.get("n_val", n_val))
        if expected_n != n_val:
            raise RuntimeError(
                f"probe: frozen {src} expects n_val={expected_n}, got {n_val}"
            )
        _validate_indices(idx, n_val, src)
        stored = data.get("pair_keys")
        if stored and val_ds is not None:
            _assert_pair_keys(val_ds, idx, stored, src)
        payload = _with_keys(idx, data)
        payload["source"] = src
        if int(size) != len(idx):
            log(
                f"probe: using frozen {len(idx)} pairs from {src} "
                f"(ignoring --probe-size {size})"
            )
        _write_probe_json(run_path, payload)
        log(f"probe: locked {len(idx)} pairs from {src} -> {run_path}")
        return idx

    idx = select_probe_indices(n_val, size, probe_seed)
    payload = _with_keys(
        idx,
        {
            "dataset": ds_name or None,
            "probe_seed": int(probe_seed),
            "probe_size": int(size),
        },
    )
    _write_probe_json(run_path, payload)
    log(f"probe: wrote {len(idx)} indices to {run_path} (seed={probe_seed})")
    return idx


def load_or_create_probe_indices(
    path: str,
    n_val: int,
    size: int,
    probe_seed: int,
    log: Callable[[str], None] = print,
    dataset_name: Optional[str] = None,
    val_ds=None,
) -> List[int]:
    """Alias for ``resolve_probe_indices``."""
    return resolve_probe_indices(
        run_path=path,
        n_val=n_val,
        size=size,
        probe_seed=probe_seed,
        log=log,
        dataset_name=dataset_name,
        val_ds=val_ds,
    )


def build_probe_loader(val_ds, indices: Sequence[int], batch_size: int, collate_fn) -> DataLoader:
    probe_ds = val_ds.select(list(indices))
    return DataLoader(
        probe_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )


def _logps_chosen_rejected(
    model,
    tokenizer,
    batch: Dict[str, List[str]],
    device,
    use_chat_template: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    prompts = batch["prompt"]
    logp_c = get_logps(
        model,
        tokenizer,
        prompts,
        batch["chosen"],
        device,
        MAX_PROMPT_LEN,
        MAX_FULL_LEN,
        use_chat_template=use_chat_template,
    )
    logp_r = get_logps(
        model,
        tokenizer,
        prompts,
        batch["rejected"],
        device,
        MAX_PROMPT_LEN,
        MAX_FULL_LEN,
        use_chat_template=use_chat_template,
    )
    return logp_c, logp_r


def cache_ref_logps(
    ref_model,
    tokenizer,
    probe_loader: DataLoader,
    device,
    use_chat_template: bool,
    cache_path: Optional[str] = None,
    indices: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Frozen-ref logps on the probe, concatenated in loader order.

    If ``cache_path`` exists and its stored indices match, load it; otherwise
    run two ref forwards and write the npz.
    """
    if cache_path and os.path.isfile(cache_path) and indices is not None:
        try:
            blob = np.load(cache_path)
            stored = blob["indices"]
            if np.array_equal(stored, np.asarray(indices, dtype=np.int64)):
                return (
                    np.asarray(blob["logp_c_ref"], dtype=np.float64),
                    np.asarray(blob["logp_r_ref"], dtype=np.float64),
                )
        except (OSError, KeyError, ValueError):
            pass

    ref_model.eval()
    c_chunks: List[np.ndarray] = []
    r_chunks: List[np.ndarray] = []
    with torch.no_grad():
        for batch in tqdm(probe_loader, desc="probe ref logps", leave=False):
            logp_c, logp_r = _logps_chosen_rejected(
                ref_model, tokenizer, batch, device, use_chat_template
            )
            c_chunks.append(logp_c.detach().float().cpu().numpy())
            r_chunks.append(logp_r.detach().float().cpu().numpy())
    logp_c_ref = (
        np.concatenate(c_chunks).astype(np.float64)
        if c_chunks
        else np.array([], dtype=np.float64)
    )
    logp_r_ref = (
        np.concatenate(r_chunks).astype(np.float64)
        if r_chunks
        else np.array([], dtype=np.float64)
    )
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        np.savez_compressed(
            cache_path,
            logp_c_ref=logp_c_ref,
            logp_r_ref=logp_r_ref,
            indices=np.asarray(indices if indices is not None else [], dtype=np.int64),
        )
    return logp_c_ref, logp_r_ref


def eval_probe_deltas(
    policy_model,
    tokenizer,
    probe_loader: DataLoader,
    logp_c_ref: np.ndarray,
    logp_r_ref: np.ndarray,
    device,
    use_chat_template: bool,
) -> np.ndarray:
    """Δ_i = (logp_π(y_w) − logp_π(y_l)) − (logp_ref(y_w) − logp_ref(y_l)). Policy forwards only."""
    policy_model.eval()
    chunks: List[np.ndarray] = []
    offset = 0
    with torch.no_grad():
        for batch in tqdm(probe_loader, desc="probe Δ", leave=False):
            logp_c, logp_r = _logps_chosen_rejected(
                policy_model, tokenizer, batch, device, use_chat_template
            )
            bsz = int(logp_c.shape[0])
            dtheta = (logp_c - logp_r).detach().float().cpu().numpy()
            dref = logp_c_ref[offset : offset + bsz] - logp_r_ref[offset : offset + bsz]
            chunks.append(dtheta.astype(np.float64) - dref)
            offset += bsz
    if not chunks:
        return np.array([], dtype=np.float64)
    if offset != int(logp_c_ref.shape[0]):
        raise RuntimeError(
            f"probe length mismatch: got {offset} policy pairs, "
            f"cached ref has {logp_c_ref.shape[0]}"
        )
    return np.concatenate(chunks)


def probe_a_mean(delta: np.ndarray, beta: float) -> float:
    """a = β mean_i σ(−β Δ_i); per-pair sigmoid, not σ of the mean."""
    if delta.size == 0:
        return float("nan")
    x = torch.as_tensor(delta, dtype=torch.float32).mul_(-float(beta))
    return float(beta) * float(torch.sigmoid(x).mean().item())


def format_probe_log_line(
    epoch_1based: int,
    step: int,
    stats: Dict[str, float],
    a_mean: float,
    lr: float,
    H_cum: float,
    grad_norm: float,
) -> str:
    n = int(stats.get("n", 0.0)) if stats else 0

    def _f(v: float, spec: str) -> str:
        if v != v:
            return "nan"
        return format(v, spec)

    return (
        f"[epoch {epoch_1based} step {step}] probe "
        f"n={n} "
        f"mean={_f(stats.get('mean', float('nan')), '.4f')} "
        f"median={_f(stats.get('median', float('nan')), '.4f')} "
        f"p5={_f(stats.get('p5', float('nan')), '.4f')} "
        f"p95={_f(stats.get('p95', float('nan')), '.4f')} "
        f"a_mean={_f(a_mean, '.6e')} "
        f"lr={_f(lr, '.4e')} "
        f"H_cum={_f(H_cum, '.6e')} "
        f"grad_norm={_f(grad_norm, '.6e')}"
    )


def append_probe_jsonl(
    path: str,
    *,
    step: int,
    epoch: int,
    lr: float,
    H_cum: float,
    grad_norm: float,
    beta: float,
    delta: np.ndarray,
    stats: Dict[str, float],
    a_mean: float,
) -> None:
    rec: Dict[str, Any] = {
        "kind": "probe",
        "step": int(step),
        "epoch": int(epoch),
        "lr": float(lr) if lr == lr else None,
        "H_cum": float(H_cum) if H_cum == H_cum else None,
        "grad_norm": float(grad_norm) if grad_norm == grad_norm else None,
        "beta": float(beta),
        "n": int(delta.size),
        "mean": stats.get("mean"),
        "median": stats.get("median"),
        "p5": stats.get("p5"),
        "p95": stats.get("p95"),
        "a_mean": float(a_mean) if a_mean == a_mean else None,
        "delta": np.asarray(delta, dtype=np.float64).tolist(),
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def snapshot_probe(
    policy_model,
    tokenizer,
    probe_loader: DataLoader,
    logp_c_ref: np.ndarray,
    logp_r_ref: np.ndarray,
    device,
    use_chat_template: bool,
    *,
    jsonl_path: str,
    step: int,
    epoch_1based: int,
    beta: float,
    lr: float,
    H_cum: float,
    grad_norm: float,
    log: Callable[[str], None],
) -> Dict[str, float]:
    """Eval probe Δ, append jsonl, write a train.log summary. Restores train() if needed."""
    was_training = bool(policy_model.training)
    try:
        delta = eval_probe_deltas(
            policy_model,
            tokenizer,
            probe_loader,
            logp_c_ref,
            logp_r_ref,
            device,
            use_chat_template,
        )
        stats = summarize_margin(delta)
        a_mean = probe_a_mean(delta, beta)
        append_probe_jsonl(
            jsonl_path,
            step=step,
            epoch=epoch_1based,
            lr=lr,
            H_cum=H_cum,
            grad_norm=grad_norm,
            beta=beta,
            delta=delta,
            stats=stats,
            a_mean=a_mean,
        )
        log(
            format_probe_log_line(
                epoch_1based, step, stats, a_mean, lr, H_cum, grad_norm
            )
        )
        return stats
    finally:
        if was_training:
            policy_model.train()
