# -*- coding: utf-8 -*-
"""
Загрузка eval_datasets, генерация (chat template Qwen) и подсчёт метрик удержания
(ref vs policy по gold). Используется train_dpo и eval_capability_retention.py.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm


def _build_messages_for_generation(instruction: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": instruction},
    ]


def _clear_unsupported_generation_config(model) -> None:
    if getattr(model, "generation_config", None) is not None:
        for key in ("top_p", "top_k"):
            if hasattr(model.generation_config, key):
                setattr(model.generation_config, key, None)


def generate_responses(
    tokenizer,
    model,
    device: str,
    instructions: List[str],
    max_new_tokens: int = 512,
    batch_size: int = 4,
    do_sample: bool = False,
    temperature: float = 0.6,
    max_prompt_tokens: int = 2048,
    desc: Optional[str] = None,
) -> List[str]:
    model.eval()
    _clear_unsupported_generation_config(model)
    out_texts: List[str] = []
    saved_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        it = range(0, len(instructions), batch_size)
        if desc:
            it = tqdm(it, desc=desc, leave=False)
        for i in it:
            batch = instructions[i : i + batch_size]
            messages_batch = [_build_messages_for_generation(x) for x in batch]
            texts = tokenizer.apply_chat_template(
                messages_batch,
                tokenize=False,
                add_generation_prompt=True,
            )
            if isinstance(texts, str):
                texts = [texts]
            inputs = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_prompt_tokens,
            ).to(device)
            in_len = int(inputs["input_ids"].shape[1])
            with torch.no_grad():
                gen = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature if do_sample else None,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            for row in range(gen.shape[0]):
                gen_ids = gen[row, in_len:]
                out_texts.append(
                    tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                )
    finally:
        tokenizer.padding_side = saved_padding_side
    return out_texts


def _discover_jsonl(eval_dir: Path) -> List[Path]:
    paths: List[Path] = []
    kdir = eval_dir / "knowledge"
    rdir = eval_dir / "reasoning"
    if kdir.is_dir():
        paths.extend(sorted(kdir.glob("*.jsonl")))
    if rdir.is_dir():
        paths.extend(sorted(rdir.glob("*.jsonl")))
    return paths


def _criterion_for_path(p: Path) -> str:
    parts = {x.lower() for x in p.parts}
    if "reasoning" in parts:
        return "reasoning"
    return "knowledge"


@dataclass
class EvalRow:
    example_id: str
    criterion: str
    source: str
    task: str
    path: str
    line_index: int
    prompt: str
    record: Dict[str, Any]


def load_eval_rows(eval_dir: Path) -> List[EvalRow]:
    rows: List[EvalRow] = []
    for path in _discover_jsonl(eval_dir):
        crit = _criterion_for_path(path)
        with open(path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                prompt = rec.get("input")
                if not isinstance(prompt, str) or not prompt.strip():
                    raise ValueError(f"{path}:{idx+1}: нет непустого поля input")
                src = str(rec.get("source", path.stem))
                task = str(rec.get("task", ""))
                eid = f"{path.name}:{idx}"
                rows.append(
                    EvalRow(
                        example_id=eid,
                        criterion=crit,
                        source=src,
                        task=task,
                        path=str(path),
                        line_index=idx,
                        prompt=prompt.strip(),
                        record=rec,
                    )
                )
    return rows


def _gold_boolq(rec: Dict[str, Any]) -> str:
    v = rec["label"]
    return "yes" if (v is True or v == "True" or str(v).lower() == "true") else "no"


def _parse_boolq(text: str) -> Optional[str]:
    t = text.lower()
    matches = list(re.finditer(r"\b(yes|no)\b", t))
    if not matches:
        return None
    return matches[-1].group(1)


def _gold_arc(rec: Dict[str, Any]) -> str:
    return str(rec["correct_label"]).strip().upper()[:1]


def _parse_arc(text: str) -> Optional[str]:
    m = re.search(r"\b([A-D])\b", text.upper())
    return m.group(1) if m else None


def _bbh_gold_kind(label: str) -> str:
    s = str(label).strip().lower()
    if s in ("yes", "no"):
        return "yesno"
    return "mc"


def _gold_bbh(rec: Dict[str, Any]) -> Tuple[str, str]:
    raw = str(rec["label"]).strip()
    kind = _bbh_gold_kind(raw)
    if kind == "yesno":
        return kind, raw.lower()
    m = re.search(r"\(([A-E])\)", raw, re.I)
    if m:
        return kind, m.group(1).upper()
    m2 = re.match(r"^([A-E])$", raw.strip(), re.I)
    if m2:
        return kind, m2.group(1).upper()
    return kind, raw.upper()


def _parse_bbh(text: str, kind: str) -> Optional[str]:
    t = text.strip()
    if kind == "yesno":
        m = list(re.finditer(r"\b(yes|no)\b", t.lower()))
        return m[-1].group(1) if m else None
    m = re.search(r"\(([A-E])\)", t)
    if m:
        return m.group(1).upper()
    m2 = re.search(r"\b([A-E])\b", t.upper())
    return m2.group(1) if m2 else None


def _gold_gsm8k_number(rec: Dict[str, Any]) -> Optional[str]:
    ans = rec.get("answer", "")
    if not isinstance(ans, str):
        return None
    m = re.search(r"####\s*(-?\d+(?:\.\d+)?)", ans.replace(",", ""))
    return m.group(1) if m else None


def _parse_gsm8k_number(text: str) -> Optional[str]:
    t = text.replace(",", "")
    m = re.search(r"####\s*(-?\d+(?:\.\d+)?)", t)
    if m:
        return m.group(1)
    nums = re.findall(r"-?\d+(?:\.\d+)?", t)
    return nums[-1] if nums else None


def _nums_equal(a: Optional[str], b: Optional[str]) -> bool:
    if a is None or b is None:
        return False
    try:
        fa, fb = float(a), float(b)
        return abs(fa - fb) < 1e-5
    except ValueError:
        return a.strip() == b.strip()


def gold_and_kind(rec: Dict[str, Any], source: str) -> Tuple[str, Any, str]:
    src = source.lower()
    if "boolq" in src:
        return "boolq", _gold_boolq(rec), "boolq"
    if "arc" in src:
        return "arc", _gold_arc(rec), "arc"
    if "bbh" in src or rec.get("bbh_task"):
        kind, g = _gold_bbh(rec)
        return "bbh", g, f"bbh_{kind}"
    if "gsm8k" in src:
        g = _gold_gsm8k_number(rec)
        return "gsm8k", g, "gsm8k"
    raise ValueError(f"Неизвестный source для gold: {source!r}, keys={list(rec.keys())}")


def parse_prediction(text: str, parse_kind: str, rec: Dict[str, Any]) -> Optional[str]:
    if parse_kind == "boolq":
        return _parse_boolq(text)
    if parse_kind == "arc":
        return _parse_arc(text)
    if parse_kind == "bbh_yesno":
        return _parse_bbh(text, "yesno")
    if parse_kind == "bbh_mc":
        return _parse_bbh(text, "mc")
    if parse_kind == "gsm8k":
        return _parse_gsm8k_number(text)
    raise ValueError(parse_kind)


def prediction_correct(
    pred: Optional[str], gold: Any, parse_kind: str
) -> Optional[bool]:
    if pred is None:
        return None
    if parse_kind == "gsm8k":
        if gold is None:
            return None
        return _nums_equal(pred, str(gold))
    return pred == gold


def _normalize_ws(s: str) -> str:
    return " ".join(s.split()).strip().lower()


def score_rows(
    rows: List[EvalRow],
    base_texts: List[str],
    policy_texts: List[str],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    assert len(rows) == len(base_texts) == len(policy_texts)
    per_file: Dict[str, Dict[str, int]] = {}
    per_crit: Dict[str, Dict[str, int]] = {
        "knowledge": {
            "n": 0,
            "base_ok": 0,
            "pol_ok": 0,
            "regress": 0,
            "improve": 0,
            "both_ok": 0,
            "both_bad": 0,
            "parse_fail_base": 0,
            "parse_fail_pol": 0,
            "raw_match": 0,
        },
        "reasoning": {
            "n": 0,
            "base_ok": 0,
            "pol_ok": 0,
            "regress": 0,
            "improve": 0,
            "both_ok": 0,
            "both_bad": 0,
            "parse_fail_base": 0,
            "parse_fail_pol": 0,
            "raw_match": 0,
        },
    }
    details: List[Dict[str, Any]] = []

    def _acc_bucket(bucket: Dict[str, int]) -> None:
        bucket["n"] += 1

    for row, bt, pt in zip(rows, base_texts, policy_texts):
        family, gold, pkind = gold_and_kind(row.record, row.source)
        pb = parse_prediction(bt, pkind, row.record)
        pp = parse_prediction(pt, pkind, row.record)
        cb = prediction_correct(pb, gold, pkind)
        cp = prediction_correct(pp, gold, pkind)

        crit = row.criterion
        for bucket in (per_crit[crit],):
            _acc_bucket(bucket)
            if cb is True:
                bucket["base_ok"] += 1
            elif cb is None:
                bucket["parse_fail_base"] += 1
            if cp is True:
                bucket["pol_ok"] += 1
            elif cp is None:
                bucket["parse_fail_pol"] += 1
            if cb is True and cp is False:
                bucket["regress"] += 1
            if cb is False and cp is True:
                bucket["improve"] += 1
            if cb is True and cp is True:
                bucket["both_ok"] += 1
            if cb is False and cp is False:
                bucket["both_bad"] += 1
            if _normalize_ws(bt) == _normalize_ws(pt):
                bucket["raw_match"] += 1

        fp = row.path
        if fp not in per_file:
            per_file[fp] = {
                "n": 0,
                "base_ok": 0,
                "pol_ok": 0,
                "regress": 0,
                "improve": 0,
                "parse_fail_base": 0,
                "parse_fail_pol": 0,
            }
        bf = per_file[fp]
        bf["n"] += 1
        if cb is True:
            bf["base_ok"] += 1
        elif cb is None:
            bf["parse_fail_base"] += 1
        if cp is True:
            bf["pol_ok"] += 1
        elif cp is None:
            bf["parse_fail_pol"] += 1
        if cb is True and cp is False:
            bf["regress"] += 1
        if cb is False and cp is True:
            bf["improve"] += 1

        details.append(
            {
                "example_id": row.example_id,
                "criterion": crit,
                "source": row.source,
                "family": family,
                "gold": gold,
                "parse_kind": pkind,
                "pred_base": pb,
                "pred_policy": pp,
                "correct_base": cb,
                "correct_policy": cp,
                "base_output": bt,
                "policy_output": pt,
            }
        )

    def _finalize(bucket: Dict[str, int]) -> Dict[str, Any]:
        n = max(1, bucket["n"])
        scored = bucket["n"] - bucket["parse_fail_base"]
        scored_p = bucket["n"] - bucket["parse_fail_pol"]
        return {
            "n": bucket["n"],
            "accuracy_base": bucket["base_ok"] / n,
            "accuracy_policy": bucket["pol_ok"] / n,
            "regressions": bucket["regress"],
            "improvements": bucket["improve"],
            "both_correct": bucket["both_ok"],
            "both_incorrect": bucket["both_bad"],
            "parse_fail_rate_base": bucket["parse_fail_base"] / n,
            "parse_fail_rate_policy": bucket["parse_fail_pol"] / n,
            "raw_output_match_rate": bucket["raw_match"] / n,
            "accuracy_base_on_parsed": (
                bucket["base_ok"] / scored if scored else 0.0
            ),
            "accuracy_policy_on_parsed": (
                bucket["pol_ok"] / scored_p if scored_p else 0.0
            ),
        }

    summary = {
        "overall": _finalize(
            {
                "n": per_crit["knowledge"]["n"] + per_crit["reasoning"]["n"],
                "base_ok": per_crit["knowledge"]["base_ok"] + per_crit["reasoning"]["base_ok"],
                "pol_ok": per_crit["knowledge"]["pol_ok"] + per_crit["reasoning"]["pol_ok"],
                "regress": per_crit["knowledge"]["regress"] + per_crit["reasoning"]["regress"],
                "improve": per_crit["knowledge"]["improve"] + per_crit["reasoning"]["improve"],
                "both_ok": per_crit["knowledge"]["both_ok"] + per_crit["reasoning"]["both_ok"],
                "both_bad": per_crit["knowledge"]["both_bad"] + per_crit["reasoning"]["both_bad"],
                "parse_fail_base": per_crit["knowledge"]["parse_fail_base"]
                + per_crit["reasoning"]["parse_fail_base"],
                "parse_fail_pol": per_crit["knowledge"]["parse_fail_pol"]
                + per_crit["reasoning"]["parse_fail_pol"],
                "raw_match": per_crit["knowledge"]["raw_match"]
                + per_crit["reasoning"]["raw_match"],
            }
        ),
        "knowledge": _finalize(per_crit["knowledge"]),
        "reasoning": _finalize(per_crit["reasoning"]),
        "per_file": {
            k: {
                **_finalize(
                    {
                        "n": v["n"],
                        "base_ok": v["base_ok"],
                        "pol_ok": v["pol_ok"],
                        "regress": v["regress"],
                        "improve": v["improve"],
                        "both_ok": 0,
                        "both_bad": 0,
                        "parse_fail_base": v["parse_fail_base"],
                        "parse_fail_pol": v["parse_fail_pol"],
                        "raw_match": 0,
                    }
                ),
                "path": k,
            }
            for k, v in per_file.items()
        },
    }
    return summary, details


def format_capability_retention_log_lines(
    summary: Dict[str, Any], epoch_display: str
) -> List[str]:
    lines = [
        f"=== capability retention (eval_datasets, gold) epoch {epoch_display} ===",
    ]
    for key in ("overall", "knowledge", "reasoning"):
        s = summary[key]
        if not isinstance(s, dict):
            continue
        if key != "overall" and int(s.get("n", 0)) == 0:
            continue
        lines.append(
            f"  [{key}] n={int(s['n'])} acc_ref={float(s['accuracy_base']):.4f} "
            f"acc_policy={float(s['accuracy_policy']):.4f} "
            f"regressions={int(s['regressions'])} improvements={int(s['improvements'])} "
            f"raw_match_rate={float(s['raw_output_match_rate']):.4f} "
            f"parse_fail_ref={float(s['parse_fail_rate_base']):.4f} "
            f"parse_fail_pol={float(s['parse_fail_rate_policy']):.4f}"
        )
    return lines


def run_retention_eval_pair(
    tokenizer,
    ref_model,
    policy_model,
    device: str,
    rows: List[EvalRow],
    cached_ref_texts: Optional[List[str]],
    max_new_tokens: int,
    batch_size: int,
    max_prompt_tokens: int,
    desc_ref: str,
    desc_pol: str,
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Генерация ref (кэшируемая) + policy, подсчёт метрик.
    Возвращает (summary, ref_texts для кэша).
    """
    prompts = [r.prompt for r in rows]
    if cached_ref_texts is None:
        ref_texts = generate_responses(
            tokenizer,
            ref_model,
            device,
            prompts,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            max_prompt_tokens=max_prompt_tokens,
            desc=desc_ref,
        )
    else:
        ref_texts = cached_ref_texts
    pol_texts = generate_responses(
        tokenizer,
        policy_model,
        device,
        prompts,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        max_prompt_tokens=max_prompt_tokens,
        desc=desc_pol,
    )
    summary, _ = score_rows(rows, ref_texts, pol_texts)
    return summary, ref_texts


def log_mlflow_capability_metrics(
    summary: Dict[str, Any], step: int, log_metric: Callable[[str, float, int], None],
) -> None:
    """log_metric(name, value, step) — как mlflow.log_metric."""
    for scope in ("overall", "knowledge", "reasoning"):
        s = summary.get(scope)
        if not isinstance(s, dict) or int(s.get("n", 0)) == 0:
            continue
        p = f"val_cap_{scope}"
        log_metric(f"{p}_acc_ref", float(s["accuracy_base"]), step)
        log_metric(f"{p}_acc_policy", float(s["accuracy_policy"]), step)
        log_metric(f"{p}_regressions", float(s["regressions"]), step)
        log_metric(f"{p}_improvements", float(s["improvements"]), step)
        log_metric(f"{p}_raw_match_rate", float(s["raw_output_match_rate"]), step)
        log_metric(f"{p}_parse_fail_ref", float(s["parse_fail_rate_base"]), step)
        log_metric(f"{p}_parse_fail_pol", float(s["parse_fail_rate_policy"]), step)
