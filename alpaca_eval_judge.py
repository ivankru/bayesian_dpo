#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AlpacaEval: оценка DPO-моделей с судьёй Qwen2.5-14B-Instruct.

Загружает инструкции и ответы baseline (text_davinci_003 или GPT-4-turbo в режиме AlpacaEval 2.0),
генерирует ответы вашей моделью из чекпоинта, затем судья сравнивает
пары (baseline, model) и считается win rate и length-controlled win rate.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# URL эталонных данных AlpacaEval (instruction + output от text_davinci_003)
ALPACA_EVAL_DATA_URL = (
    "https://huggingface.co/datasets/tatsu-lab/alpaca_eval/resolve/main/alpaca_eval.json"
)
# AlpacaEval 2.0: reference-ответы GPT-4 (gpt-4-1106-preview / GPT-4-turbo)
ALPACA_EVAL_V2_REFERENCE_URL = (
    "https://huggingface.co/datasets/tatsu-lab/alpaca_eval/resolve/main/alpaca_eval_gpt4_baseline.json"
)

JUDGE_MODEL = "Qwen/Qwen2.5-14B-Instruct"
# Базовые модели для кандидата (оценка без LoRA или база для LoRA)
BASE_MODEL_3B = "Qwen/Qwen2.5-3B-Instruct"
BASE_MODEL_7B = "Qwen/Qwen2.5-7B-Instruct"
BASE_MODEL = BASE_MODEL_3B  # по умолчанию
BASE_MODEL_CHOICES = {"3b": BASE_MODEL_3B, "7b": BASE_MODEL_7B}

# Шаблон промпта судьи (AlpacaEval-style, ChatML для Qwen)
JUDGE_SYSTEM = (
    "You are a helpful assistant, that ranks models by the quality of their answers."
)
JUDGE_USER_TEMPLATE = """I want you to create a leaderboard of different of large-language models. To do so, I will give you the instructions (prompts) given to the models, and the responses of two models. Please rank the models based on which responses would be preferred by humans. All inputs and outputs should be python dictionaries.

Here is the prompt:
{
 "instruction": "__INSTRUCTION__",
}

Here are the outputs of the models:
[
 {
 "model": "model_1",
 "answer": "__OUTPUT_1__"
 },
 {
 "model": "model_2",
 "answer": "__OUTPUT_2__"
 }
]

Now please rank the models by the quality of their answers, so that the model with rank 1 has the best output. Then return a list of the model names and ranks, i.e., produce the following output:
[
 {{'model': 'model_1', 'rank': 1}},
 {{'model': 'model_2', 'rank': 2}}
]

Your response must be a valid Python dictionary and should contain nothing else because we will directly execute it in Python. Please provide the ranking that the majority of humans would give."""


def load_alpaca_eval_data(path: Optional[str] = None) -> List[Dict[str, str]]:
    """Загружает данные AlpacaEval: список {instruction, output} (baseline)."""
    if path and os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        try:
            from urllib.request import urlopen
            with urlopen(ALPACA_EVAL_DATA_URL, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            raise FileNotFoundError(
                f"Не найден файл {path!r} и не удалось скачать {ALPACA_EVAL_DATA_URL}: {e}"
            ) from e

    out = []
    for item in data:
        if isinstance(item, dict) and "instruction" in item and "output" in item:
            out.append({
                "instruction": item["instruction"],
                "output": item["output"],
            })
    return out


def _load_json_from_path_or_url(path: Optional[str], url: str, desc: str) -> List[Dict[str, Any]]:
    """Загружает JSON: из файла path или по URL."""
    if path and os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    try:
        from urllib.request import urlopen
        with urlopen(url, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        raise FileNotFoundError(
            f"{desc}: не найден файл {path!r} и не удалось скачать {url}: {e}"
        ) from e


def load_alpaca_eval_v2_data(
    data_dir: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Загружает данные в формате AlpacaEval 2.0: инструкции из alpaca_eval.json,
    reference-ответы — из alpaca_eval_gpt4_baseline.json (GPT-4-turbo / gpt-4-1106-preview).
    Возвращает список {instruction, output}, где output — ответ reference-модели.
    """
    if data_dir and os.path.isdir(data_dir):
        path_instructions = os.path.join(data_dir, "alpaca_eval.json")
        path_reference = os.path.join(data_dir, "alpaca_eval_gpt4_baseline.json")
    else:
        path_instructions = path_reference = None

    instructions_data = _load_json_from_path_or_url(
        path_instructions, ALPACA_EVAL_DATA_URL, "AlpacaEval instructions"
    )
    reference_data = _load_json_from_path_or_url(
        path_reference, ALPACA_EVAL_V2_REFERENCE_URL, "AlpacaEval 2.0 GPT-4 reference"
    )

    # Нормализуем в списки {instruction, output}
    instructions_list = []
    for item in instructions_data:
        if isinstance(item, dict) and "instruction" in item:
            instructions_list.append({
                "instruction": item["instruction"],
                "output": item.get("output", ""),
            })

    ref_by_instruction: Dict[str, str] = {}
    for item in reference_data:
        if isinstance(item, dict) and "instruction" in item and "output" in item:
            ref_by_instruction[item["instruction"]] = item["output"]

    # Сопоставляем по instruction (порядок — по instructions_list)
    out = []
    for rec in instructions_list:
        inst = rec["instruction"]
        ref_out = ref_by_instruction.get(inst)
        if ref_out is None:
            raise ValueError(
                f"AlpacaEval 2.0: для инструкции не найден reference в alpaca_eval_gpt4_baseline.json: {inst[:80]}..."
            )
        out.append({"instruction": inst, "output": ref_out})
    return out


def load_candidate_model(
    checkpoint_dir: str,
    base_model: str = BASE_MODEL,
    device: Optional[str] = None,
):
    """Загружает модель из чекпоинта (база + LoRA). Токенайзер берётся из base_model, чтобы избежать ошибок при неполных файлах в чекпоинте."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map=device,
    )
    model = PeftModel.from_pretrained(model, checkpoint_dir)
    model.eval()
    return tokenizer, model, device


def load_base_model(
    base_model: str = BASE_MODEL,
    device: Optional[str] = None,
):
    """Загружает базовую модель без LoRA (для оценки Qwen2.5-3B-Instruct без finetuning)."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map=device,
    )
    model.eval()
    return tokenizer, model, device


def load_judge_model(device: Optional[str] = None):
    """Загружает модель-судью Qwen2.5-14B-Instruct."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        JUDGE_MODEL,
        torch_dtype=dtype,
        device_map=device,
    )
    model.eval()
    return tokenizer, model, device


def build_messages_for_generation(instruction: str) -> List[Dict[str, str]]:
    """Сообщения в формате chat для генерации ответа на инструкцию."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": instruction},
    ]


def _clear_unsupported_generation_config(model) -> None:
    """Убирает top_p/top_k из generation_config, чтобы не было предупреждений о невалидных флагах."""
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
    batch_size: int = 1,
    do_sample: bool = False,
    temperature: float = 0.6,
) -> List[str]:
    """Генерирует ответы модели на список инструкций."""
    model.eval()
    _clear_unsupported_generation_config(model)
    all_outputs = []

    for i in range(0, len(instructions), batch_size):
        batch_instructions = instructions[i : i + batch_size]
        messages_batch = [build_messages_for_generation(inst) for inst in batch_instructions]

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
            max_length=2048,
        ).to(device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Вырезаем только сгенерированную часть (после промпта)
        for j, (inp_ids, out_ids) in enumerate(zip(inputs.input_ids, out)):
            prompt_len = inp_ids.size(0)
            gen_ids = out_ids[prompt_len:]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            all_outputs.append(text)

    return all_outputs


def build_judge_messages(instruction: str, output_1: str, output_2: str) -> List[Dict[str, str]]:
    """Сообщения для судьи: instruction и два ответа (model_1, model_2)."""
    user_text = (
        JUDGE_USER_TEMPLATE.replace("__INSTRUCTION__", instruction)
        .replace("__OUTPUT_1__", output_1)
        .replace("__OUTPUT_2__", output_2)
    )
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": user_text},
    ]


def parse_judge_ranking(response: str) -> Optional[Dict[str, int]]:
    """Парсит ответ судьи: {'model_1': rank, 'model_2': rank} или None."""
    # Ищем список вида [{'model': 'model_1', 'rank': 1}, ...]
    response = response.strip()
    # Убираем markdown code block если есть
    if "```" in response:
        for part in response.split("```"):
            if "model" in part and "rank" in part:
                response = part
                break
    # Найти список словарей
    list_match = re.search(r"\[\s*\{[^]]+\}\s*\]", response, re.DOTALL)
    if not list_match:
        return None
    try:
        # Заменить одинарные кавычки для совместимости с JSON
        s = list_match.group(0).replace("'", '"')
        arr = json.loads(s)
        ranking = {}
        for item in arr:
            name = item.get("model")
            rank = item.get("rank")
            if name is not None and rank is not None:
                ranking[name] = int(rank)
        return ranking if ranking else None
    except (json.JSONDecodeError, TypeError):
        return None


def run_judge(
    tokenizer,
    model,
    device: str,
    instruction: str,
    output_baseline: str,
    output_candidate: str,
    model_1_name: str = "baseline",
    model_2_name: str = "candidate",
    max_new_tokens: int = 96,
) -> Tuple[Optional[int], str]:
    """
    Судья сравнивает output_baseline (model_1) и output_candidate (model_2).
    Возвращает (rank_candidate, raw_response).
    rank_candidate: 1 = candidate лучше, 2 = baseline лучше, None = не удалось распарсить.
    """
    # В шаблоне судьи имена model_1 и model_2 — подставляем наши имена в парсер не меняем, в промпте можно оставить model_1/model_2
    messages = build_judge_messages(instruction, output_baseline, output_candidate)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=4096,
    ).to(device)

    _clear_unsupported_generation_config(model)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    gen_ids = out[0][inputs.input_ids.size(1) :]
    raw = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    ranking = parse_judge_ranking(raw)

    if ranking is None:
        return None, raw
    # В промпте мы передаём "model_1" и "model_2"; candidate у нас был output_2 = model_2
    rank_candidate = ranking.get("model_2")
    return rank_candidate, raw


def compute_win_rate(
    eval_data: List[Dict[str, str]],
    candidate_outputs: List[str],
    judge_tokenizer,
    judge_model,
    device: str,
    max_evals: Optional[int] = None,
) -> Tuple[float, float, float, List[Dict[str, Any]]]:
    """
    Запускает судью по всем парам, считает win / tie / loss.
    Возвращает (win_rate, tie_rate, loss_rate), и список записей для отчёта.
    """
    n = len(eval_data)
    if max_evals is not None:
        n = min(n, max_evals)
    wins = ties = losses = 0
    results = []

    for i in range(n):
        item = eval_data[i]
        inst = item["instruction"]
        baseline_out = item["output"]
        cand_out = candidate_outputs[i]

        rank_cand, raw = run_judge(
            judge_tokenizer,
            judge_model,
            device,
            inst,
            baseline_out,
            cand_out,
        )

        if rank_cand == 1:
            wins += 1
        elif rank_cand == 2:
            losses += 1
        else:
            ties += 1

        results.append({
            "index": i,
            "instruction": inst,
            "output_baseline": baseline_out,
            "output_candidate": cand_out,
            "rank_candidate": rank_cand,
            "raw_judge": raw,
        })

    total = wins + ties + losses
    win_rate = wins / total if total else 0.0
    tie_rate = ties / total if total else 0.0
    loss_rate = losses / total if total else 0.0
    return win_rate, tie_rate, loss_rate, results


def win_rate_standard_error(win_rate: float, n: int) -> float:
    """Стандартная ошибка для доли (binomial SE). AlpacaEval использует для доверительных интервалов."""
    if n <= 0:
        return 0.0
    return math.sqrt(win_rate * (1.0 - win_rate) / n)


def print_metrics(
    win_rate: float,
    tie_rate: float,
    loss_rate: float,
    n: int,
    length_controlled_win_rate: Optional[float] = None,
    model_name: str = "candidate",
    baseline_name: str = "baseline",
) -> None:
    """Печатает основные метрики AlpacaEval на экран."""
    se = win_rate_standard_error(win_rate, n)
    print("\n" + "=" * 56, flush=True)
    print("  ALPACAEVAL METRICS", flush=True)
    print("=" * 56, flush=True)
    print(f"  Model:          {model_name}", flush=True)
    print(f"  Baseline:       {baseline_name}", flush=True)
    print(f"  N samples:      {n}", flush=True)
    print("-" * 56, flush=True)
    print(f"  Win rate (vs {baseline_name}):  {win_rate:.2%}  (±{se:.2%} SE)", flush=True)
    print(f"  Tie rate:                   {tie_rate:.2%}", flush=True)
    print(f"  Loss rate:                   {loss_rate:.2%}", flush=True)
    if length_controlled_win_rate is not None:
        print(f"  Length-controlled win rate: {length_controlled_win_rate:.2%}", flush=True)
    print("=" * 56 + "\n", flush=True)


def compute_length_controlled_win_rate(
    eval_data: List[Dict[str, str]],
    candidate_outputs: List[str],
    results: List[Dict[str, Any]],
    length_ratio_max: float = 1.1,
) -> float:
    """
    Length-controlled win rate: только пары, где длина ответа кандидата
    не больше чем у baseline * length_ratio_max. Использует уже посчитанные results.
    """
    wins = losses = ties = 0
    for i, item in enumerate(eval_data):
        if i >= len(results):
            break
        baseline_out = item["output"]
        cand_out = candidate_outputs[i]
        if len(cand_out) > len(baseline_out) * length_ratio_max:
            continue
        rank_cand = results[i].get("rank_candidate")
        if rank_cand == 1:
            wins += 1
        elif rank_cand == 2:
            losses += 1
        else:
            ties += 1
    total = wins + ties + losses
    return (wins / total) if total else 0.0


def _run_official_alpaca_eval(
    eval_data: List[Dict[str, str]],
    model_outputs_for_lib: List[Dict[str, Any]],
    out_dir: str,
    model_name: str,
    max_instances: Optional[int],
    log: Any,
) -> None:
    """
    Оценка через официальную библиотеку alpaca_eval (pip install alpaca-eval).
    Использует аннотатор по умолчанию (GPT-4 и т.д.); нужен OPENAI_API_KEY.
    """
    try:
        import pandas as pd
        import alpaca_eval
    except ImportError as e:
        log("Ошибка: для --alpaca-eval-lib нужна библиотека alpaca_eval.")
        log("Установите: pip install alpaca-eval")
        raise SystemExit(1) from e

    # reference_outputs в том же порядке, что и model_outputs (baseline из наших данных)
    reference_outputs = [
        {"instruction": d["instruction"], "output": d["output"], "dataset": "helpful_base"}
        for d in eval_data
    ]

    log("Running official alpaca_eval.evaluate() (default annotator, e.g. GPT-4)...")
    df_leaderboard, annotations = alpaca_eval.evaluate(
        model_outputs=model_outputs_for_lib,
        reference_outputs=reference_outputs,
        name=model_name,
        output_path=Path(out_dir),
        is_return_instead_of_print=True,
        max_instances=max_instances,
    )

    # Печать метрик на экран (как выводит сама библиотека)
    if model_name in df_leaderboard.index:
        row = df_leaderboard.loc[model_name]
    elif len(df_leaderboard) == 1:
        row = df_leaderboard.iloc[0]
    else:
        row = None
    if row is not None:
        print("\n" + "=" * 56, flush=True)
        print("  ALPACAEVAL (official library)", flush=True)
        print("=" * 56, flush=True)
        print(f"  Model:     {model_name}", flush=True)
        print("-" * 56, flush=True)
        for key in ["win_rate", "standard_error", "length_controlled_winrate", "n_total", "avg_length"]:
            if key in row and pd.notna(row.get(key)):
                val = row[key]
                if isinstance(val, float) and 0 <= val <= 1 and "rate" in key:
                    print(f"  {key}: {val:.2%}", flush=True)
                else:
                    print(f"  {key}: {val}", flush=True)
        print("=" * 56 + "\n", flush=True)
    else:
        print(df_leaderboard.to_string(), flush=True)

    log(f"Results saved to {out_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="AlpacaEval: оценка модели судьёй Qwen2.5-14B-Instruct"
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=str,
        default=None,
        help="Путь к чекпоинту модели (например checkpoints/hard_dpo_steer/best). При --base-only не используется.",
    )
    parser.add_argument(
        "--base-only",
        action="store_true",
        help="Оценить базовую модель без LoRA (--checkpoint при этом не имеет смысла и игнорируется).",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        choices=list(BASE_MODEL_CHOICES.keys()),
        default="3b",
        help="Базовая модель для кандидата: 3b (Qwen2.5-3B-Instruct) или 7b (Qwen2.5-7B-Instruct). По умолчанию: 3b.",
    )
    parser.add_argument(
        "--data",
        "-d",
        type=str,
        default=None,
        help="Путь к alpaca_eval.json (instruction + baseline output). По умолчанию — скачать с HF.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Папка для сохранения: логи, candidate outputs JSON.",
    )
    parser.add_argument(
        "--max-evals",
        type=int,
        default=None,
        help="Макс. число примеров для оценки (для отладки).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Макс. токенов генерации ответа модели.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Размер батча для генерации ответов модели.",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Использовать сэмплирование при генерации (temperature 0.6).",
    )
    parser.add_argument(
        "--length-controlled",
        action="store_true",
        help="Дополнительно вывести length-controlled win rate.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Устройство (cuda/cpu). По умолчанию cuda при наличии.",
    )
    parser.add_argument(
        "--alpaca-eval-lib",
        action="store_true",
        help="Оценка через официальную библиотеку alpaca_eval (GPT-4 аннотатор, нужен OPENAI_API_KEY). Без этого флага используется локальный судья Qwen2.5-14B.",
    )
    parser.add_argument(
        "--alpaca2",
        action="store_true",
        help="AlpacaEval 2.0: официальный формат данных, reference — GPT-4-turbo (alpaca_eval_gpt4_baseline.json). Сравнение: ваша модель vs GPT-4-turbo. Всегда считаются win rate и length-controlled win rate.",
    )
    args = parser.parse_args()

    if args.base_only and args.checkpoint:
        parser.error("Нельзя указывать одновременно --checkpoint и --base-only. Уберите один из параметров.")
    if not args.base_only and not args.checkpoint:
        parser.error("Укажите --checkpoint или используйте --base-only для оценки базовой модели.")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    base_model_id = BASE_MODEL_CHOICES[args.base_model]
    if args.base_only:
        model_name_for_log = base_model_id
        out_dir = args.output or "alpaca_eval_base"
    else:
        model_name_for_log = args.checkpoint
        out_dir = args.output or os.path.join(args.checkpoint, "alpaca_eval")
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "eval.log")

    def log(msg: str) -> None:
        print(msg, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    baseline_name = "GPT-4-turbo (reference)" if args.alpaca2 else "text_davinci_003"
    log("=== AlpacaEval (Qwen2.5-14B-Instruct judge) ===")
    log(f"Mode: {'AlpacaEval 2.0 (vs GPT-4-turbo reference)' if args.alpaca2 else 'AlpacaEval (vs davinci-003)'}")
    log(f"Model: {model_name_for_log}" + (" (base, no LoRA)" if args.base_only else ""))
    log(f"Data: {args.data or (ALPACA_EVAL_V2_REFERENCE_URL if args.alpaca2 else ALPACA_EVAL_DATA_URL)}")
    log(f"Device: {device}")

    # 1) Загрузка данных
    if args.alpaca2:
        eval_data = load_alpaca_eval_v2_data(args.data if args.data and os.path.isdir(args.data) else None)
    else:
        eval_data = load_alpaca_eval_data(args.data)
    n_total = len(eval_data)
    if args.max_evals is not None:
        eval_data = eval_data[: args.max_evals]
    n = len(eval_data)
    log(f"Examples: {n} (total in data: {n_total})")

    # 2) Загрузка модели-кандидата и генерация
    if args.base_only:
        log(f"Loading base model (no LoRA): {base_model_id}...")
        cand_tokenizer, cand_model, _ = load_base_model(base_model=base_model_id, device=device)
    else:
        log(f"Loading candidate model from checkpoint (base: {base_model_id})...")
        cand_tokenizer, cand_model, _ = load_candidate_model(
            args.checkpoint, base_model=base_model_id, device=device
        )
    instructions = [d["instruction"] for d in eval_data]
    log("Generating candidate responses...")
    candidate_outputs = generate_responses(
        cand_tokenizer,
        cand_model,
        device,
        instructions,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        do_sample=args.do_sample,
    )
    # Освобождаем память от модели-кандидата перед загрузкой судьи
    del cand_model
    torch.cuda.empty_cache() if device == "cuda" else None

    # Сохраняем ответы кандидата (формат alpaca_eval: instruction, output, dataset, generator)
    model_outputs_for_lib = [
        {
            "instruction": d["instruction"],
            "output": candidate_outputs[i],
            "dataset": "helpful_base",
            "generator": model_name_for_log,
        }
        for i, d in enumerate(eval_data)
    ]
    out_json = os.path.join(out_dir, "candidate_outputs.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            [
                {"instruction": x["instruction"], "output": x["output"], "dataset": x["dataset"], "generator": x["generator"]}
                for x in model_outputs_for_lib
            ],
            f,
            ensure_ascii=False,
            indent=2,
        )
    log(f"Saved candidate outputs to {out_json}")

    if args.alpaca_eval_lib:
        # Оценка через официальную библиотеку alpaca_eval (GPT-4 и т.д.)
        _run_official_alpaca_eval(
            eval_data=eval_data,
            model_outputs_for_lib=model_outputs_for_lib,
            out_dir=out_dir,
            model_name=model_name_for_log,
            max_instances=args.max_evals,
            log=log,
        )
        return

    # 3) Загрузка судьи и оценка (локальный Qwen2.5-14B)
    log("Loading judge model (Qwen2.5-14B-Instruct)...")
    judge_tokenizer, judge_model, _ = load_judge_model(device=device)
    log("Running judge...")
    win_rate, tie_rate, loss_rate, results = compute_win_rate(
        eval_data,
        candidate_outputs,
        judge_tokenizer,
        judge_model,
        device,
        max_evals=None,
    )

    lc_wr = None
    if args.length_controlled or args.alpaca2:
        lc_wr = compute_length_controlled_win_rate(
            eval_data,
            candidate_outputs,
            results,
        )

    log("")
    log("--- Results ---")
    log(f"Win rate (vs {baseline_name}): {win_rate:.2%}")
    log(f"Tie rate:               {tie_rate:.2%}")
    log(f"Loss rate:              {loss_rate:.2%}")
    if lc_wr is not None:
        log(f"Length-controlled win rate: {lc_wr:.2%}")

    results_dict = {
        "win_rate": win_rate,
        "tie_rate": tie_rate,
        "loss_rate": loss_rate,
        "n": n,
        "checkpoint": model_name_for_log,
        "baseline": baseline_name,
    }
    if lc_wr is not None:
        results_dict["length_controlled_win_rate"] = lc_wr
    with open(os.path.join(out_dir, "judge_results.json"), "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2)
    # Сохраняем примеры для разбора: инструкция, ответы, вердикт судьи
    examples_path = os.path.join(out_dir, "judge_examples.json")
    with open(examples_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    log(f"Judge examples (instruction, output_candidate, output_baseline, raw_judge) saved to {examples_path}")
    log(f"Results saved to {out_dir}")

    print_metrics(  
        win_rate=win_rate,
        tie_rate=tie_rate,
        loss_rate=loss_rate,
        n=n,
        length_controlled_win_rate=lc_wr,
        model_name=model_name_for_log,
        baseline_name=baseline_name,
    )


if __name__ == "__main__":
    main()
