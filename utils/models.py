# -*- coding: utf-8 -*-
"""
Загрузка модели, reference и токенайзера для DPO (soft/hard).

По умолчанию отдельная копия базовой модели под ref не загружается: роль ref играет
тот же PeftModel с временно отключённым LoRA-адаптером (_PeftRefProxy). Это экономит
~14 GB VRAM на 7B в bf16 (≈6 GB на 3B, ≈8 GB на 4B) и освобождает место под больший
батч или отключение gradient checkpointing. Для кода, которому нужна «настоящая»
отдельная PreTrainedModel (например, TRL DPOTrainer), передавайте share_ref_with_policy=False.
"""
import os
from contextlib import contextmanager
from typing import List, Optional, Sequence, Union

import torch
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.config import BASE_MODEL_3B

# Пресеты target_modules для LoRA (Qwen2/Qwen3, подходит всем архитектурам с такими именами):
#   "all" — все линейные проекции (Q/K/V/O + MLP: gate/up/down); дефолт. Это стандартный
#           выбор в большинстве современных LoRA/QLoRA рецептов (в т.ч. TRL DPO). Даёт
#           примерно в 2–3 раза больше trainable-параметров при том же rank, но они всё
#           ещё < 1% базы; VRAM растёт в основном за счёт состояния AdamW (2× fp32 на
#           trainable-параметр) — десятки MB против сотен GB активаций, пренебрежимо.
#   "attn" — только Q/K/V/O (≈1/3 параметров трансформер-блока); минимум VRAM и шага,
#           но LoRA влияет только на веса внимания, не на MLP (который держит большую
#           часть «знания» модели). Это часто ограничивает выразительность адаптера
#           для preference-alignment (особенно soft/Bayes-DPO с мягкими целями). Оставлен
#           как опция для воспроизводимости со старыми прогонами.
_LORA_TARGETS_ATTN: List[str] = ["q_proj", "k_proj", "v_proj", "o_proj"]
_LORA_TARGETS_ALL: List[str] = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

LoraTargetSpec = Union[str, Sequence[str]]


def _resolve_lora_target_modules(spec: LoraTargetSpec) -> List[str]:
    if isinstance(spec, str):
        key = spec.lower()
        if key == "attn":
            return list(_LORA_TARGETS_ATTN)
        if key == "all":
            return list(_LORA_TARGETS_ALL)
        raise ValueError(
            f"Неизвестный пресет lora_target_modules={spec!r}; ожидается 'attn', 'all' "
            "или список имён модулей."
        )
    mods = [str(m).strip() for m in spec if str(m).strip()]
    if not mods:
        raise ValueError("lora_target_modules: передан пустой список")
    return mods


def resolve_peft_adapter_dir(resume_from: str) -> str:
    """
    Путь к каталогу с PEFT-адаптером (adapter_config.json + веса).

    Разрешён корень прогона (…/run_id) с подпапкой best/, как после train_dpo.
    """
    p = os.path.abspath(os.path.expanduser(resume_from))
    if not os.path.isdir(p):
        raise FileNotFoundError(f"Чекпоинт не найден (не каталог): {resume_from!r} -> {p}")
    if os.path.isfile(os.path.join(p, "adapter_config.json")):
        return p
    best = os.path.join(p, "best")
    if os.path.isfile(os.path.join(best, "adapter_config.json")):
        return best
    raise ValueError(
        f"В {p!r} нет PEFT-адаптера: ожидается adapter_config.json в этой папке или в best/."
    )


@contextmanager
def _temporarily_disable_gradient_checkpointing(model):
    """Если у model включён gradient checkpointing — выключает на время блока и
    восстанавливает в finally. Иначе — no-op. Безопасно при отсутствии у model
    методов *_disable/*_enable (тогда тоже no-op).

    Зачем: ref-проход у `_PeftRefProxy` всегда идёт под `torch.no_grad()`. Если
    у разделяемого с policy PeftModel включён gradient checkpointing, то
    `torch.utils.checkpoint.checkpoint(...)` всё равно вызывается на каждом слое и
    (а) бесполезно тратит время на recompute (backward не будет), (б) печатает
    UserWarning «None of the inputs have requires_grad=True. Gradients will be None»,
    потому что под no_grad хук `enable_input_require_grads` фактически no-op.
    Временное отключение убирает оба эффекта без влияния на тренируемый policy-форвард.
    """
    is_on = bool(getattr(model, "is_gradient_checkpointing", False))
    if not is_on or not hasattr(model, "gradient_checkpointing_disable"):
        yield
        return
    model.gradient_checkpointing_disable()
    try:
        yield
    finally:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()


class _PeftRefProxy:
    """
    Прозрачный прокси для «ref модели» поверх PeftModel.

    Любой forward/generate временно отключает LoRA-адаптер через peft_model.disable_adapter(),
    так что вызов эквивалентен forward по базовой модели без LoRA. Остальные атрибуты
    (config, generation_config, eval/train и т.д.) делегируются напрямую в исходный PeftModel.

    Это позволяет не держать вторую полную копию базовой модели в VRAM.

    Важные детали поведения:
      - forward/generate всегда под no_grad не делаются автоматически; вызывающий код
        сам отвечает за torch.no_grad() (как и раньше было с «настоящей» ref_model).
      - На время вызова дополнительно отключается gradient checkpointing исходного
        PeftModel (если был включён) — иначе ref-проход под no_grad крутит
        torch.utils.checkpoint впустую (recompute без backward) и печатает
        UserWarning про requires_grad=True. После выхода состояние checkpointing
        восстанавливается, тренируемый policy-форвард не затрагивается.
      - .generate() временно ставит config.use_cache=True, потому что policy_model.config.use_cache
        установлен в False для обучения; без KV-cache генерация в разы медленнее.
        (use_cache всё равно несовместим с gradient checkpointing, но мы и так его
        отключаем — двойная страховка.)
      - Прокси хранит только ссылку на peft_model — никаких дополнительных параметров
        в памяти, поэтому экономия ~= размер базовой модели.
    """

    def __init__(self, peft_model) -> None:
        if not hasattr(peft_model, "disable_adapter"):
            raise TypeError(
                "_PeftRefProxy ожидает PeftModel (с методом disable_adapter); "
                f"получено {type(peft_model).__name__}"
            )
        object.__setattr__(self, "_peft", peft_model)

    def __getattr__(self, name: str):
        # __getattr__ вызывается только если стандартный lookup не нашёл атрибут:
        # _peft хранится в __dict__, поэтому рекурсии нет.
        return getattr(self._peft, name)

    def __setattr__(self, name: str, value) -> None:
        if name == "_peft":
            object.__setattr__(self, name, value)
        else:
            setattr(self._peft, name, value)

    def __call__(self, *args, **kwargs):
        with _temporarily_disable_gradient_checkpointing(self._peft):
            with self._peft.disable_adapter():
                return self._peft(*args, **kwargs)

    def generate(self, *args, **kwargs):
        cfg = getattr(self._peft, "config", None)
        saved_cache = getattr(cfg, "use_cache", None) if cfg is not None else None
        if cfg is not None:
            cfg.use_cache = True
        try:
            with _temporarily_disable_gradient_checkpointing(self._peft):
                with self._peft.disable_adapter():
                    return self._peft.generate(*args, **kwargs)
        finally:
            if cfg is not None and saved_cache is not None:
                cfg.use_cache = saved_cache

    def __repr__(self) -> str:
        return f"_PeftRefProxy(over={type(self._peft).__name__})"


def load_models_and_tokenizer(
    model_name: str = BASE_MODEL_3B,
    use_lora: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_target_modules: LoraTargetSpec = "all",
    resume_from: Optional[str] = None,
    share_ref_with_policy: bool = True,
):
    """
    Загружает tokenizer, policy (база + LoRA или из чекпоинта) и reference.

    resume_from: путь к папке с adapter_config.json или к корню прогона с подпапкой best/.
    Если задан, tokenizer и policy (LoRA) грузятся из разрешённого адаптера.

    lora_target_modules: какие модули обёртывать LoRA-адаптером (только при use_lora=True
        и без resume_from — при resume target_modules берутся из сохранённого adapter_config.json):
          - "all" (по умолчанию): Q/K/V/O + MLP (gate_proj, up_proj, down_proj) — стандартный
                                 выбор современных LoRA/QLoRA-рецептов. MLP хранит большую часть
                                 «знания» модели, и без него LoRA влияет только на паттерн внимания,
                                 что в soft/Bayes-DPO часто недостаточно выразительно.
          - "attn":              Q/K/V/O — старое поведение репозитория (минимум trainable params,
                                 но ограниченная выразительность адаптера).
          - список строк:        кастомный набор имён модулей (должны существовать в архитектуре).
        Переключение влияет на число trainable-параметров (≈×2–3 при том же rank) и, соответственно,
        на память AdamW-состояния; на пик VRAM активаций почти не влияет, т.к. LoRA-добавки малы
        по сравнению с основным forward. ВАЖНО: смена пресета меняет семантику обучения, поэтому
        эксперименты с разным lora_target_modules напрямую несравнимы.

    share_ref_with_policy:
      - True (по умолчанию) и используется LoRA: отдельная базовая модель под ref
        НЕ загружается; возвращается _PeftRefProxy поверх policy_model, который на время
        forward/generate отключает LoRA-адаптер. Экономия памяти ≈ полный размер базы.
      - False: старое поведение — отдельная «замороженная» копия базы (нужна, если ref
        передаётся в код, который ожидает полноценный PreTrainedModel, например TRL DPOTrainer).
      - Если use_lora=False и resume_from=None, policy не является PeftModel, поэтому
        shared ref невозможен и всегда используется отдельная копия (с предупреждением в stderr).

    Returns:
        tokenizer, policy_model, ref_model, device_gpu
    """
    dtype_gpu = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    device_gpu = "cuda" if torch.cuda.is_available() else "cpu"

    if resume_from:
        adapter_dir = resolve_peft_adapter_dir(resume_from)
        tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    policy_is_peft = bool(resume_from) or use_lora
    load_separate_ref = (not share_ref_with_policy) or (not policy_is_peft)
    if not share_ref_with_policy and not policy_is_peft:
        # Явный запрос на отдельный ref при непоефтной policy — валидно, просто подсвечу.
        pass
    if share_ref_with_policy and not policy_is_peft:
        import sys

        print(
            "Warning: share_ref_with_policy=True запрошен, но policy не PeftModel "
            "(use_lora=False и resume_from=None) — грузим отдельный ref_model.",
            file=sys.stderr,
        )

    ref_model = None
    if load_separate_ref:
        ref_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype_gpu,
            device_map=device_gpu,
        )
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False

    policy_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype_gpu,
        device_map=device_gpu,
    )
    policy_model.config.use_cache = False

    if resume_from:
        # Без is_trainable=True PEFT оставляет LoRA замороженной (trainable params: 0)
        # и дообучение молча не идёт; чекпоинты часто сохраняются с inference_mode=True.
        policy_model = PeftModel.from_pretrained(
            policy_model, adapter_dir, is_trainable=True
        )
        policy_model.print_trainable_parameters()
    elif use_lora:
        resolved_targets = _resolve_lora_target_modules(lora_target_modules)
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=resolved_targets,
            bias="none",
        )
        policy_model = get_peft_model(policy_model, lora_config)
        import sys

        print(
            f"LoRA target_modules ({lora_target_modules!r} -> {resolved_targets})",
            file=sys.stderr,
        )
        policy_model.print_trainable_parameters()

    if hasattr(policy_model, "enable_input_require_grads"):
        policy_model.enable_input_require_grads()
        if hasattr(policy_model, "gradient_checkpointing_enable"):
            policy_model.gradient_checkpointing_enable()
    elif hasattr(policy_model, "gradient_checkpointing_enable"):
        import sys
        print("Warning: gradient checkpointing skipped (no enable_input_require_grads); training will use more VRAM.", file=sys.stderr)

    if ref_model is None:
        # policy_is_peft гарантирован (иначе выше load_separate_ref=True).
        ref_model = _PeftRefProxy(policy_model)

    return tokenizer, policy_model, ref_model, device_gpu
