# -*- coding: utf-8 -*-
"""Воспроизводимость: установка seed для random, numpy, torch.

Обычному DPO-обучению НЕ нужна полная детерминированность — она только замедляет
обучение (cudnn autotuner выключен, детерминированные алгоритмы медленнее), а в
bf16-трансформерах без свёрток вообще почти не даёт ничего: на уровне cuDNN
трансформер — это набор matmul, у которых в bf16 всё равно нет «перебора
алгоритмов». Поэтому по умолчанию мы только фиксируем seed для Python/NumPy/Torch
(этого хватает для одинаковых данных, shuffle и инициализации LoRA), а флаги
cuDNN и torch.use_deterministic_algorithms не трогаем — остаются дефолты PyTorch
(deterministic=False, benchmark=False).

Если нужна *строгая* битовая воспроизводимость (например, для A/B-сравнения с
идентичным seed), передавайте deterministic=True. Имейте в виду:
  1) cudnn.deterministic=True и cudnn.benchmark=False отключат autotuner;
  2) torch.use_deterministic_algorithms(True) форсит детерминированные ядра и
     поднимает исключение на операциях, у которых детерминированной реализации
     нет — поэтому оборачиваем warn_only=True, чтобы не валить обучение;
  3) для детерминированного CUBLAS нужна переменная окружения
     CUBLAS_WORKSPACE_CONFIG, её мы выставляем здесь же (если ещё не задана).
Ожидаемое замедление — 10–30% на LLM-обучении, поэтому включать осознанно.
"""
import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """
    Фиксирует seed для Python random, NumPy и Torch (+ CUDA-seed при наличии GPU).

    deterministic: если True, включает строгий детерминированный режим cuDNN/CUBLAS.
      По умолчанию False — достаточный для «одинаковых экспериментов с одинаковым
      seed», но без штрафа по скорости от отключения cuDNN autotuner и форса
      детерминированных ядер. Включайте True только когда битовая
      воспроизводимость действительно нужна.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # CUBLAS требует workspace config для детерминированности matmul на CUDA.
        # Если пользователь уже задал своё значение — уважаем его.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # warn_only=True: на операциях без детерминированной реализации
        # будет warning, но обучение не упадёт — это безопаснее, чем исключение
        # в середине эпохи.
        torch.use_deterministic_algorithms(True, warn_only=True)
