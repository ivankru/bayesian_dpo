# -*- coding: utf-8 -*-
"""Reproducibility: set seeds for random, numpy, torch.

Typical DPO training does NOT need full determinism — it only slows training
(cudnn autotuner off, deterministic kernels slower), and in bf16 transformers without convolutions
it barely changes numerics: for cuDNN a transformer is mostly matmuls with no meaningful
"algorithm search" in bf16. So by default we only fix Python/NumPy/Torch seeds
(enough for identical data, shuffle, and LoRA init) and leave
cuDNN and torch.use_deterministic_algorithms untouched — PyTorch defaults
(deterministic=False, benchmark=False).

For *strict* bitwise reproducibility (e.g. A/B with identical seed), pass deterministic=True. Note:
  1) cudnn.deterministic=True and cudnn.benchmark=False disable the autotuner;
  2) torch.use_deterministic_algorithms(True) forces deterministic kernels and
     errors on ops without a deterministic implementation — we use warn_only=True so training keeps going;
  3) deterministic CUBLAS needs env var CUBLAS_WORKSPACE_CONFIG; we set it here if unset.
Expect ~10–30% slowdown on LLM training; enable deliberately.
"""
import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """
    Fix seeds for Python random, NumPy, and Torch (+ CUDA when a GPU is present).

    deterministic: if True, enable strict deterministic cuDNN/CUBLAS behavior.
      Default False — enough for "same experiment with same seed"
      without the speed hit from disabling cudnn autotuner and forcing deterministic kernels.
      Use True only when bitwise reproducibility is required.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # CUBLAS needs workspace config for deterministic matmul on CUDA.
        # If the user already set a value, keep it.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # warn_only=True: ops without a deterministic implementation warn but do not crash mid-epoch.
        torch.use_deterministic_algorithms(True, warn_only=True)
