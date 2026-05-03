# -*- coding: utf-8 -*-
"""
Validation distributions of DPO logits: delta_theta, delta_ref, diff (margin).
"""
from typing import Dict, List, Optional

import numpy as np
import torch

from utils.config import MAX_FULL_LEN, MAX_PROMPT_LEN
from utils.metrics import get_logps


def compute_val_delta_distributions(
    policy_model,
    ref_model,
    tokenizer,
    val_loader,
    device: str,
    use_chat_template: bool = False,
    max_batches: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    For each (chosen, rejected) pair on val:
      - delta_theta = logp_theta(chosen|x) - logp_theta(rejected|x)
      - delta_ref   = logp_ref(chosen|x)   - logp_ref(rejected|x)
      - diff        = delta_theta - delta_ref  (DPO margin)

    Returns a dict of three np.ndarray over all collected pairs.
    """
    policy_model.eval()
    ref_model.eval()

    dtheta_chunks: List[np.ndarray] = []
    dref_chunks: List[np.ndarray] = []
    diff_chunks: List[np.ndarray] = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            prompts = batch["prompt"]
            chosen = batch["chosen"]
            rejected = batch["rejected"]

            logp_c_pi = get_logps(
                policy_model,
                tokenizer,
                prompts,
                chosen,
                device,
                MAX_PROMPT_LEN,
                MAX_FULL_LEN,
                use_chat_template=use_chat_template,
            )
            logp_r_pi = get_logps(
                policy_model,
                tokenizer,
                prompts,
                rejected,
                device,
                MAX_PROMPT_LEN,
                MAX_FULL_LEN,
                use_chat_template=use_chat_template,
            )
            logp_c_ref = get_logps(
                ref_model,
                tokenizer,
                prompts,
                chosen,
                device,
                MAX_PROMPT_LEN,
                MAX_FULL_LEN,
                use_chat_template=use_chat_template,
            )
            logp_r_ref = get_logps(
                ref_model,
                tokenizer,
                prompts,
                rejected,
                device,
                MAX_PROMPT_LEN,
                MAX_FULL_LEN,
                use_chat_template=use_chat_template,
            )

            delta_theta = logp_c_pi - logp_r_pi
            delta_ref = logp_c_ref - logp_r_ref
            diff = delta_theta - delta_ref

            dtheta_chunks.append(delta_theta.detach().cpu().numpy())
            dref_chunks.append(delta_ref.detach().cpu().numpy())
            diff_chunks.append(diff.detach().cpu().numpy())

    if not dtheta_chunks:
        empty = np.array([], dtype=np.float64)
        return {"delta_theta": empty, "delta_ref": empty, "diff": empty}

    return {
        "delta_theta": np.concatenate(dtheta_chunks),
        "delta_ref": np.concatenate(dref_chunks),
        "diff": np.concatenate(diff_chunks),
    }
