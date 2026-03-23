# -*- coding: utf-8 -*-
"""UltraFeedback Binarized: hard DPO — {prompt, chosen, rejected}."""
from datasets import load_dataset, Dataset

from .common import ultrafeedback_message_to_response


def build_dpo_datasets_ultrafeedback():
    """
    UltraFeedback Binarized: train_prefs, test_prefs → {prompt, chosen, rejected}.
    """
    ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized")
    train_raw = ds["train_prefs"]
    val_raw = ds["test_prefs"]

    def convert(ex):
        prompt = ex["prompt"] if isinstance(ex["prompt"], str) else ex["prompt"].strip()
        chosen = ultrafeedback_message_to_response(ex["chosen"])
        rejected = ultrafeedback_message_to_response(ex["rejected"])
        if not chosen or not rejected:
            return None
        return {"prompt": prompt, "chosen": chosen, "rejected": rejected}

    train_processed = [out for ex in train_raw if (out := convert(ex)) is not None]
    val_processed = [out for ex in val_raw if (out := convert(ex)) is not None]

    return Dataset.from_list(train_processed), Dataset.from_list(val_processed)
