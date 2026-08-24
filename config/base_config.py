# -*- coding: utf-8 -*-
"""
Base training and validation constants that rarely change from experiment to experiment.

Frequently tuned hyperparameters (lr, beta, batch_size, epochs, λ, dataset) stay in the CLI;
here: token limits, chat template, anchor temperature p_pred, val entropy, and capability retention.
"""

# Length limits (prompt and prompt+response); reduce here if VRAM is tight.
MAX_PROMPT_LEN = 768  # can raise to 1024 if memory allows
MAX_FULL_LEN = 1536  # prompt+response

# Instruct chat template (Qwen / Phi-4-mini / …): log p via apply_chat_template.
USE_CHAT_TEMPLATE = True

# T>0 for σ((beta*diff)/T) in anchor mode (lambda_full_epochs > 0); see utils.loss.soft_dpo_loss.
P_PRED_TARGET_TEMPERATURE = 2.0

# Val response entropy: first N prompts; 0 disables the metric.
VAL_ENTROPY_MAX_PROMPTS = 512
# Val KL-MC: independent generations per prompt for MC estimate of KL(policy||ref).
VAL_KL_MC_NUM_SAMPLES = 8
# Prompts per generate call in MC KL(policy||ref) estimation.
VAL_KL_MC_PROMPT_BATCH_SIZE = 6
# Prompts per generate; microbatch for full forward (VRAM).
VAL_ENTROPY_PROMPT_BATCH_SIZE = 4
VAL_ENTROPY_FORWARD_CHUNK_SIZE = 2
# Independent generations per prompt; L — first response tokens for entropy estimate.
VAL_ENTROPY_NUM_SAMPLES = 8
VAL_ENTROPY_MAX_NEW_TOKENS = 128

# Capability retention (eval_datasets): limit=None — no cap on number of examples.
CAPABILITY_EVAL_LIMIT = None
# Was 256; lowered for all models after Phi-4-mini (3.8b) ran out of GPU
# memory / wall-time on long cap_ret generate (policy hitting the token cap).
CAPABILITY_EVAL_MAX_NEW_TOKENS = 212
CAPABILITY_EVAL_BATCH_SIZE = 2
CAPABILITY_EVAL_MAX_PROMPT_TOKENS = 2048

# Interval (in global train steps) for logging mean train loss and its breakdown
# by component (soft/hard/bayes): log_msg once every LOG_INTERVAL steps.
# Does not affect metrics — only how often lines appear in train.log and stderr.
LOG_INTERVAL = 100
# Interval (in steps) for logging current lr and aggregates of p_target vs p_gt mismatch
# (target_shift/gap_abs), useful for debugging λ schedule and teacher-anchor.
# Noisier than LOG_INTERVAL, so keep it less frequent.
LR_ALIGN_LOG_INTERVAL = 1000
