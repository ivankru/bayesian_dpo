# -*- coding: utf-8 -*-
"""
Базовые константы обучения и валидации, которые редко меняют от эксперимента к эксперименту.

Часто тюнятые гиперпараметры (lr, beta, batch_size, epochs, λ, датасет) остаются в CLI;
здесь — лимиты токенов, chat template, якорная температура p_pred, val entropy и capability retention.
"""

# Лимиты длины (prompt и prompt+response); при нехватке VRAM уменьшать здесь.
MAX_PROMPT_LEN = 768  # можно поднять до 1024, если хватает памяти
MAX_FULL_LEN = 1536  # prompt+response

# Qwen-Instruct: log p через apply_chat_template (единый дефолт для всех датасетов).
USE_CHAT_TEMPLATE = True

# T>0 для σ((beta*diff)/T) в якорном режиме (lambda_full_epochs > 0); см. utils.loss.soft_dpo_loss.
P_PRED_TARGET_TEMPERATURE = 2.0

# Val response entropy: первые N промптов; 0 — отключить метрику.
VAL_ENTROPY_MAX_PROMPTS = 512
# Val KL-MC: независимых генераций на промпт для MC-оценки KL(policy||ref).
VAL_KL_MC_NUM_SAMPLES = 4
# Промптов за один generate; микробатч полного forward (VRAM).
VAL_ENTROPY_PROMPT_BATCH_SIZE = 3
VAL_ENTROPY_FORWARD_CHUNK_SIZE = 2
# Независимых генераций на промпт; L — первые токены ответа для оценки энтропии.
VAL_ENTROPY_NUM_SAMPLES = 8
VAL_ENTROPY_MAX_NEW_TOKENS = 128

# Capability retention (eval_datasets): limit=None — без обрезки числа примеров.
CAPABILITY_EVAL_LIMIT = None
CAPABILITY_EVAL_MAX_NEW_TOKENS = 256
CAPABILITY_EVAL_BATCH_SIZE = 2
CAPABILITY_EVAL_MAX_PROMPT_TOKENS = 2048

# Интервал (в глобальных шагах train) вывода средней train loss с её разбивкой
# по компонентам (soft/hard/bayes): log_msg один раз в LOG_INTERVAL шагов.
# Не влияет на метрики — чисто частота стрелок в train.log и stderr.
LOG_INTERVAL = 100
# Интервал (в шагах) вывода текущего lr и агрегатов расхождения p_target vs p_gt
# (target_shift/gap_abs), полезный для отладки λ-расписания и teacher-anchor.
# Шумит сильнее LOG_INTERVAL, поэтому держим реже.
LR_ALIGN_LOG_INTERVAL = 1000
