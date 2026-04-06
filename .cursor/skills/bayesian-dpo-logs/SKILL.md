---
name: bayesian-dpo-logs
description: >-
  Parses training and validation logs for bayesian_dpo: train.log format,
  step/epoch lines, MLflow metrics, and common failure signatures (CUDA OOM,
  NaN loss, HF download). Use when debugging runs, comparing experiments, or
  explaining log output.
---

# Bayesian DPO — анализ логов

## Где искать

- **Основной файл**: `{output_dir}/train.log` — всё важное дублируется из callback `log_msg` в `train_dpo` (`utils/training.py`).
- **Консоль**: tqdm для val/init; периодические строки обучения тоже идут в `train.log`.
- **MLflow** (если включён): метрики по `global_step`, параметры прогона, артефакт `train.log`.

## Структура прогона (по порядку в логе)

1. Заголовок режима: `=== Hard DPO ===` | `=== Soft DPO ===` | `=== Bayes DPO ===`.
2. Строка с моделью/датасетом/размерами: `Model: …, Dataset: …, train size: …, val size: …`.
3. `Старт train_dpo: mode=…, beta=…, lr=…, …` — полный набор гиперпараметров прогона.
4. `MAX_PROMPT_LEN=…, MAX_FULL_LEN=…, use_chat_template=…`.
5. **Initial (before training)**:
   - `validation DPO loss`
   - `validation KL(π||ref)`
   - `validation pair NLL`
   - `validation pair acc`
6. Для soft/bayes с `lambda_label < 1`: строки `Epoch k/n, lambda_label=…` (расписание по эпохам).
7. Во время эпохи каждые **100** шагов:  
   `[epoch e] step s train_loss=… kl_pi_ref=…`
8. Каждые **1000** шагов: `step s lr=…` (и в MLflow как `lr`).
9. После эпохи: блок `=== Epoch k ===` с теми же четырьмя валидационными метриками.
10. Улучшение NLL: `New best NLL … -> checkpoint saved: …/best`.

## Интерпретация метрик

- **train_loss / kl_pi_ref**: скользящее среднее за последние 100 шагов (не с начала прогона). Резкие всплески или NaN — смотреть lr, beta, батч, смешивание `lambda_label` / `p_pred_cached`.
- **validation DPO loss / KL**: средние по val в hard-режиме; полезно сравнивать до/после эпох.
- **pair NLL / pair acc**: согласованность предпочтений на val; **чекпоинт выбирается по минимальному val NLL**, не по acc.

## Типичные сигнатуры проблем

| Симптом | Куда смотреть |
|---------|----------------|
| CUDA OOM | уменьшить `batch_size`, `MAX_FULL_LEN` / `MAX_PROMPT_LEN` в `utils/config.py` |
| NaN / inf в loss | lr, beta, смешивание меток, данные (p в [0,1]) |
| Не грузится модель/датасет | `HF_TOKEN`, сеть, кэш HF |
| Расхождение с другим прогоном | `seed`, `use_chat_template`, те же `output_dir` и версия кода |
| MLflow пустой | флаг `use_mlflow`, `mlflow_tracking_uri`, доступ к серверу |

## Команды для быстрого разбора (локально)

```bash
# последние валидационные блоки
grep -E '^(=== Epoch|validation )' checkpoints/some_run/train.log

# динамика train
grep 'train_loss=' checkpoints/some_run/train.log | tail -n 50

# сохранения best
grep 'New best NLL' checkpoints/some_run/train.log
```

## Практика для агента

1. При «лог не сходится с ожиданием» — открыть `train.log` с начала и проверить блок **Initial** и первую эпоху.
2. Сравнивая два эксперимента — выровнять по строке `Старт train_dpo:` и по **Initial** метрикам (одинаковый seed и данные дают совпадение старта, см. комментарии в `soft_dpo_steer.py`).
3. Не путать **step** в логе с **epoch**; lr логируется реже (каждые 1000 шагов).
