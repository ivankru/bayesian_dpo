---
name: bayesian-dpo-training
description: >-
  Trains and configures DPO / soft-DPO / Bayes-DPO for preference alignment in
  this repo (Qwen2.5 + LoRA, HelpSteer, UltraFeedback, HH-RLHF, OpenBMB).
  Use when the user runs or changes training, hyperparameters, datasets,
  checkpoints, reproducibility, or MLflow for bayesian_dpo.
---

# Bayesian DPO — обучение

## Точки входа

| Скрипт | Назначение |
|--------|------------|
| `hard_dpo_steer.py` | Hard DPO, chosen/rejected |
| `soft_dpo_steer.py` | Soft / Bayes DPO (`use_bayes`), train soft → val hard |
| `classic_dpo.py` | Классический DPO (если используется отдельно) |
| `*.sh` (`soft_hsteer.sh`, `hard_hhrlf.sh`, …) | Готовые команды запуска |

Ядро цикла: `utils/training.py` → `train_dpo`, одна эпоха — `train_one_epoch_dpo`.

## Режимы (`train_dpo`)

- **hard** — `hard_dpo_loss`, батч: `prompt`, `chosen`, `rejected`.
- **soft** — `soft_dpo_loss(use_bayes=False)`, батч: `resp1`, `resp2`, `p`, `p_bayes` (+ опционально `p_pred_cached`).
- **bayes** — как soft, но `use_bayes=True` (в лоссе целевая вероятность из бета-приора).

Валидация всегда **hard**: DPO loss, `val_logp_gap_mean` (среднее log π − log ref на chosen/rejected из val), pairwise NLL, pairwise accuracy. Лучший чекпоинт по **минимальному val NLL** → `output_dir/best/`.

## Датасеты (имена `--dataset`)

См. `utils/config.py`: `DPO_STEER_HARD_DATASET_CHOICES`, `DPO_STEER_SOFT_DATASET_CHOICES`. Сборка — `utils/datasets/*.py`.

## Гиперпараметры, которые часто трогают

- `beta`, `lr`, `batch_size`, `epochs`, `seed`
- Soft/Bayes: `alpha` (бета-приор для `p_bayes`), `lambda_min`, `lambda_schedule` (`linear` | `cosine`), `label_noise_prob`
- Длины: `MAX_PROMPT_LEN`, `MAX_FULL_LEN` в `utils/config.py`
- `use_chat_template`: для `hh_rlhf` в `soft_dpo_steer` по умолчанию согласовано с hard-скриптом; иначе обычно plain `prompt\nresponse` через `get_logps`

## Модели

- База: `BASE_MODEL_CHOICES` в `utils/config.py` (Qwen2.5-3B / 7B Instruct).
- Загрузка и LoRA: `utils/models.py` (`load_models_and_tokenizer`, `resume_from` для продолжения).

## Логи и артефакты

- Текстовый лог: `{output_dir}/train.log` (дублирует ключевые `print`).
- При `use_mlflow`: эксперимент по умолчанию `bayesian_dpo`, метрики `train_loss`, `train_logp_gap_mean`, `lr`, параметры из `train_dpo`; артефакт — `train.log`.

## Окружение

- Зависимости: см. `requirements.py` / окружение проекта.
- Для Hugging Face: задать `HF_TOKEN` в окружении (не коммитить токены).

## Практика для агента

1. Менять обучение — править скрипт запуска или `utils/training.py` / `utils/loss.py`, не дублировать цикл в новых файлах без причины.
2. Сравнивать прогоны — фиксировать `seed`, `output_dir`, и те же `MAX_*_LEN` / `use_chat_template`.
3. После изменения лосса или коллатора — проверить согласованность soft train и hard val (формат батча).

Детали метрик и строк логов — skill `bayesian-dpo-logs`.
