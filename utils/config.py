# -*- coding: utf-8 -*-
"""
Общие константы для DPO/soft-DPO и оценки: лимиты длины, базовые модели.
"""
# Лимиты длины (prompt и prompt+response)
MAX_PROMPT_LEN = 768   # можно поднять до 1024, если хватает памяти
MAX_FULL_LEN = 1536   #1024# prompt+response

# Базовые модели
BASE_MODEL_3B = "Qwen/Qwen2.5-3B-Instruct"
BASE_MODEL_7B = "Qwen/Qwen2.5-7B-Instruct"
BASE_MODEL_CHOICES = {"3b": BASE_MODEL_3B, "7b": BASE_MODEL_7B}
