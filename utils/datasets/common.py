# -*- coding: utf-8 -*-
"""Общие хелперы для работы с датасетами DPO."""
import math
from typing import List, Dict, Any


def sigmoid(x: float) -> float:
    """Стабильный sigmoid: 1 / (1 + exp(-x))."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def ultrafeedback_message_to_response(messages: List[Dict[str, str]]) -> str:
    """Из списка сообщений {role, content} достаёт конкатенацию ответов assistant."""
    parts = [m["content"] for m in messages if m.get("role") == "assistant"]
    return "\n".join(parts).strip() if parts else ""
