"""Robust JSON parsing helpers for LLM responses."""

from __future__ import annotations

import json
import re
from typing import Any


def extract_json_text(text: str) -> str:
    """Extract the most likely JSON object or array from raw model text."""
    if not text:
        return ""

    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S | re.I)
    if fenced:
        text = fenced.group(1)

    text = text.strip()
    object_start = text.find("{")
    array_start = text.find("[")
    starts = [idx for idx in (object_start, array_start) if idx != -1]
    if not starts:
        return text

    start = min(starts)
    if text[start] == "{":
        end = text.rfind("}")
    else:
        end = text.rfind("]")

    if end == -1 or end <= start:
        return text
    return text[start : end + 1]


def safe_json_loads(text: str) -> Any | None:
    """Return parsed JSON or None when parsing fails."""
    candidate = extract_json_text(text)
    if not candidate:
        return None
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None
