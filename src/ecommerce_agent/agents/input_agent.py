"""Input normalization node."""

from __future__ import annotations

import re


def normalize_input(raw_text: str, source_channel: str = "review") -> dict:
    cleaned = re.sub(r"\s+", " ", raw_text or "").strip()
    if not cleaned:
        customer_intent = "empty"
    elif any(word in cleaned for word in ["退款", "退货", "换货", "赔偿"]):
        customer_intent = "after_sales"
    elif any(word in cleaned for word in ["投诉", "差评", "举报", "维权"]):
        customer_intent = "complaint"
    else:
        customer_intent = "feedback"

    return {
        "cleaned_text": cleaned,
        "source_channel": source_channel,
        "customer_intent": customer_intent,
    }
