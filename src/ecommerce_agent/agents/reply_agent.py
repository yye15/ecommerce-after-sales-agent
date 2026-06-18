"""Customer reply generation agent."""

from __future__ import annotations

import json
from typing import Any

from ..llm_client import LLMClient
from ..prompts import REPLY_SYSTEM_PROMPT


def generate_reply(
    text: str,
    sentiment: dict[str, Any],
    categories: list[dict[str, Any]],
    risk: dict[str, Any],
    strategy: dict[str, Any],
    policy_context: dict[str, Any] | None = None,
    llm: LLMClient | None = None,
) -> str:
    llm = llm or LLMClient()
    policy_context = policy_context or {}
    payload = {
        "customer_text": text,
        "sentiment": sentiment,
        "categories": categories,
        "risk": risk,
        "strategy": strategy,
        "retrieved_policy_context": policy_context,
        "reply_instruction": "客服回复必须优先参考 retrieved_policy_context；如果引用政策，只能引用已检索到的政策，不要编造不存在的规则。",
    }
    llm_reply = llm.invoke_text(REPLY_SYSTEM_PROMPT, json.dumps(payload, ensure_ascii=False))
    if llm_reply:
        return llm_reply.strip()
    return fallback_reply(sentiment, categories, risk, strategy, policy_context)


def fallback_reply(
    sentiment: dict[str, Any],
    categories: list[dict[str, Any]],
    risk: dict[str, Any],
    strategy: dict[str, Any],
    policy_context: dict[str, Any] | None = None,
) -> str:
    category_text = "、".join(item["category"] for item in categories[:3])
    first_action = strategy.get("actions", [{}])[0].get("detail", "我们会尽快核实并跟进处理。")
    docs = (policy_context or {}).get("documents", [])
    policy_hint = ""
    if docs:
        policy_hint = f" 我们会参考《{docs[0].get('title', '售后政策')}》为您核实处理。"

    if risk.get("level") == "高风险":
        prefix = "您好，非常抱歉给您带来不好的体验，我们已将您的问题优先升级处理。"
    elif sentiment.get("overall_sentiment") == "POS":
        prefix = "您好，感谢您的认可和反馈。"
    else:
        prefix = "您好，非常抱歉给您带来不便。"
    return f"{prefix}关于您提到的{category_text}，{first_action}{policy_hint} 请您提供订单号，我们会尽快给出明确处理结果。"
