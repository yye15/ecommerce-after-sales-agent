"""Shared TypedDict schemas used by agent nodes."""

from __future__ import annotations

from typing import Any, TypedDict


class CustomerSupportState(TypedDict, total=False):
    raw_text: str
    cleaned_text: str
    source_channel: str
    customer_intent: str
    sentiment: dict[str, Any]
    issue_categories: list[dict[str, Any]]
    risk: dict[str, Any]
    policy_context: dict[str, Any]
    escalation_required: bool
    escalation_note: str
    strategy: dict[str, Any]
    reply: str
    final_result: dict[str, Any]
    errors: list[str]
