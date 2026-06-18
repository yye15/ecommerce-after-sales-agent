"""Batch operation analysis agent."""

from __future__ import annotations

from collections import Counter
from typing import Any


def build_operation_report(results: list[dict[str, Any]]) -> dict[str, Any]:
    risk_counter: Counter[str] = Counter()
    issue_counter: Counter[str] = Counter()
    aspect_counter: Counter[str] = Counter()
    negative_examples = []

    for result in results:
        final = result.get("final_result", result)
        risk = final.get("risk", {})
        sentiment = final.get("sentiment", {})
        categories = final.get("issue_categories", [])

        risk_counter[risk.get("level", "未知")] += 1
        for category in categories:
            issue_counter[category.get("category", "其他反馈")] += 1
        for triplet in sentiment.get("triplets", []):
            if triplet.get("sentiment") == "NEG":
                aspect_counter[triplet.get("aspect", "整体体验")] += 1
                if len(negative_examples) < 5:
                    negative_examples.append(triplet)

    top_issues = issue_counter.most_common(5)
    top_aspects = aspect_counter.most_common(5)
    recommendations = []
    if top_issues:
        recommendations.append(f"优先处理高频问题：{top_issues[0][0]}。")
    if risk_counter.get("高风险", 0):
        recommendations.append("建立高风险投诉快速升级机制，避免差评扩散。")
    if top_aspects:
        recommendations.append(f"重点复盘负面方面：{top_aspects[0][0]}。")
    if not recommendations:
        recommendations.append("当前样本风险较低，可持续监控。")

    return {
        "total_cases": len(results),
        "risk_distribution": dict(risk_counter),
        "top_issue_categories": top_issues,
        "top_negative_aspects": top_aspects,
        "negative_examples": negative_examples,
        "recommendations": recommendations,
    }
