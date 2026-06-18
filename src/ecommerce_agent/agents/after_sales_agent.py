"""After-sales strategy agent."""

from __future__ import annotations

from typing import Any


def propose_strategy(
    text: str,
    sentiment: dict[str, Any],
    categories: list[dict[str, Any]],
    risk: dict[str, Any],
    escalation_note: str = "",
) -> dict[str, Any]:
    category_names = {item["category"] for item in categories}
    actions: list[dict[str, str]] = []

    if "物流问题" in category_names:
        actions.append(
            {
                "action": "查询物流",
                "owner": "售后客服",
                "detail": "核实物流节点、预计送达时间；如超时，主动同步补偿规则。",
            }
        )
    if "客服响应问题" in category_names:
        actions.append(
            {
                "action": "升级客服工单",
                "owner": "客服主管",
                "detail": "检查历史响应记录，承诺明确处理时限，避免继续无人跟进。",
            }
        )
    if "商品质量问题" in category_names or "产品体验问题" in category_names:
        actions.append(
            {
                "action": "收集凭证",
                "owner": "售后客服",
                "detail": "引导客户提供照片、视频或订单号，用于判断退换货或补发。",
            }
        )
    if "退换货与赔付问题" in category_names:
        actions.append(
            {
                "action": "核验售后政策",
                "owner": "售后专员",
                "detail": "核对订单状态、签收时间和商品问题，给出退款、换货或补偿路径。",
            }
        )
    if "价格与权益问题" in category_names:
        actions.append(
            {
                "action": "核对权益",
                "owner": "运营客服",
                "detail": "核实优惠券、差价保护、活动规则，避免模糊解释引发投诉。",
            }
        )

    if risk.get("should_escalate"):
        actions.insert(
            0,
            {
                "action": "人工优先介入",
                "owner": "客服主管",
                "detail": escalation_note or "高风险投诉应优先处理，并保留处理记录。",
            },
        )

    if not actions:
        actions.append(
            {
                "action": "标准安抚",
                "owner": "客服",
                "detail": "感谢客户反馈，继续追问订单号和具体问题。",
            }
        )

    return {
        "priority": "P1" if risk.get("level") == "高风险" else "P2" if risk.get("level") == "中风险" else "P3",
        "actions": actions,
        "business_goal": "降低投诉风险，提升售后响应效率，并沉淀可复用的问题归因。",
    }
