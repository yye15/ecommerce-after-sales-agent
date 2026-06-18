"""Issue classification agent.

The classifier keeps the original coarse `category` field for compatibility,
and adds business-friendly secondary reasons for explanation and UI display.
"""

from __future__ import annotations

from typing import Any


ISSUE_TAXONOMY: dict[str, dict[str, Any]] = {
    "物流问题": {
        "business_dimension": "履约",
        "description": "发货、运输、配送、签收等履约问题。",
        "secondary_rules": {
            "发货延迟": ["发货", "没发货", "未发货", "没有现货", "两个星期", "一直不发"],
            "配送异常": ["物流", "快递", "配送", "派送", "运输", "没送到", "到货", "空单号"],
            "履约沟通受阻": ["打不通", "30公里", "自提", "没人联系"],
        },
    },
    "客服响应问题": {
        "business_dimension": "服务",
        "description": "客服、售后、人工服务的响应和态度问题。",
        "secondary_rules": {
            "响应慢或失联": ["客服", "售后", "回复", "没人理", "没人回答", "找不到人", "联系不到", "不理", "前台"],
            "处理拖延": ["处理", "一直处理", "迟迟", "拖", "应付", "敷衍", "入住手续"],
            "服务态度差": ["态度", "服务态度", "骂", "凶", "不耐烦"],
        },
    },
    "商品质量问题": {
        "business_dimension": "商品",
        "description": "商品破损、故障、耐用性、真假、新鲜度等质量问题。",
        "secondary_rules": {
            "破损/瑕疵": ["破", "裂开", "开裂", "瑕疵", "黑点", "脏兮兮", "包装破"],
            "核心故障": ["坏", "死悄悄", "断线", "漏水", "漏", "泄露", "泄漏", "听筒", "花屏"],
            "耐用性不足": ["质量", "耐用", "不耐用", "才用", "不到一年", "用了几天", "穿不了"],
            "材质/做工问题": ["做工", "材质", "布料", "掉色", "瓶子"],
            "真假/描述可信度": ["假货", "*货", "贴牌", "品牌不一样"],
            "食品新鲜度": ["不新鲜", "长毛", "变质", "过期", "不够脆", "不甜", "糠", "康", "不能吃", "太熟", "食品安全", "把关水平"],
        },
    },
    "产品体验问题": {
        "business_dimension": "体验",
        "description": "不一定是客观质量缺陷，但影响使用体验、功能感知或适配。",
        "secondary_rules": {
            "核心功能体验": ["音质", "续航", "屏幕", "效果", "机器", "运行", "开机", "硬件", "软件"],
            "性能/稳定性": ["很卡", "卡", "充电不了", "充不了电", "没有反映", "信号", "散热"],
            "适配/安装": ["尺码", "安装", "插槽", "发件箱", "闹钟", "设施", "房间设施"],
            "感官体验": ["味道", "烟味", "油漆味", "吵", "不透气", "刺激眼睛", "头皮屑", "痛苦", "舒适感"],
            "内容/图片不符": ["图片", "发错", "搞错", "像素", "温度", "什么都差", "有待提高", "达不到", "不配", "小了", "薄", "松懈"],
        },
    },
    "价格与权益问题": {
        "business_dimension": "权益",
        "description": "价格、优惠、发票、赠品、宣传承诺等权益问题。",
        "secondary_rules": {
            "价格争议": ["价格", "价钱", "贵", "便宜", "差价", "被坑", "性价比"],
            "优惠/赠品争议": ["优惠", "券", "赠品", "没配", "没有了"],
            "票据/运费": ["发票", "运费", "不退运费"],
            "描述不符/误导": ["被骗", "骗人", "欺骗消费者", "欺骗", "贴牌", "品牌不一样", "图片保持一致"],
        },
    },
    "退换货与赔付问题": {
        "business_dimension": "售后",
        "description": "退款、退货、换货、保修、补偿等售后处理诉求。",
        "secondary_rules": {
            "退款诉求": ["退款", "退钱"],
            "退货/换货诉求": ["退货", "换货", "退换", "退换货", "不能退", "不能换", "无理由退换货"],
            "补偿/赔付诉求": ["赔偿", "补偿", "赔付"],
            "保修/维修": ["保修", "维修", "检测"],
        },
    },
}

CATEGORY_RULES = {
    category: sorted({word for words in meta["secondary_rules"].values() for word in words}, key=len, reverse=True)
    for category, meta in ISSUE_TAXONOMY.items()
}

NEGATIVE_CONTEXT_WORDS = [
    "差",
    "坏",
    "不好",
    "不行",
    "不符",
    "不建议",
    "不能",
    "没有",
    "没",
    "无语",
    "麻烦",
    "费劲",
    "上当",
    "被骗",
    "被坑",
    "骗人",
    "欺骗",
    "投诉",
    "退货",
    "退款",
    "失望",
    "后悔",
    "垃圾",
    "烂",
    "差劲",
    "不能吃",
    "失败",
    "不值得",
    "脏",
    "长毛",
    "变质",
    "过期",
    "死",
    "抵制",
    "痛苦",
    "吵",
    "烟味",
    "油漆味",
    "有待提高",
    "达不到",
    "不配",
    "穿不了",
    "漏",
    "搞错",
    "舒适感极差",
    "把关水平",
    "什么都差",
    "怎么解决",
    "什么鬼",
    "奇葩",
]


def classify_issues(text: str, sentiment: dict[str, Any]) -> list[dict[str, Any]]:
    if _is_irrelevant_rant(text):
        return [_fallback_category(text, confidence=0.72)]

    triplets = sentiment.get("triplets", [])
    categories: dict[str, dict[str, Any]] = {}

    for item in triplets:
        merged_text = f"{item.get('aspect', '')} {item.get('opinion', '')} {item.get('evidence', '')}"
        item_is_negative = item.get("sentiment") == "NEG"
        if not item_is_negative:
            continue
        _collect_matches(
            categories,
            source_text=merged_text,
            evidence=item.get("evidence") or item.get("aspect") or merged_text,
            confidence_base=0.72,
            confidence_step=0.05,
            negative_count=1,
        )

    text_is_negative = any(word in text for word in NEGATIVE_CONTEXT_WORDS)
    if text_is_negative:
        _collect_matches(
            categories,
            source_text=text,
            evidence=text[:120],
            confidence_base=0.68,
            confidence_step=0.04,
            negative_count=1,
        )

    if not categories:
        categories["其他反馈"] = _fallback_category(text)

    return [_finalize_category(item) for item in categories.values()]


def _collect_matches(
    categories: dict[str, dict[str, Any]],
    *,
    source_text: str,
    evidence: str,
    confidence_base: float,
    confidence_step: float,
    negative_count: int,
) -> None:
    for category, keywords in CATEGORY_RULES.items():
        matched = _filter_positive_negated_hits(source_text, [word for word in keywords if word in source_text])
        if not matched:
            continue
        current = categories.setdefault(category, _new_category(category, confidence_base))
        current["confidence"] = min(0.98, current["confidence"] + confidence_step * len(matched))
        current["evidence"].append(evidence)
        current["negative_count"] += negative_count
        current["matched_keywords"].update(matched)
        current["secondary_reasons"].update(_secondary_reasons_for(category, matched))


def _new_category(category: str, confidence: float) -> dict[str, Any]:
    meta = ISSUE_TAXONOMY[category]
    return {
        "category": category,
        "primary_category": category,
        "business_dimension": meta["business_dimension"],
        "description": meta["description"],
        "secondary_reasons": set(),
        "confidence": confidence,
        "evidence": [],
        "negative_count": 0,
        "matched_keywords": set(),
    }


def _finalize_category(item: dict[str, Any]) -> dict[str, Any]:
    finalized = dict(item)
    finalized["secondary_reasons"] = sorted(item.get("secondary_reasons", [])) or ["未细分"]
    finalized["matched_keywords"] = sorted(item.get("matched_keywords", []), key=len, reverse=True)
    finalized["evidence"] = _dedupe_keep_order(item.get("evidence", []))
    return finalized


def _fallback_category(text: str, confidence: float = 0.5) -> dict[str, Any]:
    return {
        "category": "其他反馈",
        "primary_category": "其他反馈",
        "business_dimension": "其他",
        "description": "未命中明确售后问题类别。",
        "secondary_reasons": ["未细分"],
        "confidence": confidence,
        "evidence": [text[:80]],
        "negative_count": 0,
        "matched_keywords": [],
    }


def _secondary_reasons_for(category: str, matched_keywords: list[str]) -> set[str]:
    secondary_rules = ISSUE_TAXONOMY[category]["secondary_rules"]
    reasons = set()
    for reason, keywords in secondary_rules.items():
        if any(keyword in matched_keywords for keyword in keywords):
            reasons.add(reason)
    return reasons


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _is_irrelevant_rant(text: str) -> bool:
    has_noncommerce_topic = any(word in text for word in ["驾驶证", "超速", "扣6分", "罚400"])
    has_commerce_topic = any(word in text for word in ["买", "收到", "客服", "物流", "退款", "退货", "换货", "发货"])
    return has_noncommerce_topic and not has_commerce_topic


def _filter_positive_negated_hits(text: str, hits: list[str]) -> list[str]:
    filtered = []
    for hit in hits:
        if hit == "坏" and any(phrase in text for phrase in ["没有坏", "没坏", "无坏"]):
            continue
        if hit in {"漏", "漏水"} and any(phrase in text for phrase in ["没漏", "没有漏", "一点都不漏"]):
            continue
        if hit == "破" and any(phrase in text for phrase in ["没破", "没有破"]):
            continue
        if hit in {"快递", "到货"} and any(phrase in text for phrase in ["快递员辛苦", "送货速度快", "顺丰很给力"]):
            continue
        if hit in {"便宜", "贵", "价格"} and any(phrase in text for phrase in ["太实惠", "非常满意", "包装非常好", "比超市便宜"]):
            continue
        filtered.append(hit)
    return filtered
