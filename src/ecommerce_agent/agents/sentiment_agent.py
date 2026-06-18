"""Fine-grained sentiment extraction agent."""

from __future__ import annotations

import re
from typing import Any

from ..config import get_settings
from ..lalun_adapter import LALUNAdapter
from ..llm_client import LLMClient
from ..prompts import SENTIMENT_SYSTEM_PROMPT


ASPECT_KEYWORDS = {
    "物流": ["物流", "快递", "配送", "发货", "到货", "派送", "运输"],
    "客服": ["客服", "售后", "回复", "处理", "态度", "人工"],
    "商品质量": ["质量", "做工", "材质", "瑕疵", "坏", "破", "掉色", "耐用", "漏水", "假货", "不新鲜"],
    "包装": ["包装", "盒子", "外箱", "封口"],
    "价格": ["价格", "价钱", "优惠", "贵", "便宜", "券", "差价", "发票"],
    "退换货": ["退款", "退货", "换货", "退换", "赔偿", "补偿"],
    "产品体验": ["音质", "续航", "屏幕", "尺码", "味道", "效果", "安装", "图片", "机器", "运行", "开机"],
}

NEG_WORDS = [
    "不回复",
    "没人理",
    "太慢",
    "漏发",
    "少发",
    "差",
    "坏",
    "慢",
    "破",
    "不行",
    "不好",
    "不耐用",
    "裂开",
    "漏水",
    "不新鲜",
    "不给开",
    "看不到",
    "很卡",
    "太慢",
    "被坑",
    "不给",
    "失望",
    "投诉",
    "退款",
    "退货",
    "假",
    "贵",
    "无法",
    "不能",
]

POS_WORDS = [
    "好",
    "很好",
    "不错",
    "满意",
    "喜欢",
    "快",
    "便宜",
    "清晰",
    "舒服",
    "推荐",
]

STRONG_NEG_WORDS = [
    "不耐用",
    "被坑",
    "不给",
    "不给开",
    "看不到",
    "裂开",
    "漏水",
    "不新鲜",
    "假",
    "假货",
]


def analyze_sentiment(text: str, llm: LLMClient | None = None) -> dict[str, Any]:
    """Extract aspect-opinion-sentiment triplets.

    English text can use the local LALUN research model. Chinese text keeps the
    existing DeepSeek/rule path.
    """
    settings = get_settings()
    if settings.use_lalun_english and _looks_like_english(text):
        lalun_result = _try_lalun_english(text)
        if lalun_result:
            return lalun_result

    llm = llm or LLMClient()
    user_prompt = f"待分析文本：{text}"
    llm_result = llm.invoke_json(SENTIMENT_SYSTEM_PROMPT, user_prompt)
    if isinstance(llm_result, dict) and llm_result.get("triplets"):
        return _normalize_llm_result(llm_result, text)
    return rule_based_sentiment(text)


def _looks_like_english(text: str) -> bool:
    if re.search(r"[\u4e00-\u9fff]", text):
        return False
    letters = re.findall(r"[A-Za-z]", text)
    return len(letters) >= 6


def _try_lalun_english(text: str) -> dict[str, Any] | None:
    try:
        result = LALUNAdapter().analyze_english(text)
    except Exception:
        return None
    triplets = result.get("triplets") if isinstance(result, dict) else None
    if not triplets:
        return None

    normalized = []
    for item in triplets:
        sentiment = str(item.get("sentiment", "NEU")).upper()
        if sentiment not in {"POS", "NEG", "NEU"}:
            sentiment = "NEU"
        normalized.append(
            {
                "aspect": str(item.get("aspect", "overall")),
                "opinion": str(item.get("opinion", "mentioned")),
                "sentiment": sentiment,
                "evidence": str(item.get("evidence", text[:80])),
            }
        )

    return {
        "overall_sentiment": _overall_from_triplets(normalized),
        "summary": _summary_from_lalun_triplets(normalized),
        "triplets": normalized,
        "engine": "lalun",
        "lalun_dataset": result.get("dataset", "14res"),
    }


def rule_based_sentiment(text: str) -> dict[str, Any]:
    triplets: list[dict[str, str]] = []
    lowered = text.lower()

    for aspect, keywords in ASPECT_KEYWORDS.items():
        matched_keyword = next((kw for kw in keywords if kw in text), None)
        if not matched_keyword:
            continue

        sentiment = "NEU"
        opinion = "提及"

        neg_near = _find_nearest_opinion(text, matched_keyword, NEG_WORDS)
        pos_near = _find_nearest_opinion(text, matched_keyword, POS_WORDS)
        neg = neg_near[0] if neg_near else _find_first(text, NEG_WORDS)
        pos = pos_near[0] if pos_near else _find_first(text, POS_WORDS)

        if neg_near and pos_near:
            if neg_near[0] in STRONG_NEG_WORDS or neg_near[1] <= pos_near[1]:
                sentiment = "NEG"
                opinion = neg_near[0]
            else:
                sentiment = "POS"
                opinion = pos_near[0]
        elif neg_near:
            sentiment = "NEG"
            opinion = neg_near[0]
        elif pos_near:
            sentiment = "POS"
            opinion = pos_near[0]
        elif neg and not pos:
            sentiment = "NEG"
            opinion = neg
        elif pos and not neg:
            sentiment = "POS"
            opinion = pos

        triplets.append(
            {
                "aspect": matched_keyword if aspect == "产品体验" else aspect,
                "opinion": opinion,
                "sentiment": sentiment,
                "evidence": _evidence_window(text, matched_keyword),
            }
        )

    if not triplets:
        sentiment = "NEG" if _find_first(text, NEG_WORDS) else "POS" if _find_first(text, POS_WORDS) else "NEU"
        triplets.append(
            {
                "aspect": "整体体验",
                "opinion": "整体评价",
                "sentiment": sentiment,
                "evidence": text[:80],
            }
        )

    sentiments = {item["sentiment"] for item in triplets}
    non_neutral = sentiments - {"NEU"}
    if len(non_neutral) > 1:
        overall = "MIXED"
    elif non_neutral:
        overall = next(iter(non_neutral))
    else:
        overall = "NEU"
    return {
        "overall_sentiment": overall,
        "summary": _summary_from_triplets(triplets),
        "triplets": triplets,
        "engine": "rule_fallback",
    }


def _normalize_llm_result(result: dict[str, Any], text: str) -> dict[str, Any]:
    normalized = []
    for item in result.get("triplets", []):
        sentiment = str(item.get("sentiment", "NEU")).upper()
        if sentiment not in {"POS", "NEG", "NEU"}:
            sentiment = "NEU"
        normalized.append(
            {
                "aspect": str(item.get("aspect", "整体体验")),
                "opinion": str(item.get("opinion", "提及")),
                "sentiment": sentiment,
                "evidence": str(item.get("evidence", text[:80])),
            }
        )
    result["triplets"] = normalized
    result["engine"] = "deepseek"
    if not result.get("summary"):
        result["summary"] = _summary_from_triplets(normalized)
    return result


def _find_first(text: str, words: list[str]) -> str | None:
    for word in words:
        if word in text:
            return word
    return None


def _near(text: str, aspect: str, opinion: str, distance: int = 16) -> bool:
    aspect_index = text.find(aspect)
    opinion_index = text.find(opinion)
    if aspect_index == -1 or opinion_index == -1:
        return False
    return abs(aspect_index - opinion_index) <= distance


def _find_nearest_opinion(
    text: str,
    aspect: str,
    words: list[str],
    max_distance: int = 18,
) -> tuple[str, int] | None:
    aspect_index = text.find(aspect)
    if aspect_index == -1:
        return None

    best: tuple[str, int] | None = None
    for word in words:
        start = 0
        while True:
            opinion_index = text.find(word, start)
            if opinion_index == -1:
                break
            distance = abs(opinion_index - aspect_index)
            if opinion_index < aspect_index:
                distance += 8
            if distance <= max_distance and (best is None or distance < best[1]):
                best = (word, distance)
            start = opinion_index + len(word)
    return best


def _evidence_window(text: str, keyword: str, width: int = 18) -> str:
    index = text.find(keyword)
    if index == -1:
        return text[:80]
    start = max(0, index - width)
    end = min(len(text), index + len(keyword) + width)
    return text[start:end]


def _summary_from_triplets(triplets: list[dict[str, str]]) -> str:
    negative = [item["aspect"] for item in triplets if item["sentiment"] == "NEG"]
    positive = [item["aspect"] for item in triplets if item["sentiment"] == "POS"]
    if negative and positive:
        return f"客户对{', '.join(positive)}较满意，但对{', '.join(negative)}存在不满。"
    if negative:
        return f"客户主要不满集中在{', '.join(negative)}。"
    if positive:
        return f"客户主要正向反馈集中在{', '.join(positive)}。"
    return "客户表达了中性或信息不足的反馈。"


def _overall_from_triplets(triplets: list[dict[str, str]]) -> str:
    sentiments = {item["sentiment"] for item in triplets}
    non_neutral = sentiments - {"NEU"}
    if len(non_neutral) > 1:
        return "MIXED"
    if non_neutral:
        return next(iter(non_neutral))
    return "NEU"


def _summary_from_lalun_triplets(triplets: list[dict[str, str]]) -> str:
    negative = [item["aspect"] for item in triplets if item["sentiment"] == "NEG"]
    positive = [item["aspect"] for item in triplets if item["sentiment"] == "POS"]
    if negative and positive:
        return f"LALUN found positive aspects ({', '.join(positive)}) and negative aspects ({', '.join(negative)})."
    if negative:
        return f"LALUN found negative aspects: {', '.join(negative)}."
    if positive:
        return f"LALUN found positive aspects: {', '.join(positive)}."
    return "LALUN found neutral or insufficient sentiment evidence."
