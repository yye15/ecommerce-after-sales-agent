"""Lightweight local RAG retrieval for customer service policies.

This first version deliberately avoids an external vector database. It uses a
small local knowledge base and transparent keyword scoring, which is easier to
run, debug, and explain. The module can later be replaced by Chroma or a
hybrid retriever without changing the graph contract.
"""

from __future__ import annotations

from typing import Any

from .knowledge_base import KnowledgeDocument, load_knowledge_base


CATEGORY_QUERY_TERMS = {
    "商品质量问题": ["质量", "坏", "裂开", "漏水", "不能用", "耐用", "故障", "退换货"],
    "产品体验问题": ["体验", "效果", "屏幕", "尺寸", "气味", "噪音", "使用"],
    "物流问题": ["物流", "快递", "配送", "发货", "送达", "空单号"],
    "客服响应问题": ["客服", "售后", "回复", "没人处理", "升级", "主管"],
    "价格与权益问题": ["价格", "优惠", "差价", "赠品", "发票", "运费", "权益"],
    "退换货与赔付问题": ["退款", "退货", "换货", "赔偿", "补偿", "保修", "售后"],
}

RISK_QUERY_TERMS = {
    "高风险": ["高风险", "投诉", "维权", "举报", "升级", "主管", "优先"],
    "中风险": ["核实", "售后", "质量", "物流", "补偿"],
    "低风险": ["温和", "体验反馈", "记录", "优化"],
}


def retrieve_policy_context(
    text: str,
    categories: list[dict[str, Any]],
    risk: dict[str, Any],
    *,
    top_k: int = 4,
) -> dict[str, Any]:
    query_terms = _build_query_terms(text, categories, risk)
    docs = load_knowledge_base()
    scored = sorted(
        (
            (_score_document(doc, text, query_terms), doc)
            for doc in docs
        ),
        key=lambda item: item[0],
        reverse=True,
    )
    selected = [
        _doc_to_context(doc, score)
        for score, doc in scored[:top_k]
        if score > 0
    ]
    return {
        "query_terms": sorted(query_terms),
        "documents": selected,
        "context_text": _format_context(selected),
        "retriever": "local_keyword_policy_rag_v1",
    }


def _build_query_terms(text: str, categories: list[dict[str, Any]], risk: dict[str, Any]) -> set[str]:
    terms = {word for word in _tokenize_text(text) if len(word) >= 2}
    for item in categories:
        category = item.get("category", "")
        terms.add(category)
        terms.update(item.get("secondary_reasons", []))
        terms.update(item.get("matched_keywords", []))
        terms.update(CATEGORY_QUERY_TERMS.get(category, []))
    risk_level = risk.get("level", "")
    terms.add(risk_level)
    terms.update(RISK_QUERY_TERMS.get(risk_level, []))
    return {term for term in terms if term}


def _score_document(doc: KnowledgeDocument, text: str, query_terms: set[str]) -> int:
    haystack = f"{doc.title}\n{doc.content}"
    score = 0
    for term in query_terms:
        if term and term in haystack:
            score += 3
        if term and term in doc.title:
            score += 2
    for keyword in doc.keywords:
        if keyword and (keyword in text or keyword in query_terms):
            score += 4
    return score


def _doc_to_context(doc: KnowledgeDocument, score: int) -> dict[str, Any]:
    return {
        "doc_id": doc.doc_id,
        "title": doc.title,
        "source": doc.source,
        "score": score,
        "content": doc.content,
        "keywords": list(doc.keywords),
    }


def _format_context(docs: list[dict[str, Any]]) -> str:
    if not docs:
        return "未检索到明确售后政策。"
    blocks = []
    for idx, doc in enumerate(docs, start=1):
        blocks.append(
            f"[{idx}] {doc['title']} ({doc['source']})\n"
            f"{doc['content']}"
        )
    return "\n\n".join(blocks)


def _tokenize_text(text: str) -> list[str]:
    separators = "，。！？；：、,.!?;:\n\t "
    current = []
    tokens = []
    for char in text:
        if char in separators:
            if current:
                tokens.append("".join(current))
                current = []
        else:
            current.append(char)
    if current:
        tokens.append("".join(current))
    return tokens
