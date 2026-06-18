"""Create reviewable multi-turn customer-service dialogue candidates.

The source problem comes from real Chinese e-commerce reviews. The dialogue is
synthetically expanded for evaluation, so every row still requires human review
before it becomes part of the golden set.
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ecommerce_agent.graph import run_case  # noqa: E402


DEFAULT_INPUT = PROJECT_ROOT / "data" / "raw" / "online_shopping_10_cats" / "online_shopping_10_cats.csv"
DEFAULT_BASE_EVAL = PROJECT_ROOT / "data" / "eval" / "customer_service_eval_50.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "eval" / "dialogue_eval_candidates_50.csv"

AFTER_SALES_KEYWORDS = [
    "退款",
    "退货",
    "换货",
    "售后",
    "客服",
    "投诉",
    "赔偿",
    "补偿",
    "物流",
    "快递",
    "发货",
    "没收到",
    "没送到",
    "坏",
    "质量",
    "不能用",
    "不回复",
    "没人",
    "维修",
    "保修",
    "发票",
    "差评",
    "屏幕",
    "漏水",
    "假货",
    "不耐用",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create synthetic dialogue eval candidates from real reviews.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="online_shopping_10_cats CSV path.")
    parser.add_argument("--base-eval", default=str(DEFAULT_BASE_EVAL), help="Existing eval set to avoid duplicates.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Candidate CSV output path.")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--use-llm", action="store_true", help="Use DeepSeek suggestions instead of rule fallback.")
    return parser.parse_args()


def normalize_text(text: str) -> str:
    text = (text or "").replace("\ufeff", "").replace("\u3000", " ")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def read_used_reviews(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return {normalize_text(row.get("review_text", "")) for row in csv.DictReader(f)}


def is_meaningful_review(text: str) -> bool:
    if not (12 <= len(text) <= 220):
        return False
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
    if len(chinese_chars) < 8:
        return False
    if len(set(text)) <= 5:
        return False
    return True


def after_sales_score(text: str, label: str) -> int:
    score = 0
    score += sum(1 for keyword in AFTER_SALES_KEYWORDS if keyword in text) * 3
    if label == "0":
        score += 4
    if any(word in text for word in ["投诉", "退款", "退货", "换货", "客服", "售后", "没收到", "坏", "质量"]):
        score += 8
    if any(word in text for word in ["不错", "满意", "喜欢", "很好"]) and label == "1":
        score -= 2
    return score


def load_source_reviews(path: Path, used_reviews: set[str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = []
        for raw in csv.DictReader(f):
            text = normalize_text(raw.get("review", ""))
            if not is_meaningful_review(text) or text in used_reviews:
                continue
            label = str(raw.get("label", "")).strip()
            if label not in {"0", "1"}:
                continue
            rows.append(
                {
                    "review_text": text,
                    "product_category": normalize_text(raw.get("cat", "")),
                    "source_sentiment_label": label,
                    "rating_bucket": "negative" if label == "0" else "positive",
                    "score": str(after_sales_score(text, label)),
                }
            )
    return rows


def build_dialogue(review: str, category: str, label: str) -> str:
    if label == "1":
        return "\n".join(
            [
                f"用户：我买的{category or '商品'}整体体验还不错，想补充一下使用感受。",
                "坐席：您好，可以的，您具体想咨询哪方面呢？",
                f"用户：{review}",
            ]
        )

    if any(word in review for word in ["退款", "退货", "换货", "赔偿", "补偿"]):
        opening = f"用户：我买的{category or '商品'}现在想处理退换货或赔付问题。"
    elif any(word in review for word in ["物流", "快递", "发货", "没收到", "没送到"]):
        opening = f"用户：我这个{category or '商品'}订单的物流有问题，想找人处理。"
    elif any(word in review for word in ["客服", "售后", "没人", "不回复"]):
        opening = "用户：我之前联系过客服，但是问题一直没有解决。"
    elif any(word in review for word in ["坏", "质量", "屏幕", "漏水", "假货", "不耐用"]):
        opening = f"用户：我买的{category or '商品'}出现了质量或使用问题。"
    else:
        opening = f"用户：我买的{category or '商品'}体验不太好，想反馈一下。"

    return "\n".join(
        [
            opening,
            "坐席：您好，非常抱歉给您带来困扰，请您具体描述一下问题。",
            f"用户：{review}",
        ]
    )


def category_names(final: dict[str, Any]) -> str:
    return ";".join(
        item.get("category", "")
        for item in final.get("issue_categories", [])
        if item.get("category")
    )


def risk_reasons(final: dict[str, Any]) -> str:
    return " | ".join(final.get("risk", {}).get("reasons", []))


def build_candidate(row: dict[str, str], index: int, use_llm: bool) -> dict[str, Any]:
    dialogue = build_dialogue(
        row["review_text"],
        row.get("product_category", ""),
        row.get("source_sentiment_label", ""),
    )
    final = run_case(dialogue, use_llm=use_llm).get("final_result", {})
    return {
        "candidate_id": f"D{index:03d}",
        "dialogue_text": dialogue,
        "source_review_text": row["review_text"],
        "product_category": row.get("product_category", ""),
        "source_sentiment_label": row.get("source_sentiment_label", ""),
        "source_rating_bucket": row.get("rating_bucket", ""),
        "suggested_categories": category_names(final),
        "suggested_risk_level": final.get("risk", {}).get("level", ""),
        "suggested_priority": final.get("risk", {}).get("priority") or final.get("strategy", {}).get("priority", ""),
        "suggested_rule_reason": risk_reasons(final),
        "human_expected_categories": "",
        "human_expected_risk_level": "",
        "human_expected_priority": "",
        "human_rule_reason": "",
        "review_status": "needs_review",
    }


def select_rows(rows: list[dict[str, str]], limit: int, seed: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    high_score = [row for row in rows if int(row["score"]) >= 10]
    medium_score = [row for row in rows if 4 <= int(row["score"]) < 10]
    low_score = [row for row in rows if int(row["score"]) < 4]
    for bucket in (high_score, medium_score, low_score):
        rng.shuffle(bucket)

    selected = []
    quotas = [
        (high_score, int(limit * 0.5)),
        (medium_score, int(limit * 0.35)),
        (low_score, limit),
    ]
    seen = set()
    for bucket, quota in quotas:
        for row in bucket:
            if len(selected) >= limit or len([item for item in selected if item in bucket]) >= quota:
                break
            text = row["review_text"]
            if text not in seen:
                selected.append(row)
                seen.add(text)
    for row in high_score + medium_score + low_score:
        if len(selected) >= limit:
            break
        text = row["review_text"]
        if text not in seen:
            selected.append(row)
            seen.add(text)
    return selected[:limit]


def write_candidates(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "candidate_id",
        "dialogue_text",
        "source_review_text",
        "product_category",
        "source_sentiment_label",
        "source_rating_bucket",
        "suggested_categories",
        "suggested_risk_level",
        "suggested_priority",
        "suggested_rule_reason",
        "human_expected_categories",
        "human_expected_risk_level",
        "human_expected_priority",
        "human_rule_reason",
        "review_status",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    used_reviews = read_used_reviews(Path(args.base_eval))
    source_rows = load_source_reviews(input_path, used_reviews)
    selected = select_rows(source_rows, args.limit, args.seed)
    candidates = [
        build_candidate(row, index + 1, use_llm=args.use_llm)
        for index, row in enumerate(selected)
    ]
    write_candidates(candidates, output_path)

    print(f"input_file={input_path}")
    print(f"output_file={output_path}")
    print(f"available_source_rows={len(source_rows)}")
    print(f"candidate_count={len(candidates)}")
    print("status=needs_human_review")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
