"""Review and repair the 100-row ASTE audit sample.

The goal is not to make the labels verbose. The goal is to make them usable for
LALUN span training, which means both aspect and opinion should appear in the
original review text.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "labeled" / "deepseek_aste_audit_100.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "labeled" / "deepseek_aste_audit_100_reviewed.csv"

PRODUCT_CANDIDATES = [
    "水果",
    "苹果",
    "火龙果",
    "果子",
    "衣服",
    "裤子",
    "酒店",
    "房间",
    "书",
    "宝贝",
    "东西",
    "机子",
    "手机",
    "电脑",
    "平板",
    "这款",
    "耳机",
    "瓶子",
    "潘婷",
    "清扬",
    "蒙牛",
    "联想",
]

ASPECT_REPLACEMENTS = {
    "商品": PRODUCT_CANDIDATES,
    "产品": PRODUCT_CANDIDATES,
    "购物": ["买", "购物", "京东"],
    "像素": ["相素", "像素"],
    "32g容量": ["32g"],
    "购物体验": ["方便", "京东"],
    "重量": ["重"],
    "外观": ["看上去", "实物"],
    "退订": ["退"],
    "物流": ["物流", "快递", "运输公司", "发货"],
    "价格": ["价格", "价", "钱"],
    "厚度": ["薄"],
    "店铺": ["光顾", "京东"],
    "新鲜度": ["新鲜"],
    "快递小哥": ["小哥", "快递"],
    "性价比": ["性价比", "价"],
    "气味": ["汗味", "味道"],
    "周围卫生": ["周围"],
    "位置便利性": ["指示标志", "找"],
    "扩展": ["存储卡"],
    "颜色": ["颜色", "青"],
    "口感": ["口感", "甜", "味道"],
    "尺寸": ["尺寸", "小"],
    "卖家信誉": ["卖家", "信誉"],
    "商品质量": PRODUCT_CANDIDATES + ["质量"],
}

OPINION_REPLACEMENTS = {
    "挺好": ["挺好的", "挺好"],
    "外面买的更香": ["更香"],
    "字体很小": ["很小"],
    "估计是组装机": ["组装机"],
    "太失望": ["太太太失望", "失望"],
    "还可从": ["还可从"],
    "烂": ["够烂", "烂"],
    "卡": ["好卡", "卡"],
}

MANUAL_REVIEWED = {
    "R003283": [
        {"aspect": "推荐", "opinion": "很好", "sentiment": "POS", "evidence": "推荐的很好"},
        {"aspect": "宝宝", "opinion": "不喜欢", "sentiment": "NEG", "evidence": "宝宝好像不喜欢"},
    ],
    "R013_REMOVED": [],
    "R037030": [],
    "R037825": [],
    "R027173": [],
    "R052301": [
        {"aspect": "屏幕", "opinion": "很小", "sentiment": "NEG", "evidence": "网页字体又显得很小"},
        {"aspect": "眼睛", "opinion": "累", "sentiment": "NEG", "evidence": "眼睛累"},
        {"aspect": "重", "opinion": "有点重", "sentiment": "NEG", "evidence": "确实有点重"},
    ],
    "R060256": [
        {"aspect": "室内", "opinion": "差太多", "sentiment": "NEG", "evidence": "室内、用具、环境及性价比还是差太多"},
        {"aspect": "用具", "opinion": "差太多", "sentiment": "NEG", "evidence": "室内、用具、环境及性价比还是差太多"},
        {"aspect": "环境", "opinion": "差太多", "sentiment": "NEG", "evidence": "室内、用具、环境及性价比还是差太多"},
        {"aspect": "性价比", "opinion": "差太多", "sentiment": "NEG", "evidence": "性价比还是差太多"},
        {"aspect": "汗味", "opinion": "受不了", "sentiment": "NEG", "evidence": "有股汗味让人实在受不了"},
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review ASTE audit sample.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def clean_text(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def parse_triplets(text: str) -> list[dict[str, Any]]:
    try:
        data = json.loads(text or "[]")
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def normalize_sentiment(value: Any) -> str:
    sentiment = str(value or "NEU").upper()
    return sentiment if sentiment in {"POS", "NEG", "NEU"} else "NEU"


def first_present(review: str, candidates: list[str]) -> str | None:
    for item in candidates:
        if item and item in review:
            return item
    return None


def nearest_present(review: str, candidates: list[str], anchor: str) -> str | None:
    present = [item for item in candidates if item and item in review]
    if not present:
        return None
    anchor_index = review.find(anchor) if anchor in review else len(review) // 2
    return min(present, key=lambda item: abs(review.find(item) - anchor_index))


def repair_aspect(review: str, aspect: str, opinion: str, category: str) -> str | None:
    if aspect in review:
        return aspect
    candidates = ASPECT_REPLACEMENTS.get(aspect, [])
    repaired = nearest_present(review, candidates, opinion) if candidates else None
    if repaired:
        return repaired
    if category and category in review:
        return category
    return nearest_present(review, PRODUCT_CANDIDATES, opinion)


def repair_opinion(review: str, opinion: str, evidence: str) -> str | None:
    if opinion in review:
        return opinion
    for candidate in OPINION_REPLACEMENTS.get(opinion, []):
        if candidate in review:
            return candidate
    simplified = opinion.strip("的一了啊！!，,。.")
    if simplified and simplified in review:
        return simplified
    if evidence:
        evidence = clean_text(evidence)
        if evidence and evidence in review and len(evidence) <= 12:
            return evidence
    return None


def review_triplets(row: dict[str, str]) -> tuple[list[dict[str, str]], list[str]]:
    review = clean_text(row.get("review", ""))
    review_id = row.get("review_id", "")
    if review_id in MANUAL_REVIEWED:
        triplets = MANUAL_REVIEWED[review_id]
    else:
        triplets = parse_triplets(row.get("triplets_json", "[]"))

    repaired: list[dict[str, str]] = []
    notes = []
    seen = set()
    for item in triplets:
        aspect_raw = str(item.get("aspect", "")).strip()
        opinion_raw = str(item.get("opinion", "")).strip()
        evidence = str(item.get("evidence", "")).strip()
        sentiment = normalize_sentiment(item.get("sentiment"))

        aspect = repair_aspect(review, aspect_raw, opinion_raw, row.get("category", ""))
        opinion = repair_opinion(review, opinion_raw, evidence)
        if not aspect or not opinion:
            notes.append(f"drop:{aspect_raw}/{opinion_raw}")
            continue
        if aspect not in review or opinion not in review:
            notes.append(f"drop_span:{aspect}/{opinion}")
            continue
        key = (aspect, opinion, sentiment)
        if key in seen:
            continue
        seen.add(key)
        if aspect != aspect_raw or opinion != opinion_raw:
            notes.append(f"repair:{aspect_raw}/{opinion_raw}->{aspect}/{opinion}")
        repaired.append(
            {
                "aspect": aspect,
                "opinion": opinion,
                "sentiment": sentiment,
                "evidence": evidence or review[:80],
            }
        )
    return repaired, notes


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    with input_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    reviewed = 0
    rejected = 0
    total_triplets = 0
    for row in rows:
        triplets, notes = review_triplets(row)
        if triplets:
            row["review_status"] = "reviewed"
            row["reviewed_triplets_json"] = json.dumps(triplets, ensure_ascii=False)
            reviewed += 1
            total_triplets += len(triplets)
        else:
            row["review_status"] = "rejected"
            row["reviewed_triplets_json"] = "[]"
            rejected += 1
        row["human_notes"] = "; ".join(notes)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(
        json.dumps(
            {
                "input_rows": len(rows),
                "reviewed_rows": reviewed,
                "rejected_rows": rejected,
                "reviewed_triplets": total_triplets,
                "output": str(output_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
