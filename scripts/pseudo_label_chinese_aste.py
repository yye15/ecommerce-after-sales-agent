"""Pseudo-label Chinese e-commerce reviews with DeepSeek for LALUN fine-tuning.

The output CSV is designed for human audit. After reviewing some rows, convert
it to LALUN JSON with scripts/convert_aste_labels_to_lalun.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ecommerce_agent.llm_client import LLMClient  # noqa: E402


DEFAULT_SOURCE = PROJECT_ROOT / "data" / "raw" / "online_shopping_10_cats" / "online_shopping_10_cats.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "labeled" / "deepseek_aste_300.csv"

SYSTEM_PROMPT = """你是中文电商评论的方面级情感三元组标注专家。
请为每条评论抽取 aspect/opinion/sentiment 三元组。

标注规则：
1. aspect 是被评价对象，例如 商品质量、物流、客服、包装、价格、屏幕、音质、口感、尺寸。
2. opinion 是原文中能体现评价的词或短语，例如 太慢、不耐用、不错、差、破损。
3. sentiment 只能是 POS、NEG、NEU。
4. aspect 和 opinion 尽量使用原文中出现的词，方便后续定位文本跨度。
5. 如果没有明确评价，triplets 可以为空数组。
6. 只输出 JSON 数组，不要输出解释。

输出格式：
[
  {
    "review_id": "R0001",
    "overall_sentiment": "POS|NEG|NEU|MIXED",
    "triplets": [
      {"aspect": "物流", "opinion": "太慢", "sentiment": "NEG", "evidence": "物流太慢"}
    ]
  }
]"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create DeepSeek pseudo ASTE labels.")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-chars", type=int, default=120)
    parser.add_argument("--negative-ratio", type=float, default=0.65)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def clean_text(text: str) -> str:
    return " ".join((text or "").replace("\ufeff", "").split()).strip()


def load_reviews(path: Path, limit: int, seed: int, max_chars: int, negative_ratio: float) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    seen: set[str] = set()
    positive: list[dict[str, str]] = []
    negative: list[dict[str, str]] = []
    for idx, row in enumerate(rows):
        review = clean_text(row.get("review", ""))
        if not review or review in seen:
            continue
        if len(review) < 6 or len(review) > max_chars:
            continue
        seen.add(review)
        item = {
            "review_id": f"R{idx + 1:06d}",
            "category": clean_text(row.get("cat", "")),
            "source_label": clean_text(row.get("label", "")),
            "review": review,
        }
        if item["source_label"] == "0":
            negative.append(item)
        else:
            positive.append(item)

    rng = random.Random(seed)
    rng.shuffle(negative)
    rng.shuffle(positive)
    neg_count = min(len(negative), int(limit * negative_ratio))
    pos_count = min(len(positive), limit - neg_count)
    selected = negative[:neg_count] + positive[:pos_count]
    if len(selected) < limit:
        used = {item["review_id"] for item in selected}
        remaining = [item for item in negative[neg_count:] + positive[pos_count:] if item["review_id"] not in used]
        selected.extend(remaining[: limit - len(selected)])
    rng.shuffle(selected)
    return selected[:limit]


def read_existing(path: Path) -> tuple[list[dict[str, str]], set[str]]:
    if not path.exists():
        return [], set()
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows, {row.get("review_id", "") for row in rows if row.get("label_status") == "pseudo_labeled"}


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "review_id",
        "category",
        "source_label",
        "review",
        "overall_sentiment",
        "triplets_json",
        "label_status",
        "error",
        "review_status",
        "reviewed_triplets_json",
        "human_notes",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def make_user_prompt(batch: list[dict[str, str]]) -> str:
    payload = [{"review_id": item["review_id"], "review": item["review"]} for item in batch]
    return json.dumps(payload, ensure_ascii=False, indent=2)


def parse_label_response(data: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(data, list):
        return {}
    parsed = {}
    for item in data:
        if isinstance(item, dict) and item.get("review_id"):
            parsed[str(item["review_id"])] = item
    return parsed


def normalize_triplets(triplets: Any) -> list[dict[str, str]]:
    if not isinstance(triplets, list):
        return []
    normalized = []
    for item in triplets:
        if not isinstance(item, dict):
            continue
        sentiment = str(item.get("sentiment", "NEU")).upper()
        if sentiment not in {"POS", "NEG", "NEU"}:
            sentiment = "NEU"
        normalized.append(
            {
                "aspect": str(item.get("aspect", "")).strip(),
                "opinion": str(item.get("opinion", "")).strip(),
                "sentiment": sentiment,
                "evidence": str(item.get("evidence", "")).strip(),
            }
        )
    return [item for item in normalized if item["aspect"] and item["opinion"]]


def main() -> int:
    args = parse_args()
    source = Path(args.source)
    output = Path(args.output)
    selected = load_reviews(source, args.limit, args.seed, args.max_chars, args.negative_ratio)
    existing_rows, done_ids = read_existing(output) if args.resume else ([], set())
    rows_by_id = {row["review_id"]: row for row in existing_rows if row.get("review_id")}

    pending = [item for item in selected if item["review_id"] not in done_ids]
    print(f"Selected {len(selected)} reviews; pending {len(pending)}; output={output}")
    if args.dry_run:
        print(json.dumps(selected[:3], ensure_ascii=False, indent=2))
        return 0

    llm = LLMClient(temperature=0)
    if not llm.available:
        raise RuntimeError("LLM is not available. Check LLM_API_KEY, LLM_MODEL_ID, and LLM_BASE_URL in .env.")

    all_rows = list(rows_by_id.values())
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start : start + args.batch_size]
        response = llm.invoke_json(SYSTEM_PROMPT, make_user_prompt(batch))
        labels = parse_label_response(response)

        for item in batch:
            label = labels.get(item["review_id"])
            if label:
                triplets = normalize_triplets(label.get("triplets"))
                row = {
                    **item,
                    "overall_sentiment": str(label.get("overall_sentiment", "")).upper(),
                    "triplets_json": json.dumps(triplets, ensure_ascii=False),
                    "label_status": "pseudo_labeled",
                    "error": "",
                    "review_status": "needs_review",
                    "reviewed_triplets_json": "",
                    "human_notes": "",
                }
            else:
                row = {
                    **item,
                    "overall_sentiment": "",
                    "triplets_json": "[]",
                    "label_status": "failed",
                    "error": "empty_or_invalid_llm_response",
                    "review_status": "needs_retry",
                    "reviewed_triplets_json": "",
                    "human_notes": "",
                }
            rows_by_id[item["review_id"]] = row

        all_rows = [rows_by_id[item["review_id"]] for item in selected if item["review_id"] in rows_by_id]
        write_rows(output, all_rows)
        print(f"Labeled {min(start + len(batch), len(pending))}/{len(pending)}")
        if args.sleep:
            time.sleep(args.sleep)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
