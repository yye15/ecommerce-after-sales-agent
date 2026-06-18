"""Prepare a small Chinese MARC review sample for agent evaluation.

This script reads the Chinese split of the Multilingual Amazon Reviews Corpus
from HuggingFace in streaming mode, cleans reviews, balances star ratings, and
writes a compact CSV for later LLM pre-labeling and human audit.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable
from urllib.request import urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "raw" / "marc_zh_sample.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Chinese MARC review samples.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output CSV path.")
    parser.add_argument(
        "--input-file",
        help="Optional local MARC JSONL/CSV file. When set, no network request is made.",
    )
    parser.add_argument("--target-size", type=int, default=300, help="Total sample size.")
    parser.add_argument("--min-length", type=int, default=8, help="Minimum review length.")
    parser.add_argument("--max-length", type=int, default=260, help="Maximum review length.")
    parser.add_argument("--split", default="train", choices=["train", "validation", "test"])
    parser.add_argument("--seed", type=int, default=42, help="Reserved for future random sampling.")
    return parser.parse_args()


def normalize_text(text: str) -> str:
    text = (text or "").replace("\u3000", " ")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_meaningful_chinese_review(text: str, min_length: int, max_length: int) -> bool:
    if not (min_length <= len(text) <= max_length):
        return False
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
    if len(chinese_chars) < max(4, min_length // 2):
        return False
    if len(set(text)) <= 3:
        return False
    low_value_patterns = ["好评", "不错不错", "很好很好", "赞赞赞"]
    if text in low_value_patterns:
        return False
    return True


def risk_bucket_from_rating(stars: int) -> str:
    if stars <= 2:
        return "negative"
    if stars == 3:
        return "mixed"
    return "positive"


def wanted_quotas(target_size: int) -> dict[str, int]:
    negative = int(target_size * 0.5)
    mixed = int(target_size * 0.3)
    positive = target_size - negative - mixed
    return {"negative": negative, "mixed": mixed, "positive": positive}


def load_marc_zh_stream(split: str) -> Iterable[dict]:
    split_name = "dev" if split == "validation" else split
    url = (
        "https://amazon-reviews-ml.s3-us-west-2.amazonaws.com/"
        f"json/{split_name}/dataset_zh_{split_name}.json"
    )
    try:
        response = urlopen(url, timeout=60)
    except Exception as exc:
        raise SystemExit(
            "无法访问 MARC 官方 S3 数据文件。请检查网络，或手动下载："
            f"{url}"
        ) from exc

    for raw_line in response:
        line = raw_line.decode("utf-8").strip()
        if not line:
            continue
        yield json.loads(line)


def load_local_reviews(path: str) -> Iterable[dict]:
    input_path = Path(path)
    if not input_path.exists():
        raise SystemExit(f"本地输入文件不存在：{input_path}")

    suffix = input_path.suffix.lower()
    if suffix in {".jsonl", ".json"}:
        with input_path.open("r", encoding="utf-8-sig") as f:
            first = f.read(1)
            f.seek(0)
            if first == "[":
                data = json.load(f)
                for item in data:
                    yield item
            else:
                for line in f:
                    line = line.strip()
                    if line:
                        yield json.loads(line)
        return

    if suffix == ".csv":
        with input_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                yield row
        return

    raise SystemExit("仅支持 .jsonl、.json、.csv 本地文件。")


def prepare_reviews(args: argparse.Namespace) -> list[dict]:
    quotas = wanted_quotas(args.target_size)
    counts: Counter[str] = Counter()
    seen_texts: set[str] = set()
    rows: list[dict] = []

    source_iter = load_local_reviews(args.input_file) if args.input_file else load_marc_zh_stream(args.split)

    for item in source_iter:
        stars = int(item.get("stars") or item.get("rating") or 0)
        if stars < 1 or stars > 5:
            continue

        bucket = risk_bucket_from_rating(stars)
        if counts[bucket] >= quotas[bucket]:
            if all(counts[key] >= value for key, value in quotas.items()):
                break
            continue

        review_text = normalize_text(
            item.get("review_body")
            or item.get("review_text")
            or item.get("text")
            or item.get("content")
            or ""
        )
        if not is_meaningful_chinese_review(review_text, args.min_length, args.max_length):
            continue
        if review_text in seen_texts:
            continue
        seen_texts.add(review_text)

        rows.append(
            {
                "id": len(rows) + 1,
                "source": "MARC",
                "language": item.get("language", "zh"),
                "product_category": item.get("product_category", ""),
                "rating": stars,
                "rating_bucket": bucket,
                "review_title": normalize_text(item.get("review_title") or item.get("title") or ""),
                "review_text": review_text,
            }
        )
        counts[bucket] += 1

        if all(counts[key] >= value for key, value in quotas.items()):
            break

    return rows


def write_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "source",
        "language",
        "product_category",
        "rating",
        "rating_bucket",
        "review_title",
        "review_text",
    ]
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = prepare_reviews(args)
    output_path = Path(args.output)
    write_csv(rows, output_path)

    counts = Counter(row["rating_bucket"] for row in rows)
    print(f"已写入：{output_path}")
    print(f"样本数：{len(rows)}")
    print(f"分布：{dict(counts)}")
    if len(rows) < args.target_size:
        print("提示：样本数未达到目标，可能是网络中断或清洗条件过严。")


if __name__ == "__main__":
    main()
