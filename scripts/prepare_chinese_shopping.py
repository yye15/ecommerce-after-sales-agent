"""Prepare Chinese online shopping reviews from ChineseNlpCorpus.

Dataset: SophonPlus/ChineseNlpCorpus online_shopping_10_cats
Source: https://github.com/SophonPlus/ChineseNlpCorpus

The source CSV has columns:
- cat: product/service category
- label: 1 positive, 0 negative
- review: review text
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from urllib.request import urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_ZIP = RAW_DIR / "online_shopping_10_cats.zip"
DEFAULT_EXTRACT_DIR = RAW_DIR / "online_shopping_10_cats"
DEFAULT_OUTPUT = RAW_DIR / "chinese_shopping_sample.csv"
DOWNLOAD_URL = (
    "https://github.com/SophonPlus/ChineseNlpCorpus/raw/master/"
    "datasets/online_shopping_10_cats/online_shopping_10_cats.zip"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Chinese online shopping reviews.")
    parser.add_argument("--target-size", type=int, default=300, help="Total sample size.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output CSV path.")
    parser.add_argument("--raw-csv", help="Optional existing online_shopping_10_cats.csv path.")
    parser.add_argument("--min-length", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=260)
    parser.add_argument("--categories", nargs="*", help="Optional category whitelist.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling.")
    return parser.parse_args()


def normalize_text(text: str) -> str:
    text = (text or "").replace("\u3000", " ")
    text = text.replace("\ufeff", "")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_meaningful_review(text: str, min_length: int, max_length: int) -> bool:
    if not (min_length <= len(text) <= max_length):
        return False
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
    if len(chinese_chars) < max(4, min_length // 2):
        return False
    if len(set(text)) <= 3:
        return False
    return True


def download_file(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0:
        return

    with urlopen(url, timeout=90) as response:
        data = response.read()
    output_path.write_bytes(data)


def find_csv_in_extract_dir(extract_dir: Path) -> Path | None:
    candidates = list(extract_dir.rglob("online_shopping_10_cats.csv"))
    return candidates[0] if candidates else None


def ensure_dataset(raw_csv: str | None = None) -> Path:
    if raw_csv:
        path = Path(raw_csv)
        if not path.exists():
            raise SystemExit(f"指定的 CSV 不存在：{path}")
        return path

    existing = find_csv_in_extract_dir(DEFAULT_EXTRACT_DIR)
    if existing:
        return existing

    download_file(DOWNLOAD_URL, DEFAULT_ZIP)
    DEFAULT_EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(DEFAULT_ZIP, "r") as zf:
        zf.extractall(DEFAULT_EXTRACT_DIR)

    csv_path = find_csv_in_extract_dir(DEFAULT_EXTRACT_DIR)
    if not csv_path:
        raise SystemExit("压缩包中未找到 online_shopping_10_cats.csv")
    return csv_path


def wanted_quotas(target_size: int) -> dict[str, int]:
    negative = int(target_size * 0.6)
    positive = target_size - negative
    return {"negative": negative, "positive": positive}


def prepare_rows(csv_path: Path, args: argparse.Namespace) -> list[dict]:
    categories = set(args.categories or [])
    seen: set[str] = set()
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for item in reader:
            category = normalize_text(item.get("cat", ""))
            if categories and category not in categories:
                continue

            label = str(item.get("label", "")).strip()
            if label not in {"0", "1"}:
                continue

            review = normalize_text(item.get("review", ""))
            if not is_meaningful_review(review, args.min_length, args.max_length):
                continue
            if review in seen:
                continue
            seen.add(review)

            bucket = "positive" if label == "1" else "negative"
            buckets[(category, bucket)].append(
                {
                    "source": "ChineseNlpCorpus/online_shopping_10_cats",
                    "product_category": category,
                    "sentiment_label": int(label),
                    "rating_bucket": bucket,
                    "review_text": review,
                }
            )

    rng = random.Random(args.seed)
    for bucket_rows in buckets.values():
        rng.shuffle(bucket_rows)

    rows: list[dict] = []
    counts: Counter[str] = Counter()
    quotas = wanted_quotas(args.target_size)
    ordered_categories = sorted({category for category, _ in buckets})

    for bucket in ("negative", "positive"):
        while counts[bucket] < quotas[bucket]:
            progressed = False
            for category in ordered_categories:
                if counts[bucket] >= quotas[bucket]:
                    break
                candidate_rows = buckets.get((category, bucket), [])
                if not candidate_rows:
                    continue
                row = candidate_rows.pop()
                row["id"] = len(rows) + 1
                rows.append(row)
                counts[bucket] += 1
                progressed = True
            if not progressed:
                break

    return rows


def write_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "source",
        "product_category",
        "sentiment_label",
        "rating_bucket",
        "review_text",
    ]
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    csv_path = ensure_dataset(args.raw_csv)
    rows = prepare_rows(csv_path, args)
    output_path = Path(args.output)
    write_csv(rows, output_path)

    print(f"source_file={csv_path}")
    print(f"output_file={output_path}")
    print(f"sample_count={len(rows)}")
    print(f"sentiment_distribution={dict(Counter(row['rating_bucket'] for row in rows))}")
    print(f"category_distribution={dict(Counter(row['product_category'] for row in rows))}")


if __name__ == "__main__":
    main()
