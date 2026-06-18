"""Build a small Chinese ASTE dataset in LALUN JSON format.

The current project has Chinese e-commerce reviews and rule/LLM sentiment
triplets. LALUN fine-tuning needs span-level labels, so this script only keeps
triplets whose aspect and opinion strings can be located in the original text.

Output is written to a configurable LALUN data directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ecommerce_agent.agents.sentiment_agent import rule_based_sentiment  # noqa: E402


DEFAULT_INPUTS = [
    PROJECT_ROOT / "data" / "raw" / "chinese_shopping_sample.csv",
    PROJECT_ROOT / "data" / "eval" / "golden_candidates_50.csv",
]
DEFAULT_OUTPUT = Path(
    os.getenv("LALUN_OUTPUT_DIR", "external/LALUN/delivery_105/data/aste_data_bert/V2/zh_ecommerce")
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Chinese ASTE data for LALUN.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-examples", type=int, default=8)
    parser.add_argument(
        "--max-chars",
        type=int,
        default=80,
        help="Skip long reviews because LALUN's table encoder is memory-heavy.",
    )
    return parser.parse_args()


def read_csv_flexible(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            with path.open("r", encoding=encoding, newline="") as f:
                return list(csv.DictReader(f))
        except UnicodeDecodeError:
            continue
    return []


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", "", text or "")
    return text.strip()


def chars(text: str) -> list[str]:
    return [char for char in text if char.strip()]


def chain_adj(length: int) -> list[list[int]]:
    matrix = [[0 for _ in range(length)] for _ in range(length)]
    for i in range(length):
        matrix[i][i] = 1
        if i > 0:
            matrix[i][i - 1] = 1
        if i < length - 1:
            matrix[i][i + 1] = 1
    return matrix


def make_postag(length: int, spans: list[tuple[int, int, str]]) -> list[str]:
    tags = ["NN"] * length
    for start, end, role in spans:
        tag = "JJ" if role == "opinion" else "NN"
        for idx in range(start, end):
            if 0 <= idx < length:
                tags[idx] = tag
    return tags


def find_span(text: str, phrase: str) -> tuple[int, int] | None:
    phrase = clean_text(phrase)
    if not phrase:
        return None
    start = text.find(phrase)
    if start == -1:
        return None
    return start, start + len(phrase)


def build_example(index: int, text: str, max_chars: int | None = None) -> dict[str, Any] | None:
    text = clean_text(text)
    if not text:
        return None
    if max_chars and len(text) > max_chars:
        return None

    sentiment = rule_based_sentiment(text)
    pairs = []
    entities = []
    spans_for_postag: list[tuple[int, int, str]] = []
    seen_entities = set()

    for triplet in sentiment.get("triplets", []):
        aspect_span = find_span(text, str(triplet.get("aspect", "")))
        opinion_span = find_span(text, str(triplet.get("opinion", "")))
        polarity = str(triplet.get("sentiment", "NEU")).upper()
        if polarity not in {"POS", "NEG", "NEU"}:
            polarity = "NEU"
        if not aspect_span or not opinion_span:
            continue

        a_start, a_end = aspect_span
        o_start, o_end = opinion_span
        if a_start == a_end or o_start == o_end:
            continue

        pair = [a_start, a_end, o_start, o_end, polarity]
        if pair in pairs:
            continue
        pairs.append(pair)

        aspect_text = text[a_start:a_end]
        opinion_text = text[o_start:o_end]
        aspect_entity = ("target", a_start, a_end, aspect_text)
        opinion_entity = ("opinion", o_start, o_end, opinion_text)
        if aspect_entity not in seen_entities:
            seen_entities.add(aspect_entity)
            entities.append(["target", a_start, a_end, str([aspect_text]), aspect_text])
            spans_for_postag.append((a_start, a_end, "target"))
        if opinion_entity not in seen_entities:
            seen_entities.add(opinion_entity)
            entities.append(["opinion", o_start, o_end, str([opinion_text]), opinion_text])
            spans_for_postag.append((o_start, o_end, "opinion"))

    if not pairs:
        return None

    token_list = chars(text)
    return {
        "ID": index,
        "sentence": " ".join(token_list),
        "entities": entities,
        "pairs": pairs,
        "tokens": str(token_list),
        "adj": chain_adj(len(token_list)),
        "postag": make_postag(len(token_list), spans_for_postag),
    }


def load_review_texts() -> list[str]:
    texts = []
    seen = set()
    for path in DEFAULT_INPUTS:
        for row in read_csv_flexible(path):
            text = clean_text(row.get("review_text") or row.get("review") or row.get("text") or "")
            if text and text not in seen:
                seen.add(text)
                texts.append(text)
    return texts


def split_examples(examples: list[dict[str, Any]], seed: int) -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(seed)
    shuffled = examples[:]
    rng.shuffle(shuffled)
    total = len(shuffled)
    train_end = max(1, int(total * 0.7))
    dev_end = max(train_end + 1, int(total * 0.85)) if total >= 3 else train_end
    return {
        "train": shuffled[:train_end],
        "dev": shuffled[train_end:dev_end] or shuffled[:1],
        "test": shuffled[dev_end:] or shuffled[-1:],
    }


def write_json(path: Path, data: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    review_texts = load_review_texts()
    examples = []
    for text in review_texts:
        example = build_example(len(examples), text, max_chars=args.max_chars)
        if example:
            example["ID"] = len(examples)
            examples.append(example)

    splits = split_examples(examples, args.seed)
    for split, split_examples_ in splits.items():
        write_json(output_dir / f"{split}.json", split_examples_)

    report = {
        "input_reviews": len(review_texts),
        "usable_examples": len(examples),
        "output_dir": str(output_dir),
        "max_chars": args.max_chars,
        "splits": {name: len(items) for name, items in splits.items()},
        "warning": None,
    }
    if len(examples) < args.min_examples:
        report["warning"] = (
            "Very few span-labeled examples were produced. Use DeepSeek/manual "
            "ASTE labeling before serious fine-tuning."
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
