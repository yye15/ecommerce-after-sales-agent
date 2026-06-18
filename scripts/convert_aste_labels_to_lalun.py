"""Convert audited Chinese ASTE CSV labels into LALUN JSON files."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "labeled" / "deepseek_aste_300.csv"
DEFAULT_OUTPUT = Path(
    os.getenv("LALUN_OUTPUT_DIR", "external/LALUN/delivery_105/data/aste_data_bert/V2/zh_ecommerce")
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert ASTE labels to LALUN JSON.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-examples", type=int, default=20)
    return parser.parse_args()


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


def parse_triplets(row: dict[str, str]) -> list[dict[str, Any]]:
    text = row.get("reviewed_triplets_json") or row.get("triplets_json") or "[]"
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def find_span(text: str, phrase: str) -> tuple[int, int] | None:
    phrase = clean_text(phrase)
    if not phrase:
        return None
    start = text.find(phrase)
    if start == -1:
        return None
    return start, start + len(phrase)


def build_example(index: int, row: dict[str, str]) -> dict[str, Any] | None:
    text = clean_text(row.get("review", ""))
    if not text:
        return None

    pairs = []
    entities = []
    spans_for_postag: list[tuple[int, int, str]] = []
    seen_entities = set()

    for triplet in parse_triplets(row):
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
        "source_review_id": row.get("review_id", ""),
        "sentence": " ".join(token_list),
        "entities": entities,
        "pairs": pairs,
        "tokens": str(token_list),
        "adj": chain_adj(len(token_list)),
        "postag": make_postag(len(token_list), spans_for_postag),
    }


def split_examples(examples: list[dict[str, Any]], seed: int) -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(seed)
    shuffled = examples[:]
    rng.shuffle(shuffled)
    total = len(shuffled)
    train_end = max(1, int(total * 0.8))
    dev_end = max(train_end + 1, int(total * 0.9)) if total >= 3 else train_end
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
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    with input_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    accepted_status = {"", "needs_review", "reviewed", "accepted"}
    examples = []
    skipped_status = 0
    for row in rows:
        if row.get("review_status", "") not in accepted_status:
            skipped_status += 1
            continue
        example = build_example(len(examples), row)
        if example:
            example["ID"] = len(examples)
            examples.append(example)

    splits = split_examples(examples, args.seed)
    for split, split_examples_ in splits.items():
        write_json(output_dir / f"{split}.json", split_examples_)

    report = {
        "input_rows": len(rows),
        "usable_examples": len(examples),
        "skipped_status": skipped_status,
        "output_dir": str(output_dir),
        "splits": {name: len(items) for name, items in splits.items()},
        "warning": None,
    }
    if len(examples) < args.min_examples:
        report["warning"] = "Few usable examples. Check whether aspect/opinion strings appear in the original review."
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
