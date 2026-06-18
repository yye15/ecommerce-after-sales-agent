"""Export a balanced human-audit sample from pseudo-labeled ASTE data."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "labeled" / "deepseek_aste_300.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "labeled" / "deepseek_aste_audit_100.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export ASTE labels for manual audit.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    with input_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    negative = [row for row in rows if row.get("source_label") == "0"]
    positive = [row for row in rows if row.get("source_label") != "0"]
    rng = random.Random(args.seed)
    rng.shuffle(negative)
    rng.shuffle(positive)

    neg_count = min(len(negative), int(args.limit * 0.65))
    pos_count = min(len(positive), args.limit - neg_count)
    selected = negative[:neg_count] + positive[:pos_count]
    if len(selected) < args.limit:
        used = {row["review_id"] for row in selected}
        selected.extend([row for row in rows if row["review_id"] not in used][: args.limit - len(selected)])
    rng.shuffle(selected)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(selected[: args.limit])

    print(f"Exported {min(args.limit, len(selected))} rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
