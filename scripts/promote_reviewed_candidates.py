"""Promote reviewed candidate rows into the golden evaluation CSV format."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "eval" / "golden_candidates_50.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "eval" / "golden_eval_reviewed.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert reviewed candidates to golden eval format.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Reviewed candidate CSV.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output golden eval CSV.")
    return parser.parse_args()


def is_reviewed(row: dict[str, str]) -> bool:
    return (
        row.get("human_expected_categories", "").strip()
        and row.get("human_expected_risk_level", "").strip()
        and row.get("human_expected_priority", "").strip()
    )


def convert_row(row: dict[str, str], index: int) -> dict[str, str]:
    return {
        "case_id": f"R{index:03d}",
        "review_text": row.get("review_text", ""),
        "expected_categories": row.get("human_expected_categories", ""),
        "expected_risk_level": row.get("human_expected_risk_level", ""),
        "expected_priority": row.get("human_expected_priority", ""),
        "expected_rule_reason": row.get("human_rule_reason", ""),
        "notes": f"from {row.get('candidate_id', '')}; product={row.get('product_category', '')}",
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    with input_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    reviewed = [convert_row(row, index + 1) for index, row in enumerate(rows) if is_reviewed(row)]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case_id",
        "review_text",
        "expected_categories",
        "expected_risk_level",
        "expected_priority",
        "expected_rule_reason",
        "notes",
    ]
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(reviewed)

    print(f"input_file={input_path}")
    print(f"output_file={output_path}")
    print(f"reviewed_count={len(reviewed)}")


if __name__ == "__main__":
    main()
