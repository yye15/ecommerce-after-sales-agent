"""Merge the base 50-case eval set with reviewed dialogue candidates."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = PROJECT_ROOT / "data" / "eval" / "customer_service_eval_50.csv"
DEFAULT_CANDIDATES = PROJECT_ROOT / "data" / "eval" / "jddc_eval_candidates_50.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "eval" / "customer_service_eval_100.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge reviewed JDDC candidates into the evaluation set.")
    parser.add_argument("--base", default=str(DEFAULT_BASE), help="Base evaluation CSV.")
    parser.add_argument("--candidates", default=str(DEFAULT_CANDIDATES), help="Reviewed candidate CSV.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Merged evaluation CSV.")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def is_reviewed_candidate(row: dict[str, str]) -> bool:
    return (
        row.get("human_expected_categories", "").strip()
        and row.get("human_expected_risk_level", "").strip()
        and row.get("human_expected_priority", "").strip()
    )


def normalize_base_row(row: dict[str, str]) -> dict[str, str]:
    return {
        "review_text": row.get("review_text", ""),
        "expected_categories": row.get("expected_categories", ""),
        "expected_risk_level": row.get("expected_risk_level", ""),
        "expected_priority": row.get("expected_priority", ""),
        "expected_rule_reason": row.get("expected_rule_reason", ""),
        "notes": row.get("notes", ""),
    }


def convert_candidate(row: dict[str, str]) -> dict[str, str]:
    source_note = row.get("source_file", "")
    if row.get("source_review_text"):
        source_note = f"synthetic_dialogue_from_real_review; product={row.get('product_category', '')}"
    return {
        "review_text": row.get("dialogue_text", "") or row.get("review_text", ""),
        "expected_categories": row.get("human_expected_categories", ""),
        "expected_risk_level": row.get("human_expected_risk_level", ""),
        "expected_priority": row.get("human_expected_priority", ""),
        "expected_rule_reason": row.get("human_rule_reason", ""),
        "notes": f"from {row.get('candidate_id', '')}; source={source_note}; row={row.get('source_row_id', '')}",
    }


def write_eval(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case_id",
        "review_text",
        "expected_categories",
        "expected_risk_level",
        "expected_priority",
        "expected_rule_reason",
        "notes",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(rows, start=1):
            output = {"case_id": f"R{index:03d}"}
            output.update(row)
            writer.writerow(output)


def main() -> int:
    args = parse_args()
    base_path = Path(args.base)
    candidate_path = Path(args.candidates)
    output_path = Path(args.output)

    base_rows = [normalize_base_row(row) for row in read_csv(base_path)]
    candidate_rows = [
        convert_candidate(row)
        for row in read_csv(candidate_path)
        if is_reviewed_candidate(row)
    ]
    merged = base_rows + candidate_rows
    write_eval(merged, output_path)

    print(f"base_count={len(base_rows)}")
    print(f"reviewed_candidate_count={len(candidate_rows)}")
    print(f"merged_count={len(merged)}")
    print(f"output_file={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
