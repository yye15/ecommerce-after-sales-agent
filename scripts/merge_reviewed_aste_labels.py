"""Merge reviewed ASTE audit labels back into the full pseudo-labeled CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FULL = PROJECT_ROOT / "data" / "labeled" / "deepseek_aste_300.csv"
DEFAULT_REVIEWED = PROJECT_ROOT / "data" / "labeled" / "deepseek_aste_audit_100_reviewed.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "labeled" / "deepseek_aste_300_mixed_reviewed.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge reviewed ASTE labels.")
    parser.add_argument("--full", default=str(DEFAULT_FULL))
    parser.add_argument("--reviewed", default=str(DEFAULT_REVIEWED))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main() -> int:
    args = parse_args()
    full_path = Path(args.full)
    reviewed_path = Path(args.reviewed)
    output_path = Path(args.output)

    full_rows = read_csv(full_path)
    reviewed_rows = {row["review_id"]: row for row in read_csv(reviewed_path)}

    merged_count = 0
    rejected_count = 0
    for row in full_rows:
        reviewed = reviewed_rows.get(row["review_id"])
        if not reviewed:
            continue
        row["review_status"] = reviewed.get("review_status", row.get("review_status", ""))
        row["reviewed_triplets_json"] = reviewed.get("reviewed_triplets_json", "")
        row["human_notes"] = reviewed.get("human_notes", "")
        merged_count += 1
        if row["review_status"] == "rejected":
            rejected_count += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=full_rows[0].keys())
        writer.writeheader()
        writer.writerows(full_rows)

    print(f"Merged {merged_count} reviewed rows, including {rejected_count} rejected rows.")
    print(f"Output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
