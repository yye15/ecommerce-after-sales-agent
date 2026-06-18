"""Create a reviewable golden-set candidate table from real reviews.

This script does not claim that agent predictions are golden labels. It only
uses the current agent output as suggestions, then leaves human label columns
blank for manual review.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ecommerce_agent.graph import run_case  # noqa: E402


DEFAULT_INPUT = PROJECT_ROOT / "data" / "raw" / "chinese_shopping_sample.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "eval" / "golden_candidates_50.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create human-reviewable golden-set candidates.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Prepared real review CSV.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output candidate CSV.")
    parser.add_argument("--limit", type=int, default=50, help="Number of candidates to export.")
    parser.add_argument("--use-llm", action="store_true", help="Use LLM suggestions instead of rules only.")
    return parser.parse_args()


def load_reviews(path: Path, limit: int) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[:limit]


def category_names(final: dict[str, Any]) -> str:
    return ";".join(
        item.get("category", "")
        for item in final.get("issue_categories", [])
        if item.get("category")
    )


def risk_reasons(final: dict[str, Any]) -> str:
    return " | ".join(final.get("risk", {}).get("reasons", []))


def build_candidate(row: dict[str, str], index: int, use_llm: bool) -> dict[str, Any]:
    text = row.get("review_text") or row.get("review") or row.get("text") or ""
    final = run_case(text, use_llm=use_llm).get("final_result", {})
    strategy = final.get("strategy", {})
    return {
        "candidate_id": f"C{index:03d}",
        "review_text": text,
        "product_category": row.get("product_category", ""),
        "source_sentiment_label": row.get("sentiment_label", ""),
        "source_rating_bucket": row.get("rating_bucket", ""),
        "suggested_categories": category_names(final),
        "suggested_risk_level": final.get("risk", {}).get("level", ""),
        "suggested_priority": strategy.get("priority", ""),
        "suggested_rule_reason": risk_reasons(final),
        "human_expected_categories": "",
        "human_expected_risk_level": "",
        "human_expected_priority": "",
        "human_rule_reason": "",
        "review_status": "needs_review",
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "candidate_id",
        "review_text",
        "product_category",
        "source_sentiment_label",
        "source_rating_bucket",
        "suggested_categories",
        "suggested_risk_level",
        "suggested_priority",
        "suggested_rule_reason",
        "human_expected_categories",
        "human_expected_risk_level",
        "human_expected_priority",
        "human_rule_reason",
        "review_status",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    source_rows = load_reviews(input_path, args.limit)
    candidates = [
        build_candidate(row, index=index + 1, use_llm=args.use_llm)
        for index, row in enumerate(source_rows)
    ]
    write_csv(candidates, output_path)
    print(f"input_file={input_path}")
    print(f"output_file={output_path}")
    print(f"candidate_count={len(candidates)}")
    print("status=needs_human_review")


if __name__ == "__main__":
    main()
