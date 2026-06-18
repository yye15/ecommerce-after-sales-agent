"""Evaluate the e-commerce agent against a human-labeled golden set."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ecommerce_agent.graph import run_case  # noqa: E402


DEFAULT_INPUT = PROJECT_ROOT / "data" / "eval" / "customer_service_eval_50.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "eval" / "customer_service_eval_results.csv"
DEFAULT_ERROR_OUTPUT = PROJECT_ROOT / "data" / "eval" / "customer_service_eval_errors.csv"
DEFAULT_SUMMARY_OUTPUT = PROJECT_ROOT / "data" / "eval" / "customer_service_eval_summary.json"

RISK_RANK = {"低风险": 1, "中风险": 2, "高风险": 3}
PRIORITY_RANK = {"P3": 1, "P2": 2, "P1": 3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate agent outputs on a human-labeled set.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Golden CSV path.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Detailed result CSV path.")
    parser.add_argument("--errors-output", default=str(DEFAULT_ERROR_OUTPUT), help="Error analysis CSV path.")
    parser.add_argument("--summary-output", default=str(DEFAULT_SUMMARY_OUTPUT), help="Summary JSON path.")
    parser.add_argument("--use-llm", action="store_true", help="Use the configured LLM instead of rule fallback.")
    parser.add_argument(
        "--fail-under",
        type=float,
        default=None,
        help="Exit with code 1 if average score is below this value, for example 0.8.",
    )
    return parser.parse_args()


def split_expected_categories(value: str) -> set[str]:
    return {item.strip() for item in (value or "").split(";") if item.strip()}


def load_golden_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def extract_final_result(agent_result: dict[str, Any]) -> dict[str, Any]:
    return agent_result.get("final_result", agent_result)


def risk_gap(expected: str, actual: str) -> int | None:
    if expected not in RISK_RANK or actual not in RISK_RANK:
        return None
    return RISK_RANK[actual] - RISK_RANK[expected]


def priority_gap(expected: str, actual: str) -> int | None:
    expected = "P3" if expected == "P4" else expected
    actual = "P3" if actual == "P4" else actual
    if expected not in PRIORITY_RANK or actual not in PRIORITY_RANK:
        return None
    return PRIORITY_RANK[actual] - PRIORITY_RANK[expected]


def evaluate_row(row: dict[str, str], use_llm: bool) -> dict[str, Any]:
    start = time.perf_counter()
    final = extract_final_result(run_case(row["review_text"], use_llm=use_llm))
    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    expected_categories = split_expected_categories(row.get("expected_categories", ""))
    predicted_categories = {
        item.get("category", "")
        for item in final.get("issue_categories", [])
        if item.get("category")
    }

    expected_risk = row.get("expected_risk_level", "").strip()
    actual_risk = final.get("risk", {}).get("level", "")
    expected_priority = (row.get("expected_priority", "") or "").strip()
    if expected_priority == "P4":
        expected_priority = "P3"
    actual_priority = (
        final.get("risk", {}).get("priority")
        or final.get("strategy", {}).get("priority")
        or ""
    )

    category_overlap_match = bool(expected_categories & predicted_categories) if expected_categories else True
    category_exact_match = expected_categories == predicted_categories if expected_categories else True
    risk_match = expected_risk == actual_risk
    priority_match = expected_priority == actual_priority
    gap = risk_gap(expected_risk, actual_risk)
    p_gap = priority_gap(expected_priority, actual_priority)

    error_types = []
    if not category_overlap_match:
        error_types.append("category_miss")
    elif not category_exact_match:
        error_types.append("category_partial")
    if gap is not None and gap < 0:
        error_types.append("risk_underestimated")
    elif gap is not None and gap > 0:
        error_types.append("risk_overestimated")
    elif not risk_match:
        error_types.append("risk_mismatch")
    if not priority_match:
        error_types.append("priority_mismatch")

    return {
        "case_id": row.get("case_id", ""),
        "review_text": row.get("review_text", ""),
        "expected_categories": ";".join(sorted(expected_categories)),
        "actual_categories": ";".join(sorted(predicted_categories)),
        "category_match": int(category_overlap_match),
        "category_exact_match": int(category_exact_match),
        "expected_risk_level": expected_risk,
        "actual_risk_level": actual_risk,
        "risk_match": int(risk_match),
        "risk_gap": "" if gap is None else gap,
        "actual_risk_score": final.get("risk", {}).get("score", ""),
        "expected_priority": expected_priority,
        "actual_priority": actual_priority,
        "priority_match": int(priority_match),
        "priority_gap": "" if p_gap is None else p_gap,
        "expected_rule_reason": row.get("expected_rule_reason", ""),
        "actual_risk_reasons": " | ".join(final.get("risk", {}).get("reasons", [])),
        "triggered_rules": ";".join(
            rule.get("rule_id", "") for rule in final.get("risk", {}).get("triggered_rules", [])
        ),
        "latency_ms": latency_ms,
        "error_types": ";".join(error_types),
    }


def _safe_rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def summarize(results: list[dict[str, Any]]) -> dict[str, float | int | dict[str, int]]:
    total = len(results)
    if total == 0:
        return {
            "total": 0,
            "category_accuracy": 0.0,
            "category_exact_accuracy": 0.0,
            "risk_accuracy": 0.0,
            "priority_accuracy": 0.0,
            "high_risk_recall": 0.0,
            "high_risk_precision": 0.0,
            "high_risk_f1": 0.0,
            "under_risk_rate": 0.0,
            "over_risk_rate": 0.0,
            "average_score": 0.0,
        }

    category_accuracy = sum(int(row.get("category_match", 0)) for row in results) / total
    category_exact_accuracy = sum(int(row.get("category_exact_match", row.get("category_match", 0))) for row in results) / total
    risk_accuracy = sum(int(row.get("risk_match", 0)) for row in results) / total
    priority_accuracy = sum(int(row.get("priority_match", 0)) for row in results) / total

    expected_high = [row for row in results if row.get("expected_risk_level") == "高风险"]
    actual_high = [row for row in results if row.get("actual_risk_level") == "高风险"]
    true_high = [
        row
        for row in results
        if row.get("expected_risk_level") == "高风险" and row.get("actual_risk_level") == "高风险"
    ]
    high_risk_recall = _safe_rate(len(true_high), len(expected_high))
    high_risk_precision = _safe_rate(len(true_high), len(actual_high))
    high_risk_f1 = (
        2 * high_risk_precision * high_risk_recall / (high_risk_precision + high_risk_recall)
        if high_risk_precision + high_risk_recall
        else 0.0
    )

    under_risk_count = sum(
        1
        for row in results
        if row.get("risk_gap") != "" and int(row.get("risk_gap", 0)) < 0
    )
    over_risk_count = sum(
        1
        for row in results
        if row.get("risk_gap") != "" and int(row.get("risk_gap", 0)) > 0
    )
    severe_under_risk_count = sum(
        1
        for row in results
        if row.get("risk_gap") != "" and int(row.get("risk_gap", 0)) <= -2
    )

    risk_distribution: dict[str, int] = {}
    expected_risk_distribution: dict[str, int] = {}
    for row in results:
        risk_distribution[str(row.get("actual_risk_level", ""))] = risk_distribution.get(str(row.get("actual_risk_level", "")), 0) + 1
        expected_risk_distribution[str(row.get("expected_risk_level", ""))] = expected_risk_distribution.get(str(row.get("expected_risk_level", "")), 0) + 1

    average_latency_ms = sum(float(row.get("latency_ms", 0) or 0) for row in results) / total
    average_score = (
        risk_accuracy * 0.35
        + high_risk_recall * 0.25
        + category_accuracy * 0.2
        + priority_accuracy * 0.1
        + (1 - under_risk_count / total) * 0.1
    )

    return {
        "total": total,
        "category_accuracy": category_accuracy,
        "category_exact_accuracy": category_exact_accuracy,
        "risk_accuracy": risk_accuracy,
        "priority_accuracy": priority_accuracy,
        "high_risk_recall": high_risk_recall,
        "high_risk_precision": high_risk_precision,
        "high_risk_f1": high_risk_f1,
        "under_risk_count": under_risk_count,
        "over_risk_count": over_risk_count,
        "severe_under_risk_count": severe_under_risk_count,
        "under_risk_rate": under_risk_count / total,
        "over_risk_rate": over_risk_count / total,
        "average_latency_ms": average_latency_ms,
        "average_score": average_score,
        "expected_risk_distribution": expected_risk_distribution,
        "actual_risk_distribution": risk_distribution,
    }


FIELDNAMES = [
    "case_id",
    "review_text",
    "expected_categories",
    "actual_categories",
    "category_match",
    "category_exact_match",
    "expected_risk_level",
    "actual_risk_level",
    "risk_match",
    "risk_gap",
    "actual_risk_score",
    "expected_priority",
    "actual_priority",
    "priority_match",
    "priority_gap",
    "expected_rule_reason",
    "actual_risk_reasons",
    "triggered_rules",
    "latency_ms",
    "error_types",
]


def write_results(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def write_error_rows(rows: list[dict[str, Any]], path: Path) -> None:
    error_rows = [row for row in rows if row.get("error_types")]
    write_results(error_rows, path)


def write_summary(summary: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def print_summary(summary: dict[str, Any], output_path: Path, error_path: Path, summary_path: Path) -> None:
    print("=" * 72)
    print("Customer service agent evaluation summary")
    print("=" * 72)
    print(f"cases: {summary['total']}")
    print(f"category_accuracy: {summary['category_accuracy']:.2%}")
    print(f"category_exact_accuracy: {summary['category_exact_accuracy']:.2%}")
    print(f"risk_accuracy: {summary['risk_accuracy']:.2%}")
    print(f"priority_accuracy: {summary['priority_accuracy']:.2%}")
    print(f"high_risk_recall: {summary['high_risk_recall']:.2%}")
    print(f"high_risk_precision: {summary['high_risk_precision']:.2%}")
    print(f"high_risk_f1: {summary['high_risk_f1']:.2%}")
    print(f"under_risk_rate: {summary['under_risk_rate']:.2%}")
    print(f"over_risk_rate: {summary['over_risk_rate']:.2%}")
    print(f"average_latency_ms: {summary['average_latency_ms']:.2f}")
    print(f"average_score: {summary['average_score']:.2%}")
    print(f"details: {output_path}")
    print(f"errors: {error_path}")
    print(f"summary_json: {summary_path}")


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    error_path = Path(args.errors_output)
    summary_path = Path(args.summary_output)
    rows = load_golden_rows(input_path)
    results = [evaluate_row(row, use_llm=args.use_llm) for row in rows]
    write_results(results, output_path)
    write_error_rows(results, error_path)
    summary = summarize(results)
    write_summary(summary, summary_path)
    print_summary(summary, output_path, error_path, summary_path)

    if args.fail_under is not None and float(summary["average_score"]) < args.fail_under:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
