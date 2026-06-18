"""Prepare human-reviewable evaluation candidates from JDDC-style dialogues.

The official JDDC data may be distributed in different formats. This script is
therefore intentionally tolerant: it accepts JSONL, JSON and CSV files, extracts
multi-turn customer-service dialogues, filters after-sales-related cases, and
uses the current agent only as a label suggester.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ecommerce_agent.graph import run_case  # noqa: E402


DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "eval" / "jddc_eval_candidates_50.csv"

AFTER_SALES_KEYWORDS = [
    "退款",
    "退货",
    "换货",
    "售后",
    "客服",
    "投诉",
    "赔偿",
    "补偿",
    "物流",
    "快递",
    "发货",
    "没收到",
    "没送到",
    "坏了",
    "质量",
    "不能用",
    "不回复",
    "没人处理",
    "维修",
    "保修",
    "发票",
    "差评",
]

ROLE_KEYS = ["role", "speaker", "sender", "from", "identity", "type"]
TEXT_KEYS = ["content", "text", "utterance", "message", "query", "answer", "sentence"]
DIALOGUE_KEYS = ["messages", "dialogue", "dialog", "conversation", "turns", "utterances", "session"]
CUSTOMER_ROLE_HINTS = ["user", "customer", "buyer", "客户", "用户", "买家", "顾客"]
SERVICE_ROLE_HINTS = ["agent", "service", "seller", "客服", "商家", "系统", "assistant"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create JDDC evaluation candidates for human review.")
    parser.add_argument("--input", required=True, help="JDDC JSONL/JSON/CSV file path.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output candidate CSV path.")
    parser.add_argument("--limit", type=int, default=50, help="Number of candidates to export.")
    parser.add_argument("--min-turns", type=int, default=2, help="Minimum dialogue turns.")
    parser.add_argument("--use-llm", action="store_true", help="Use DeepSeek suggestions instead of rules only.")
    return parser.parse_args()


def load_records(path: Path) -> list[Any]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8-sig") as f:
            return [json.loads(line) for line in f if line.strip()]
    if suffix == ".json":
        with path.open("r", encoding="utf-8-sig") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        for key in ["data", "records", "dialogs", "dialogues", "conversations"]:
            if isinstance(data.get(key), list):
                return data[key]
        return [data]
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
    raise ValueError(f"Unsupported input format: {path}")


def extract_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in TEXT_KEYS:
            if key in value and value[key]:
                return str(value[key]).strip()
    return str(value).strip()


def normalize_role(value: Any) -> str:
    role = str(value or "").strip().lower()
    if any(hint in role for hint in CUSTOMER_ROLE_HINTS):
        return "用户"
    if any(hint in role for hint in SERVICE_ROLE_HINTS):
        return "坐席"
    return "说话人"


def iter_turns_from_sequence(items: Iterable[Any]) -> list[dict[str, str]]:
    turns = []
    for index, item in enumerate(items, start=1):
        if isinstance(item, dict):
            role_value = next((item.get(key) for key in ROLE_KEYS if item.get(key)), "")
            role = normalize_role(role_value) if role_value else f"说话人{index}"
            text = extract_text(item)
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            role = normalize_role(item[0])
            text = extract_text(item[1])
        else:
            role = f"说话人{index}"
            text = extract_text(item)
        if text:
            turns.append({"role": role, "text": text})
    return turns


def extract_dialogue(record: Any) -> tuple[str, int]:
    if isinstance(record, str):
        lines = [line.strip() for line in record.splitlines() if line.strip()]
        return "\n".join(lines), len(lines)

    if not isinstance(record, dict):
        text = extract_text(record)
        return text, 1 if text else 0

    for key in DIALOGUE_KEYS:
        value = record.get(key)
        if isinstance(value, list):
            turns = iter_turns_from_sequence(value)
            text = "\n".join(f"{turn['role']}：{turn['text']}" for turn in turns)
            return text, len(turns)
        if isinstance(value, str) and value.strip():
            lines = [line.strip() for line in value.splitlines() if line.strip()]
            return "\n".join(lines), len(lines)

    customer_text = extract_text(record.get("customer") or record.get("user") or record.get("query"))
    service_text = extract_text(record.get("service") or record.get("agent") or record.get("answer") or record.get("response"))
    turns = []
    if customer_text:
        turns.append(f"用户：{customer_text}")
    if service_text:
        turns.append(f"客服：{service_text}")
    if turns:
        return "\n".join(turns), len(turns)

    text = extract_text(record)
    return text, 1 if text else 0


def is_after_sales_dialogue(text: str, turns_count: int, min_turns: int) -> bool:
    if turns_count < min_turns:
        return False
    return any(keyword in text for keyword in AFTER_SALES_KEYWORDS)


def category_names(final: dict[str, Any]) -> str:
    return ";".join(
        item.get("category", "")
        for item in final.get("issue_categories", [])
        if item.get("category")
    )


def risk_reasons(final: dict[str, Any]) -> str:
    return " | ".join(final.get("risk", {}).get("reasons", []))


def build_candidate(
    record: Any,
    index: int,
    source_path: Path,
    source_row_id: str,
    dialogue_text: str,
    turns_count: int,
    use_llm: bool,
) -> dict[str, Any]:
    final = run_case(dialogue_text, use_llm=use_llm).get("final_result", {})
    return {
        "candidate_id": f"JD{index:03d}",
        "dialogue_text": dialogue_text,
        "source_file": str(source_path),
        "source_row_id": source_row_id,
        "turns_count": turns_count,
        "suggested_categories": category_names(final),
        "suggested_risk_level": final.get("risk", {}).get("level", ""),
        "suggested_priority": final.get("risk", {}).get("priority") or final.get("strategy", {}).get("priority", ""),
        "suggested_rule_reason": risk_reasons(final),
        "human_expected_categories": "",
        "human_expected_risk_level": "",
        "human_expected_priority": "",
        "human_rule_reason": "",
        "review_status": "needs_review",
    }


def write_candidates(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "candidate_id",
        "dialogue_text",
        "source_file",
        "source_row_id",
        "turns_count",
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


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    records = load_records(input_path)
    candidates: list[dict[str, Any]] = []

    for row_index, record in enumerate(records, start=1):
        dialogue_text, turns_count = extract_dialogue(record)
        if not is_after_sales_dialogue(dialogue_text, turns_count, args.min_turns):
            continue
        candidates.append(
            build_candidate(
                record=record,
                index=len(candidates) + 1,
                source_path=input_path,
                source_row_id=str(row_index),
                dialogue_text=dialogue_text,
                turns_count=turns_count,
                use_llm=args.use_llm,
            )
        )
        if len(candidates) >= args.limit:
            break

    write_candidates(candidates, output_path)
    print(f"input_file={input_path}")
    print(f"output_file={output_path}")
    print(f"source_records={len(records)}")
    print(f"candidate_count={len(candidates)}")
    print("status=needs_human_review")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
