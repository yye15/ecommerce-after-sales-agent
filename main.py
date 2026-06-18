"""Command-line entry for the e-commerce customer service agent."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ecommerce_agent.agents.operation_agent import build_operation_report
from ecommerce_agent.graph import run_case
from ecommerce_agent.lalun_adapter import inspect_lalun


DEFAULT_TEXT = "这个耳机音质不错，但是物流太慢，客服也一直不回复，我有点想投诉。"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="多智能体电商客服与售后运营 Agent")
    parser.add_argument("--text", default=DEFAULT_TEXT, help="单条评论或售后对话")
    parser.add_argument("--batch", help="CSV 文件路径，需包含 text 或 review 列")
    parser.add_argument("--no-llm", action="store_true", help="不调用大模型，只使用规则兜底")
    parser.add_argument("--json", action="store_true", help="输出完整 JSON")
    parser.add_argument("--lalun-status", action="store_true", help="检查本地 LALUN 模型状态")
    return parser.parse_args()


def load_batch(path: str) -> list[str]:
    rows: list[str] = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get("text") or row.get("review") or row.get("content")
            if text:
                rows.append(text.strip())
    return rows


def print_human_result(result: dict) -> None:
    final = result.get("final_result", result)
    print("\n" + "=" * 72)
    print("多智能体售后分析结果")
    print("=" * 72)
    print(f"输入文本：{final.get('input')}")

    sentiment = final.get("sentiment", {})
    print(f"\n情绪概括：{sentiment.get('summary')}")
    print("情感三元组：")
    for item in sentiment.get("triplets", []):
        print(f"- {item.get('aspect')} / {item.get('opinion')} / {item.get('sentiment')}")

    print("\n问题分类：")
    for item in final.get("issue_categories", []):
        print(f"- {item.get('category')}，置信度 {item.get('confidence'):.2f}")

    risk = final.get("risk", {})
    print(f"\n风险等级：{risk.get('level')}，分数 {risk.get('score')}")
    for reason in risk.get("reasons", []):
        print(f"- {reason}")

    print("\n售后策略：")
    strategy = final.get("strategy", {})
    print(f"优先级：{strategy.get('priority')}")
    for action in strategy.get("actions", []):
        print(f"- [{action.get('owner')}] {action.get('action')}：{action.get('detail')}")

    print("\n客服回复：")
    print(final.get("reply"))


def main() -> None:
    args = parse_args()
    if args.lalun_status:
        print(json.dumps(inspect_lalun(), ensure_ascii=False, indent=2))
        return

    if args.batch:
        texts = load_batch(args.batch)
        results = [run_case(text, use_llm=not args.no_llm) for text in texts]
        report = build_operation_report(results)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    result = run_case(args.text, use_llm=not args.no_llm)
    if args.json:
        print(json.dumps(result.get("final_result", result), ensure_ascii=False, indent=2))
    else:
        print_human_result(result)


if __name__ == "__main__":
    main()
