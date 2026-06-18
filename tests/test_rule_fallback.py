from ecommerce_agent.agents.operation_agent import build_operation_report
from ecommerce_agent.agents.risk_agent import assess_risk
from ecommerce_agent.agents.sentiment_agent import rule_based_sentiment
from ecommerce_agent.graph import run_case


def test_rule_sentiment_extracts_mixed_triplets():
    result = rule_based_sentiment("这个耳机音质不错，但是物流太慢，客服也一直不回复。")
    triplets = result["triplets"]
    assert any(item["aspect"] == "音质" and item["sentiment"] == "POS" for item in triplets)
    assert any(item["aspect"] == "物流" and item["sentiment"] == "NEG" for item in triplets)
    assert any(item["aspect"] == "客服" and item["sentiment"] == "NEG" for item in triplets)


def test_risk_high_for_complaint():
    sentiment = rule_based_sentiment("包装破了，客服一直不回复，我要投诉。")
    risk = assess_risk("包装破了，客服一直不回复，我要投诉。", sentiment, [])
    assert risk["level"] == "高风险"
    assert risk["should_escalate"] is True


def test_graph_runs_without_llm():
    result = run_case("物流太慢了，客服也不回复。", use_llm=False)
    final = result["final_result"]
    assert final["risk"]["level"] in {"中风险", "高风险"}
    assert final["reply"]
    assert final["strategy"]["actions"]


def test_operation_report_counts_cases():
    results = [
        run_case("物流太慢了。", use_llm=False),
        run_case("衣服质量很好。", use_llm=False),
    ]
    report = build_operation_report(results)
    assert report["total_cases"] == 2
    assert report["recommendations"]
