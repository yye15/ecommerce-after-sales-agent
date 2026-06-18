from ecommerce_agent.graph import run_case


def test_durability_issue_is_medium_risk():
    result = run_case("这个水杯看上去很实用，其实并不耐用，但是外观还不错", use_llm=False)
    final = result["final_result"]
    assert final["risk"]["level"] == "中风险"
    assert any(item["category"] == "商品质量问题" for item in final["issue_categories"])


def test_price_rights_issue_is_medium_risk():
    result = run_case("价格比我买的时候便宜了好多，感觉被坑了", use_llm=False)
    final = result["final_result"]
    assert final["risk"]["level"] == "中风险"
    assert any(item["category"] == "价格与权益问题" for item in final["issue_categories"])


def test_positive_quality_review_stays_low_risk():
    result = run_case("衣服质量很好，尺码也合适，下次还会买", use_llm=False)
    final = result["final_result"]
    assert final["risk"]["level"] == "低风险"


def test_quality_failure_has_explainable_rule_trace():
    result = run_case("这个水杯看上去很实用，其实并不耐用，但是外观还不错", use_llm=False)
    risk = result["final_result"]["risk"]
    rule_ids = {rule["rule_id"] for rule in risk["triggered_rules"]}
    assert risk["level"] == "中风险"
    assert risk["priority"] == "P2"
    assert "R6_QUALITY_OR_DURABILITY" in rule_ids
    assert any("质量/耐用性" in reason for reason in risk["reasons"])


def test_safety_issue_is_high_risk():
    result = run_case("这个充电器用了两天就冒烟，感觉有安全隐患，我要投诉", use_llm=False)
    risk = result["final_result"]["risk"]
    rule_ids = {rule["rule_id"] for rule in risk["triggered_rules"]}
    assert risk["level"] == "高风险"
    assert risk["priority"] == "P1"
    assert risk["should_escalate"] is True
    assert "R1_SAFETY" in rule_ids
    assert "R2_COMPLAINT" in rule_ids


def test_mild_logistics_is_not_high_risk():
    result = run_case("物流有点慢，但是东西还可以", use_llm=False)
    risk = result["final_result"]["risk"]
    assert risk["level"] in {"低风险", "中风险"}
    assert risk["level"] != "高风险"
