from scripts.evaluate_golden_set import split_expected_categories, summarize


def test_split_expected_categories():
    assert split_expected_categories("商品质量问题; 客服响应问题") == {
        "商品质量问题",
        "客服响应问题",
    }


def test_summarize_counts_under_risk():
    summary = summarize(
        [
            {
                "category_match": 1,
                "risk_match": 1,
                "risk_gap": 0,
                "expected_risk_level": "高风险",
                "actual_risk_level": "高风险",
                "priority_match": 1,
            },
            {
                "category_match": 0,
                "risk_match": 0,
                "risk_gap": -1,
                "expected_risk_level": "高风险",
                "actual_risk_level": "中风险",
                "priority_match": 0,
            },
        ]
    )
    assert summary["total"] == 2
    assert summary["category_accuracy"] == 0.5
    assert summary["risk_accuracy"] == 0.5
    assert summary["under_risk_count"] == 1
    assert summary["high_risk_recall"] == 0.5
