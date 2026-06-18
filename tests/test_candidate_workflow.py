from scripts.promote_reviewed_candidates import convert_row, is_reviewed


def test_is_reviewed_requires_human_labels():
    assert not is_reviewed({"human_expected_categories": "商品质量问题"})
    assert is_reviewed(
        {
            "human_expected_categories": "商品质量问题",
            "human_expected_risk_level": "中风险",
            "human_expected_priority": "P2",
        }
    )


def test_convert_reviewed_candidate_row():
    row = {
        "candidate_id": "C001",
        "review_text": "质量不好",
        "product_category": "衣服",
        "human_expected_categories": "商品质量问题",
        "human_expected_risk_level": "中风险",
        "human_expected_priority": "P2",
        "human_rule_reason": "质量负面",
    }
    converted = convert_row(row, 1)
    assert converted["case_id"] == "R001"
    assert converted["expected_categories"] == "商品质量问题"
    assert converted["expected_risk_level"] == "中风险"
