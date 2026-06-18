from scripts.create_dialogue_eval_candidates import build_dialogue


def test_positive_dialogue_template_does_not_inject_after_sales_word():
    dialogue = build_dialogue("质量很好，下次还会买", "衣服", "1")
    assert "售后" not in dialogue
    assert "用户：" in dialogue
    assert "坐席：" in dialogue


def test_negative_dialogue_template_keeps_real_review():
    review = "才用几天就坏了，想换货"
    dialogue = build_dialogue(review, "耳机", "0")
    assert review in dialogue
    assert "退换货" in dialogue
