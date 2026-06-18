from scripts.prepare_jddc_eval_candidates import extract_dialogue, is_after_sales_dialogue


def test_extract_jddc_messages_uses_neutral_service_role():
    text, turns = extract_dialogue(
        {
            "messages": [
                {"role": "用户", "content": "我想退货，客服没人处理。"},
                {"role": "客服", "content": "您好，请提供订单号。"},
            ]
        }
    )
    assert turns == 2
    assert "用户：我想退货" in text
    assert "坐席：您好" in text
    assert "客服：您好" not in text


def test_after_sales_filter_keeps_refund_dialogue():
    text = "用户：我要退款\n坐席：请提供订单号"
    assert is_after_sales_dialogue(text, turns_count=2, min_turns=2)


def test_after_sales_filter_drops_normal_consultation():
    text = "用户：有没有粉色\n坐席：暂时缺货"
    assert not is_after_sales_dialogue(text, turns_count=2, min_turns=2)
