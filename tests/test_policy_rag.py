from ecommerce_agent.graph import run_case
from ecommerce_agent.rag.policy_rag import retrieve_policy_context


def test_policy_rag_retrieves_quality_policy():
    context = retrieve_policy_context(
        "耳机用了三天听筒就坏了，我要退货",
        [{"category": "商品质量问题", "secondary_reasons": ["核心故障"], "matched_keywords": ["听筒", "坏"]}],
        {"level": "高风险"},
    )

    titles = {item["title"] for item in context["documents"]}
    assert any("质量问题退换货" in title for title in titles)
    assert context["retriever"] == "local_keyword_policy_rag_v1"


def test_agent_final_result_contains_policy_context():
    final = run_case("耳机用了三天听筒就坏了，我要退货", use_llm=False)["final_result"]

    assert "policy_context" in final
    assert final["policy_context"]["documents"]
    assert "P002 质量问题退换货" in final["reply"]
