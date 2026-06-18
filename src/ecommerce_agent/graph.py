"""LangGraph workflow for the e-commerce customer service agent."""

from __future__ import annotations

from typing import Literal

from .agents.after_sales_agent import propose_strategy
from .agents.classification_agent import classify_issues
from .agents.input_agent import normalize_input
from .agents.reply_agent import generate_reply
from .agents.risk_agent import assess_risk
from .agents.sentiment_agent import analyze_sentiment
from .llm_client import LLMClient, NullLLMClient
from .rag.policy_rag import retrieve_policy_context
from .schemas import CustomerSupportState


def build_graph(llm: LLMClient | NullLLMClient | None = None):
    """Build and compile the LangGraph application."""
    try:
        from langgraph.graph import END, START, StateGraph
    except Exception as exc:  # pragma: no cover - depends on local env
        raise RuntimeError("缺少 langgraph 依赖，请先安装 requirements。") from exc

    llm_client = llm or LLMClient()
    graph = StateGraph(CustomerSupportState)

    def input_node(state: CustomerSupportState) -> dict:
        return normalize_input(state.get("raw_text", ""))

    def sentiment_node(state: CustomerSupportState) -> dict:
        text = state.get("cleaned_text", "")
        return {"sentiment": analyze_sentiment(text, llm_client)}

    def classification_node(state: CustomerSupportState) -> dict:
        text = state.get("cleaned_text", "")
        return {"issue_categories": classify_issues(text, state.get("sentiment", {}))}

    def risk_node(state: CustomerSupportState) -> dict:
        text = state.get("cleaned_text", "")
        return {
            "risk": assess_risk(
                text,
                state.get("sentiment", {}),
                state.get("issue_categories", []),
            )
        }

    def route_by_risk(state: CustomerSupportState) -> Literal["human_escalation", "strategy"]:
        if state.get("risk", {}).get("should_escalate"):
            return "human_escalation"
        return "strategy"

    def policy_retrieval_node(state: CustomerSupportState) -> dict:
        text = state.get("cleaned_text", "")
        return {
            "policy_context": retrieve_policy_context(
                text,
                state.get("issue_categories", []),
                state.get("risk", {}),
            )
        }

    def escalation_node(state: CustomerSupportState) -> dict:
        reasons = "；".join(state.get("risk", {}).get("reasons", []))
        return {
            "escalation_required": True,
            "escalation_note": f"建议客服主管优先介入。原因：{reasons}",
        }

    def strategy_node(state: CustomerSupportState) -> dict:
        text = state.get("cleaned_text", "")
        strategy = propose_strategy(
            text,
            state.get("sentiment", {}),
            state.get("issue_categories", []),
            state.get("risk", {}),
            state.get("escalation_note", ""),
        )
        return {"strategy": strategy}

    def reply_node(state: CustomerSupportState) -> dict:
        text = state.get("cleaned_text", "")
        reply = generate_reply(
            text,
            state.get("sentiment", {}),
            state.get("issue_categories", []),
            state.get("risk", {}),
            state.get("strategy", {}),
            state.get("policy_context", {}),
            llm_client,
        )
        return {"reply": reply}

    def finalize_node(state: CustomerSupportState) -> dict:
        final_result = {
            "input": state.get("cleaned_text", ""),
            "customer_intent": state.get("customer_intent", ""),
            "sentiment": state.get("sentiment", {}),
            "issue_categories": state.get("issue_categories", []),
            "risk": state.get("risk", {}),
            "policy_context": state.get("policy_context", {}),
            "strategy": state.get("strategy", {}),
            "reply": state.get("reply", ""),
            "escalation_required": state.get("escalation_required", False),
            "escalation_note": state.get("escalation_note", ""),
        }
        return {"final_result": final_result}

    graph.add_node("input_understanding", input_node)
    graph.add_node("sentiment_analysis", sentiment_node)
    graph.add_node("issue_classification", classification_node)
    graph.add_node("risk_assessment", risk_node)
    graph.add_node("policy_retrieval", policy_retrieval_node)
    graph.add_node("human_escalation", escalation_node)
    graph.add_node("after_sales_strategy", strategy_node)
    graph.add_node("reply_generation", reply_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "input_understanding")
    graph.add_edge("input_understanding", "sentiment_analysis")
    graph.add_edge("sentiment_analysis", "issue_classification")
    graph.add_edge("issue_classification", "risk_assessment")
    graph.add_edge("risk_assessment", "policy_retrieval")
    graph.add_conditional_edges(
        "policy_retrieval",
        route_by_risk,
        {
            "human_escalation": "human_escalation",
            "strategy": "after_sales_strategy",
        },
    )
    graph.add_edge("human_escalation", "after_sales_strategy")
    graph.add_edge("after_sales_strategy", "reply_generation")
    graph.add_edge("reply_generation", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile()


def run_case(text: str, use_llm: bool = True) -> dict:
    """Run one case. Falls back to sequential execution if LangGraph is unavailable."""
    llm = LLMClient() if use_llm else NullLLMClient()
    initial_state: CustomerSupportState = {"raw_text": text}
    try:
        app = build_graph(llm)
        return app.invoke(initial_state)
    except RuntimeError:
        return run_case_sequential(text, llm)


def run_case_sequential(text: str, llm: LLMClient | NullLLMClient | None = None) -> dict:
    """Sequential fallback with the same output shape as the graph."""
    llm_client = llm or LLMClient()
    state: CustomerSupportState = {"raw_text": text}
    state.update(normalize_input(text))
    state["sentiment"] = analyze_sentiment(state["cleaned_text"], llm_client)
    state["issue_categories"] = classify_issues(state["cleaned_text"], state["sentiment"])
    state["risk"] = assess_risk(state["cleaned_text"], state["sentiment"], state["issue_categories"])
    state["policy_context"] = retrieve_policy_context(
        state["cleaned_text"],
        state["issue_categories"],
        state["risk"],
    )
    if state["risk"].get("should_escalate"):
        state["escalation_required"] = True
        state["escalation_note"] = f"建议客服主管优先介入。原因：{'；'.join(state['risk'].get('reasons', []))}"
    state["strategy"] = propose_strategy(
        state["cleaned_text"],
        state["sentiment"],
        state["issue_categories"],
        state["risk"],
        state.get("escalation_note", ""),
    )
    state["reply"] = generate_reply(
        state["cleaned_text"],
        state["sentiment"],
        state["issue_categories"],
        state["risk"],
        state["strategy"],
        state["policy_context"],
        llm_client,
    )
    state["final_result"] = {
        "input": state.get("cleaned_text", ""),
        "customer_intent": state.get("customer_intent", ""),
        "sentiment": state.get("sentiment", {}),
        "issue_categories": state.get("issue_categories", []),
        "risk": state.get("risk", {}),
        "policy_context": state.get("policy_context", {}),
        "strategy": state.get("strategy", {}),
        "reply": state.get("reply", ""),
        "escalation_required": state.get("escalation_required", False),
        "escalation_note": state.get("escalation_note", ""),
    }
    return state
