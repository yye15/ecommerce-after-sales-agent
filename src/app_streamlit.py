"""Streamlit UI for the e-commerce customer service agent."""

from __future__ import annotations

import csv
import io
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ecommerce_agent.agents.operation_agent import build_operation_report
from ecommerce_agent.graph import run_case
from ecommerce_agent.lalun_adapter import LALUNAdapter, inspect_lalun


st.set_page_config(page_title="电商客服与售后运营 Agent", layout="wide")

st.title("基于 LangGraph 与细粒度情感分析的电商客服 Agent")
st.caption("输入评论或售后对话，自动生成情感三元组、问题分类、风险判断、售后策略和客服回复。")

with st.sidebar:
    st.header("运行设置")
    use_llm = st.toggle("调用 DeepSeek API", value=True)
    use_lalun_zh = st.toggle("展示 LALUN 中文实验结果", value=True)
    st.write("关闭 DeepSeek 后会使用规则兜底，适合离线演示。LALUN 中文模型目前作为实验对照，不影响主流程。")
    st.divider()
    if st.button("检查 LALUN 状态"):
        st.json(inspect_lalun())

tab_single, tab_batch = st.tabs(["单条分析", "批量运营分析"])

with tab_single:
    default_text = "这个耳机音质不错，但是物流太慢，客服也一直不回复，我有点想投诉。"
    text = st.text_area("客户评论/售后对话", value=default_text, height=140)
    if st.button("开始分析", type="primary"):
        with st.spinner("Agent 正在分析..."):
            result = run_case(text, use_llm=use_llm)
            final = result.get("final_result", result)

        c1, c2, c3 = st.columns(3)
        c1.metric("风险等级", final["risk"]["level"])
        c2.metric("风险分数", final["risk"]["score"])
        c3.metric("优先级", final["risk"].get("priority", final["strategy"]["priority"]))

        st.subheader("风险判断依据")
        for reason in final["risk"].get("reasons", []):
            st.markdown(f"- {reason}")
        triggered_rules = final["risk"].get("triggered_rules", [])
        if triggered_rules:
            with st.expander("命中的业务规则"):
                st.dataframe(triggered_rules, use_container_width=True)

        st.subheader("主流程情感三元组")
        st.dataframe(final["sentiment"]["triplets"], use_container_width=True)

        if use_lalun_zh:
            st.subheader("LALUN 中文实验结果")
            with st.spinner("LALUN 中文微调模型正在抽取三元组..."):
                lalun_result = LALUNAdapter().analyze_chinese_experimental(text)
            if lalun_result.get("enabled"):
                triplets = lalun_result.get("triplets", [])
                if triplets:
                    st.dataframe(triplets, use_container_width=True)
                else:
                    st.info("LALUN 本次没有抽取到三元组。该模型目前是实验版，主流程仍以 DeepSeek/规则结果为准。")
                with st.expander("LALUN 原始输出"):
                    st.json(lalun_result)
            else:
                st.warning(lalun_result.get("error", "LALUN 中文实验模型暂不可用。"))

        st.subheader("问题分类")
        st.dataframe(final["issue_categories"], use_container_width=True)

        st.subheader("售后策略")
        st.dataframe(final["strategy"]["actions"], use_container_width=True)

        policy_context = final.get("policy_context", {})
        st.subheader("RAG 检索到的售后政策")
        docs = policy_context.get("documents", [])
        if docs:
            st.caption(f"检索器：{policy_context.get('retriever', 'unknown')}")
            st.dataframe(
                [
                    {
                        "policy": doc.get("title"),
                        "source": doc.get("source"),
                        "score": doc.get("score"),
                        "keywords": "、".join(doc.get("keywords", [])[:6]),
                    }
                    for doc in docs
                ],
                use_container_width=True,
            )
            with st.expander("政策原文"):
                for doc in docs:
                    st.markdown(f"#### {doc.get('title')}")
                    st.markdown(doc.get("content", ""))
        else:
            st.info("未检索到明确售后政策。")

        st.subheader("客服回复")
        st.success(final["reply"])

        with st.expander("完整 JSON"):
            st.json(final)

with tab_batch:
    st.write("上传 CSV，需要包含 `text`、`review` 或 `content` 列。")
    uploaded = st.file_uploader("上传评论 CSV", type=["csv"])
    if uploaded and st.button("生成运营报告"):
        content = uploaded.getvalue().decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(content))
        texts = [
            (row.get("text") or row.get("review") or row.get("content") or "").strip()
            for row in reader
        ]
        texts = [item for item in texts if item]
        with st.spinner("正在批量分析..."):
            results = [run_case(item, use_llm=use_llm) for item in texts]
            report = build_operation_report(results)

        st.subheader("运营报告")
        c1, c2 = st.columns(2)
        c1.metric("样本数量", report["total_cases"])
        c2.metric("高风险数量", report["risk_distribution"].get("高风险", 0))
        st.write("风险分布")
        st.json(report["risk_distribution"])
        st.write("高频问题")
        st.dataframe(report["top_issue_categories"], use_container_width=True)
        st.write("高频负面方面")
        st.dataframe(report["top_negative_aspects"], use_container_width=True)
        st.write("运营建议")
        for item in report["recommendations"]:
            st.markdown(f"- {item}")
