from types import SimpleNamespace

from ecommerce_agent.agents import sentiment_agent
from ecommerce_agent.agents.sentiment_agent import analyze_sentiment
from ecommerce_agent.llm_client import NullLLMClient


def test_english_sentiment_prefers_lalun(monkeypatch):
    class FakeLALUNAdapter:
        def analyze_english(self, text):
            return {
                "enabled": True,
                "dataset": "14res",
                "triplets": [
                    {
                        "aspect": "service",
                        "opinion": "slow",
                        "sentiment": "NEG",
                        "evidence": text,
                    }
                ],
            }

    monkeypatch.setattr(sentiment_agent, "LALUNAdapter", FakeLALUNAdapter)
    monkeypatch.setattr(
        sentiment_agent,
        "get_settings",
        lambda: SimpleNamespace(use_lalun_english=True),
    )

    result = analyze_sentiment("The service was slow .", llm=NullLLMClient())

    assert result["engine"] == "lalun"
    assert result["triplets"][0]["aspect"] == "service"
    assert result["triplets"][0]["sentiment"] == "NEG"


def test_chinese_sentiment_does_not_use_lalun(monkeypatch):
    class BrokenLALUNAdapter:
        def analyze_english(self, text):  # pragma: no cover - should not be called
            raise AssertionError("Chinese text should not call LALUN")

    monkeypatch.setattr(sentiment_agent, "LALUNAdapter", BrokenLALUNAdapter)
    monkeypatch.setattr(
        sentiment_agent,
        "get_settings",
        lambda: SimpleNamespace(use_lalun_english=True),
    )

    result = analyze_sentiment("物流太慢了", llm=NullLLMClient())

    assert result["engine"] == "rule_fallback"
