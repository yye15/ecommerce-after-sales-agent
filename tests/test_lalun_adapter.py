import json
from pathlib import Path

from ecommerce_agent.lalun_adapter import LALUNAdapter


def test_lalun_adapter_parses_subprocess_json(monkeypatch):
    adapter = LALUNAdapter(lalun_root=Path("external/LALUN/delivery_105"))
    adapter.status["available"] = True

    class Completed:
        returncode = 0
        stdout = json.dumps(
            {
                "enabled": True,
                "engine": "lalun",
                "triplets": [{"aspect": "bread", "opinion": "top notch", "sentiment": "POS"}],
            }
        )
        stderr = ""

    def fake_run(*args, **kwargs):
        return Completed()

    monkeypatch.setattr("subprocess.run", fake_run)

    result = adapter.analyze_english("The bread is top notch as well .")
    assert result["enabled"] is True
    assert result["triplets"][0]["aspect"] == "bread"
    assert result["triplets"][0]["sentiment"] == "POS"


def test_lalun_adapter_reports_unavailable(monkeypatch):
    adapter = LALUNAdapter(lalun_root=Path("external/LALUN/delivery_105"))
    adapter.status["available"] = False

    result = adapter.analyze_english("The bread is top notch as well .")
    assert result["enabled"] is False
    assert "error" in result


def test_lalun_adapter_parses_chinese_subprocess_json(monkeypatch):
    adapter = LALUNAdapter(lalun_root=Path("external/LALUN/delivery_105"))
    adapter.status["chinese_available"] = True

    class Completed:
        returncode = 0
        stdout = json.dumps(
            {
                "enabled": True,
                "engine": "lalun_zh_experimental",
                "triplets": [{"aspect": "音质", "opinion": "不错", "sentiment": "POS"}],
            },
            ensure_ascii=False,
        )
        stderr = ""

    def fake_run(*args, **kwargs):
        return Completed()

    monkeypatch.setattr("subprocess.run", fake_run)

    result = adapter.analyze_chinese_experimental("这个耳机音质不错")
    assert result["enabled"] is True
    assert result["triplets"][0]["aspect"] == "音质"
    assert result["triplets"][0]["sentiment"] == "POS"


def test_lalun_adapter_reports_chinese_unavailable(monkeypatch):
    adapter = LALUNAdapter(lalun_root=Path("external/LALUN/delivery_105"))
    adapter.status["chinese_available"] = False

    result = adapter.analyze_chinese_experimental("这个耳机音质不错")
    assert result["enabled"] is False
    assert "error" in result
