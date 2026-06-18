"""DeepSeek/OpenAI-compatible LLM client wrapper."""

from __future__ import annotations

from typing import Any

from .config import Settings, get_settings
from .utils.json_tools import safe_json_loads


class LLMClient:
    """Small wrapper around LangChain ChatOpenAI with graceful fallback."""

    def __init__(self, settings: Settings | None = None, temperature: float | None = None):
        self.settings = settings or get_settings()
        self.temperature = (
            temperature
            if temperature is not None
            else self.settings.default_temperature
        )
        self._model = None
        self._import_error: Exception | None = None

    @property
    def available(self) -> bool:
        return self.settings.has_llm and self._load_model() is not None

    def _load_model(self):
        if self._model is not None:
            return self._model
        if not self.settings.has_llm:
            return None

        try:
            from langchain_openai import ChatOpenAI
        except Exception as exc:  # pragma: no cover - depends on local env
            self._import_error = exc
            return None

        self._model = ChatOpenAI(
            model=self.settings.llm_model_id,
            api_key=self.settings.llm_api_key,
            base_url=self.settings.llm_base_url,
            timeout=self.settings.llm_timeout,
            temperature=self.temperature,
        )
        return self._model

    def invoke_text(self, system_prompt: str, user_prompt: str) -> str | None:
        model = self._load_model()
        if model is None:
            return None

        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            response = model.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ]
            )
            return response.content
        except Exception:
            return None

    def invoke_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any] | list[Any] | None:
        text = self.invoke_text(system_prompt, user_prompt)
        if text is None:
            return None
        return safe_json_loads(text)


class NullLLMClient:
    """LLM-compatible stub used for demos/tests without network calls."""

    available = False

    def invoke_text(self, system_prompt: str, user_prompt: str) -> str | None:
        return None

    def invoke_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any] | list[Any] | None:
        return None
