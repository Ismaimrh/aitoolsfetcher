"""Tests for the LLM classifier. Provider SDKs are replaced by fakes: no network, no keys."""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from aitoolsfetcher.classifiers.llm import LLMClassifier
from aitoolsfetcher.core.models import AICategory


class _Recorder:
    """Async callable that records its kwargs and returns a canned response."""

    def __init__(self, response: Any = None, error: Exception | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.response = response
        self.error = error

    async def __call__(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


def _install_fake_anthropic(monkeypatch, text: str = "", error: Exception | None = None):
    response = types.SimpleNamespace(content=[types.SimpleNamespace(text=text)])
    create = _Recorder(response, error)
    created_with: list[dict[str, Any]] = []

    class AsyncAnthropic:
        def __init__(self, **kwargs: Any) -> None:
            created_with.append(kwargs)
            self.messages = types.SimpleNamespace(create=create)

    module = types.ModuleType("anthropic")
    module.AsyncAnthropic = AsyncAnthropic  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "anthropic", module)
    return create, created_with


def _install_fake_openai(monkeypatch, text: str | None = ""):
    message = types.SimpleNamespace(content=text)
    response = types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])
    create = _Recorder(response)

    class AsyncOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=create))

    module = types.ModuleType("openai")
    module.AsyncOpenAI = AsyncOpenAI  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", module)
    return create


class TestProviders:
    async def test_anthropic_request_and_parsing(self, monkeypatch, sample_tool):
        create, created_with = _install_fake_anthropic(
            monkeypatch,
            '{"is_ai_tool": true, "category": "agent", "confidence": 0.92, "summary": "Agents"}',
        )
        classifier = LLMClassifier(api_key="test-key", provider="anthropic")
        result = await classifier.classify(sample_tool)

        assert result.is_ai_tool is True
        assert result.category is AICategory.AGENT
        assert result.relevance_score == 0.92
        assert result.summary == "Agents"
        assert created_with == [{"api_key": "test-key"}]
        [call] = create.calls
        assert call["system"].startswith("You are an AI tool classifier")
        prompt = call["messages"][0]["content"]
        assert "Tool: AutoAgent" in prompt
        assert "Topics: ai-agent, llm, mcp, autonomous" in prompt

    async def test_openai_request_and_parsing(self, monkeypatch, sample_tool):
        create = _install_fake_openai(
            monkeypatch,
            '```json\n{"is_ai_tool": true, "category": "infra", "confidence": 0.7}\n```',
        )
        result = await LLMClassifier(api_key="k", provider="OpenAI").classify(sample_tool)

        assert result.category is AICategory.INFRA
        assert result.relevance_score == 0.7
        assert create.calls[0]["messages"][0]["role"] == "system"

    async def test_client_is_created_once(self, monkeypatch, sample_tools):
        _, created_with = _install_fake_anthropic(
            monkeypatch, '{"is_ai_tool": false, "confidence": 0.9}'
        )
        classifier = LLMClassifier(api_key="k")
        results = await classifier.classify_batch(sample_tools)
        assert len(results) == 3
        assert len(created_with) == 1

    async def test_api_error_returns_not_ai(self, monkeypatch, sample_tool):
        _install_fake_anthropic(monkeypatch, error=RuntimeError("401 unauthorized"))
        result = await LLMClassifier(api_key="bad").classify(sample_tool)
        assert result.is_ai_tool is False
        assert result.relevance_score == 0.0

    async def test_missing_package_returns_not_ai(self, monkeypatch, sample_tool):
        monkeypatch.setitem(sys.modules, "anthropic", None)  # makes the import fail
        result = await LLMClassifier(api_key="k").classify(sample_tool)
        assert result.relevance_score == 0.0

    def test_unsupported_provider(self):
        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            LLMClassifier(api_key="k", provider="gemini")

    def test_is_available(self, monkeypatch):
        assert LLMClassifier.is_available("nope") is False
        monkeypatch.setattr("importlib.util.find_spec", lambda name: object())
        assert LLMClassifier.is_available("anthropic") is True


class TestParseResponse:
    def parse(self, raw: str, tool):
        return LLMClassifier._parse_response(raw, tool)

    def test_not_ai_with_high_confidence_gets_low_relevance(self, sample_tool):
        result = self.parse(
            '{"is_ai_tool": false, "category": "agent", "confidence": 0.9}', sample_tool
        )
        assert result.is_ai_tool is False
        assert result.relevance_score == 0.1
        assert result.category is AICategory.OTHER

    def test_unknown_category_becomes_other(self, sample_tool):
        result = self.parse(
            '{"is_ai_tool": true, "category": "robot", "confidence": 1}', sample_tool
        )
        assert result.category is AICategory.OTHER

    def test_confidence_is_clamped(self, sample_tool):
        assert (
            self.parse('{"is_ai_tool": true, "confidence": 7}', sample_tool).relevance_score == 1.0
        )
        assert (
            self.parse('{"is_ai_tool": true, "confidence": -1}', sample_tool).relevance_score == 0.0
        )

    @pytest.mark.parametrize(
        "raw",
        ["not json", "[1, 2]", '{"is_ai_tool": true, "confidence": "high"}', ""],
    )
    def test_malformed_replies_fall_back(self, raw, sample_tool):
        result = self.parse(raw, sample_tool)
        assert result.is_ai_tool is False
        assert result.relevance_score == 0.0
        assert result.summary == sample_tool.description[:120]

    def test_summary_defaults_to_description(self, sample_tool):
        result = self.parse('{"is_ai_tool": true, "confidence": 0.5}', sample_tool)
        assert result.summary == sample_tool.description[:120]
