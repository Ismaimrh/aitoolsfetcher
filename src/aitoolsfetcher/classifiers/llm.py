"""LLM-based AI tool classifier — optional, requires an API key."""

from __future__ import annotations

import importlib.util
import json
import logging
from typing import Any

from aitoolsfetcher.classifiers.base import BaseClassifier, ClassificationResult
from aitoolsfetcher.core.models import AICategory, AITool

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are an AI tool classifier. Given information about a software tool or repository, determine:
1. Whether it is an AI-related tool (true/false)
2. Its category (one of: agent, framework, model, dataset, infra, dev_tool, library, cli_tool, \
other)
3. Your confidence in that true/false answer, from 0.0 to 1.0
4. A one-line summary of what it does

Respond ONLY with valid JSON, for example:
{"is_ai_tool": true, "category": "agent", "confidence": 0.95, "summary": "Builds AI agents"}"""

_USER_TEMPLATE = """Tool: {name}
Author: {author}
Description: {description}
Topics: {topics}
Language: {language}
Platform: {platform}"""

_VALID_CATEGORIES = {c.value for c in AICategory}

#: Python package needed for each supported provider.
_PROVIDER_PACKAGES = {"anthropic": "anthropic", "openai": "openai"}


class LLMClassifier(BaseClassifier):
    """Classifier that asks an LLM (Anthropic or OpenAI) about each tool.

    Only used when an LLM API key is configured. A failed call never raises:
    the tool is returned as not AI-related with a score of 0.
    """

    def __init__(self, api_key: str, provider: str = "anthropic") -> None:
        """Create an LLM classifier.

        Args:
            api_key: Provider API key.
            provider: ``"anthropic"`` or ``"openai"``.

        Raises:
            ValueError: If the provider is not supported.
        """
        provider = provider.lower()
        if provider not in _PROVIDER_PACKAGES:
            raise ValueError(
                f"Unsupported LLM provider: {provider}. Use one of: {', '.join(_PROVIDER_PACKAGES)}"
            )
        self._api_key = api_key
        self._provider = provider
        self._client: Any = None

    @staticmethod
    def is_available(provider: str) -> bool:
        """Check whether the SDK for a provider is installed.

        Args:
            provider: Provider name.

        Returns:
            True if the provider is supported and its package can be imported.
        """
        package = _PROVIDER_PACKAGES.get(provider.lower())
        return package is not None and importlib.util.find_spec(package) is not None

    def _get_client(self) -> Any:
        """Create the provider SDK client once and reuse it."""
        if self._client is not None:
            return self._client
        try:
            if self._provider == "anthropic":
                import anthropic

                self._client = anthropic.AsyncAnthropic(api_key=self._api_key)
            else:
                import openai

                self._client = openai.AsyncOpenAI(api_key=self._api_key)
        except ImportError as e:
            raise ImportError(
                f"The {self._provider} package is required for the LLM classifier. "
                "Install it with: pip install 'aitoolsfetcher[llm]'"
            ) from e
        return self._client

    def _build_prompt(self, tool: AITool) -> str:
        """Build the user message describing one tool."""
        return _USER_TEMPLATE.format(
            name=tool.name,
            author=tool.author,
            description=tool.description[:300],
            topics=", ".join(tool.topics[:10]),
            language=tool.language or "unknown",
            platform=tool.source_platform,
        )

    async def classify(self, tool: AITool) -> ClassificationResult:
        """Classify one tool with the LLM.

        Args:
            tool: Tool to classify.

        Returns:
            The classification. On any API or parsing error, a "not AI" result
            with score 0 is returned and a warning is logged.
        """
        prompt = self._build_prompt(tool)

        try:
            raw = await self._call_llm(prompt)
        except Exception as e:  # one bad call must not stop the batch
            logger.warning("LLM classification failed for %s: %s", tool.name, e)
            return _fallback(tool)
        return self._parse_response(raw, tool)

    async def _call_llm(self, prompt: str) -> str:
        """Send the prompt to the provider and return the raw text reply."""
        client = self._get_client()
        if self._provider == "anthropic":
            response = await client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=200,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            return str(response.content[0].text)

        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=200,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return str(response.choices[0].message.content or "")

    @staticmethod
    def _parse_response(raw: str, tool: AITool) -> ClassificationResult:
        """Parse the LLM's JSON reply into a ClassificationResult.

        The relevance score is the probability that the tool is AI-related:
        the model's confidence when it says yes, and one minus that confidence
        when it says no.
        """
        raw = raw.strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1]) if len(lines) > 2 else raw

        try:
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError("expected a JSON object")
            confidence = min(max(float(data.get("confidence", 0.0)), 0.0), 1.0)
        except (ValueError, TypeError):
            logger.warning("Could not parse LLM response: %s", raw[:200])
            return _fallback(tool)

        is_ai = bool(data.get("is_ai_tool", False))
        cat_str = str(data.get("category", "other")).lower()
        category = AICategory(cat_str) if cat_str in _VALID_CATEGORIES else AICategory.OTHER
        relevance = confidence if is_ai else 1.0 - confidence
        summary = str(data.get("summary", tool.description[:120]))

        return ClassificationResult(
            is_ai_tool=is_ai,
            category=category if is_ai else AICategory.OTHER,
            relevance_score=round(relevance, 3),
            summary=summary[:200],
        )


def _fallback(tool: AITool) -> ClassificationResult:
    """Result used when the LLM can't classify a tool."""
    return ClassificationResult(
        is_ai_tool=False,
        category=AICategory.OTHER,
        relevance_score=0.0,
        summary=tool.description[:120],
    )
