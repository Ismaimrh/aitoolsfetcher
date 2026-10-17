"""LLM-based AI tool classifier — optional, requires API key."""

from __future__ import annotations

import json
import logging
from typing import Optional

from aitoolsfetcher.classifiers.base import BaseClassifier, ClassificationResult
from aitoolsfetcher.core.models import AICategory, AITool

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are an AI tool classifier. Given information about a software tool/repository, determine:
1. Whether it is an AI-related tool (true/false)
2. Its category (one of: agent, framework, model, dataset, infra, dev_tool, library, cli_tool, other)
3. A confidence score from 0.0 to 1.0
4. A one-line summary of what it does

Respond ONLY with valid JSON:
{"is_ai_tool": true, "category": "agent", "confidence": 0.95, "summary": "Build autonomous AI agents with tool use"}"""

_USER_TEMPLATE = """Tool: {name}
Author: {author}
Description: {description}
Topics: {topics}
Language: {language}
Platform: {platform}"""

_VALID_CATEGORIES = {c.value for c in AICategory}


class LLMClassifier(BaseClassifier):
    """High-accuracy classifier using Claude or OpenAI API.

    Activated only when an LLM API key is provided. Falls back gracefully
    if the API call fails.
    """

    def __init__(self, api_key: str, provider: str = "anthropic"):
        self._api_key = api_key
        self._provider = provider.lower()
        self._client: Optional[object] = None

    async def _get_anthropic_client(self):
        try:
            import anthropic
            return anthropic.AsyncAnthropic(api_key=self._api_key)
        except ImportError:
            raise ImportError(
                "anthropic package required for LLM classifier. "
                "Install with: pip install aitoolsfetcher[llm]"
            )

    async def _get_openai_client(self):
        try:
            import openai
            return openai.AsyncOpenAI(api_key=self._api_key)
        except ImportError:
            raise ImportError(
                "openai package required for LLM classifier. "
                "Install with: pip install aitoolsfetcher[llm]"
            )

    def _build_prompt(self, tool: AITool) -> str:
        return _USER_TEMPLATE.format(
            name=tool.name,
            author=tool.author,
            description=tool.description[:300],
            topics=", ".join(tool.topics[:10]),
            language=tool.language or "unknown",
            platform=tool.source_platform,
        )

    async def classify(self, tool: AITool) -> ClassificationResult:
        """Classify a tool using LLM API."""
        prompt = self._build_prompt(tool)

        try:
            raw = await self._call_llm(prompt)
            return self._parse_response(raw, tool)
        except Exception as e:
            logger.warning("LLM classification failed for %s: %s", tool.name, e)
            return ClassificationResult(
                is_ai_tool=False,
                category=AICategory.OTHER,
                relevance_score=0.0,
                summary=tool.description[:120],
            )

    async def classify_batch(self, tools: list[AITool]) -> list[ClassificationResult]:
        """Classify tools one by one. Could be optimized with batch API in future."""
        results = []
        for tool in tools:
            result = await self.classify(tool)
            results.append(result)
        return results

    async def _call_llm(self, prompt: str) -> str:
        """Call the LLM API and return raw text response."""
        if self._provider == "anthropic":
            client = await self._get_anthropic_client()
            response = await client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=200,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        elif self._provider == "openai":
            client = await self._get_openai_client()
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=200,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            return response.choices[0].message.content or ""
        else:
            raise ValueError(f"Unsupported LLM provider: {self._provider}")

    @staticmethod
    def _parse_response(raw: str, tool: AITool) -> ClassificationResult:
        """Parse LLM JSON response into ClassificationResult."""
        # Extract JSON from response (handles markdown code blocks)
        raw = raw.strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1]) if len(lines) > 2 else raw

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM response as JSON: %s", raw[:200])
            return ClassificationResult(
                is_ai_tool=False,
                category=AICategory.OTHER,
                relevance_score=0.0,
                summary=tool.description[:120],
            )

        is_ai = bool(data.get("is_ai_tool", False))
        cat_str = str(data.get("category", "other")).lower()
        category = AICategory(cat_str) if cat_str in _VALID_CATEGORIES else AICategory.OTHER
        confidence = float(data.get("confidence", 0.0))
        confidence = min(max(confidence, 0.0), 1.0)
        summary = str(data.get("summary", tool.description[:120]))

        return ClassificationResult(
            is_ai_tool=is_ai,
            category=category,
            relevance_score=round(confidence, 3),
            summary=summary[:200],
        )
