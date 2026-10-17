"""Tests for keyword classifier."""

from __future__ import annotations

import pytest

from aitoolsfetcher.classifiers.keyword import KeywordClassifier
from aitoolsfetcher.core.models import AICategory, AITool


@pytest.fixture
def classifier():
    return KeywordClassifier()


class TestKeywordClassifier:
    async def test_classifies_ai_agent(self, classifier: KeywordClassifier):
        tool = AITool(
            name="AutoAgent",
            description="Build autonomous AI agents with tool use",
            source_platform="github",
            source_url="https://github.com/test/autoagent",
            topics=["ai-agent", "llm", "autonomous"],
            author="test",
        )
        result = await classifier.classify(tool)
        assert result.is_ai_tool
        assert result.category == AICategory.AGENT
        assert result.relevance_score > 0.3

    async def test_classifies_framework(self, classifier: KeywordClassifier):
        tool = AITool(
            name="DeepFrame",
            description="A deep learning framework built on pytorch",
            source_platform="github",
            source_url="https://github.com/test/deepframe",
            topics=["deep-learning", "pytorch", "framework"],
            author="test",
        )
        result = await classifier.classify(tool)
        assert result.is_ai_tool
        assert result.category == AICategory.FRAMEWORK

    async def test_classifies_infra(self, classifier: KeywordClassifier):
        tool = AITool(
            name="VecDB",
            description="Fast vector database for RAG and embedding storage",
            source_platform="github",
            source_url="https://github.com/test/vecdb",
            topics=["vector-database", "rag", "embedding"],
            author="test",
        )
        result = await classifier.classify(tool)
        assert result.is_ai_tool
        assert result.category == AICategory.INFRA

    async def test_rejects_non_ai_tool(self, classifier: KeywordClassifier):
        tool = AITool(
            name="WebServer",
            description="A fast HTTP web server written in Go",
            source_platform="github",
            source_url="https://github.com/test/webserver",
            topics=["web", "http", "server", "go"],
            author="test",
        )
        result = await classifier.classify(tool)
        assert not result.is_ai_tool
        assert result.relevance_score < 0.3

    async def test_penalizes_awesome_lists(self, classifier: KeywordClassifier):
        tool = AITool(
            name="awesome-ai-tools",
            description="A curated awesome-list of AI and machine learning tools",
            source_platform="github",
            source_url="https://github.com/test/awesome-ai",
            topics=["awesome-list", "ai", "machine-learning"],
            author="test",
        )
        result = await classifier.classify(tool)
        # Should still be AI-related but with lower score
        assert result.relevance_score < 0.6

    async def test_batch_classify(self, classifier: KeywordClassifier, sample_tools: list[AITool]):
        results = await classifier.classify_batch(sample_tools)
        assert len(results) == len(sample_tools)
        # VectorForge should be AI, WebFramework should not
        assert results[0].is_ai_tool  # VectorForge
        assert not results[2].is_ai_tool  # WebFramework
