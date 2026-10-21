"""Tests for core data models."""

from __future__ import annotations

from aitoolsfetcher.core.models import AICategory, AITool, FetchResult, SearchFilters


class TestAITool:
    def test_id_is_deterministic(self, sample_tool: AITool):
        """Same name/platform/author always produces the same ID."""
        tool2 = sample_tool.model_copy()
        assert sample_tool.id == tool2.id

    def test_id_differs_across_platforms(self):
        """Same tool on different platforms gets different IDs."""
        base = {"name": "TestTool", "description": "test", "author": "user"}
        t1 = AITool(source_platform="github", source_url="https://a.com", **base)
        t2 = AITool(source_platform="reddit", source_url="https://b.com", **base)
        assert t1.id != t2.id

    def test_source_code_or_source_prefers_code(self, sample_tool: AITool):
        assert sample_tool.source_code_or_source == sample_tool.source_code_url

    def test_source_code_or_source_fallback(self):
        tool = AITool(
            name="Test",
            description="test",
            source_platform="reddit",
            source_url="https://reddit.com/r/test",
        )
        assert tool.source_code_or_source == "https://reddit.com/r/test"

    def test_default_values(self):
        tool = AITool(
            name="Min",
            description="Minimal tool",
            source_platform="github",
            source_url="https://github.com/x/y",
        )
        assert tool.category == AICategory.OTHER
        assert tool.stars is None
        assert tool.topics == []
        assert tool.ai_relevance_score == 0.0


class TestSearchFilters:
    def test_defaults(self):
        f = SearchFilters()
        assert f.query == ""
        assert f.limit == 20
        assert f.min_relevance == 0.5

    def test_custom_values(self):
        f = SearchFilters(query="agents", limit=10, min_stars=100)
        assert f.query == "agents"
        assert f.min_stars == 100


class TestFetchResult:
    def test_summary(self):
        r = FetchResult(new_count=5, cached_count=3, total_fetched=15)
        assert "5 new AI tools" in r.summary
        assert "15 fetched" in r.summary

    def test_summary_with_errors(self):
        r = FetchResult(new_count=2, total_fetched=5, errors={"reddit": "timeout"})
        assert "reddit" in r.summary
