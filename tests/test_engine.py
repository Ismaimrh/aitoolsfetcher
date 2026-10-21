"""Tests for the fetch engine, with fake sources and a temporary cache."""

from __future__ import annotations

import pytest

from aitoolsfetcher.classifiers.base import BaseClassifier, ClassificationResult
from aitoolsfetcher.classifiers.keyword import KeywordClassifier
from aitoolsfetcher.config import Settings
from aitoolsfetcher.core import engine as engine_module
from aitoolsfetcher.core.cache import ToolCache
from aitoolsfetcher.core.engine import AIToolsFetcher
from aitoolsfetcher.core.models import AICategory, AITool, SearchFilters
from aitoolsfetcher.sources.base import BaseSource


class FakeSource(BaseSource):
    """Source that returns canned tools, or raises, without any network access."""

    def __init__(
        self,
        name: str,
        tools: list[AITool] | None = None,
        error: Exception | None = None,
    ) -> None:
        super().__init__()
        self.name = name
        self._tools = tools or []
        self._error = error
        self.search_calls: list[SearchFilters] = []
        self.trending_calls: list[int] = []
        self.closed = False

    async def search(self, filters: SearchFilters) -> list[AITool]:
        self.search_calls.append(filters)
        if self._error:
            raise self._error
        return list(self._tools)

    async def trending(self, limit: int = 20) -> list[AITool]:
        self.trending_calls.append(limit)
        if self._error:
            raise self._error
        return list(self._tools)

    async def close(self) -> None:
        self.closed = True


class FixedClassifier(BaseClassifier):
    """Classifier that returns a preset result per tool name."""

    def __init__(self, results: dict[str, ClassificationResult]) -> None:
        self._results = results

    async def classify(self, tool: AITool) -> ClassificationResult:
        return self._results[tool.name]


def _result(is_ai: bool, score: float, category: AICategory = AICategory.OTHER):
    return ClassificationResult(is_ai, category, score, "")


@pytest.fixture
def make_fetcher(tmp_path):
    """Build a fetcher with injected sources and classifier and a temp cache."""
    created: list[AIToolsFetcher] = []

    def _make(sources, classifier=None, ttl_days=90):
        fetcher = AIToolsFetcher(
            settings=Settings(cache_ttl_days=ttl_days),
            sources=sources,
            classifier=classifier or KeywordClassifier(),
            cache=ToolCache(db_path=str(tmp_path / "cache.db")),
        )
        created.append(fetcher)
        return fetcher

    return _make


class TestSearch:
    async def test_merges_sources_and_sorts_by_relevance(self, make_fetcher, sample_tools):
        a = FakeSource("a", sample_tools[:1])
        b = FakeSource("b", sample_tools[1:])
        classifier = FixedClassifier(
            {
                "VectorForge": _result(True, 0.6, AICategory.INFRA),
                "PromptBench": _result(True, 0.9, AICategory.DEV_TOOL),
                "WebFramework": _result(False, 0.05),
            }
        )
        async with make_fetcher({"a": a, "b": b}, classifier) as fetcher:
            result = await fetcher.search("llm", limit=10)

        assert [t.name for t in result.tools] == ["PromptBench", "VectorForge", "WebFramework"]
        assert result.tools[0].category is AICategory.DEV_TOOL
        assert result.tools[0].ai_relevance_score == 0.9
        assert result.total_fetched == 3
        assert result.errors == {}

    async def test_filters_by_min_relevance_and_limit(self, make_fetcher, sample_tools):
        classifier = FixedClassifier(
            {
                "VectorForge": _result(True, 0.6),
                "PromptBench": _result(True, 0.9),
                "WebFramework": _result(False, 0.05),
            }
        )
        async with make_fetcher({"a": FakeSource("a", sample_tools)}, classifier) as fetcher:
            result = await fetcher.search("x", min_relevance=0.5, limit=1)
        assert [t.name for t in result.tools] == ["PromptBench"]

    async def test_passes_filters_to_sources(self, make_fetcher):
        source = FakeSource("a")
        async with make_fetcher({"a": source}) as fetcher:
            await fetcher.search("agents", limit=7, min_stars=50, language="python")

        [filters] = source.search_calls
        assert filters.query == "agents"
        assert filters.limit == 7
        assert filters.min_stars == 50
        assert filters.language == "python"

    async def test_only_requested_sources_are_queried(self, make_fetcher):
        a, b = FakeSource("a"), FakeSource("b")
        async with make_fetcher({"a": a, "b": b}) as fetcher:
            await fetcher.search("x", sources=["b", "unknown"])
        assert a.search_calls == []
        assert len(b.search_calls) == 1

    async def test_unknown_sources_only_returns_empty_result(self, make_fetcher):
        async with make_fetcher({"a": FakeSource("a")}) as fetcher:
            result = await fetcher.search("x", sources=["nope"])
        assert result.tools == []
        assert result.total_fetched == 0

    async def test_failing_source_is_reported_and_others_still_return(
        self, make_fetcher, sample_tool
    ):
        good = FakeSource("good", [sample_tool])
        bad = FakeSource("bad", error=RuntimeError("boom"))
        async with make_fetcher({"good": good, "bad": bad}) as fetcher:
            result = await fetcher.search("x")

        assert [t.name for t in result.tools] == [sample_tool.name]
        assert result.errors == {"bad": "boom"}
        assert "errors: bad" in result.summary

    async def test_error_without_message_uses_exception_name(self, make_fetcher):
        async with make_fetcher({"bad": FakeSource("bad", error=TimeoutError())}) as fetcher:
            result = await fetcher.search("x")
        assert result.errors == {"bad": "TimeoutError"}

    async def test_tracks_new_versus_cached(self, make_fetcher, sample_tools):
        source = FakeSource("a", sample_tools[:2])
        async with make_fetcher({"a": source}) as fetcher:
            first = await fetcher.search("x")
            source._tools = sample_tools  # one more tool on the second run
            second = await fetcher.search("x")

        assert first.new_count == 2
        assert set(first.new_ids) == {t.id for t in sample_tools[:2]}
        assert second.new_count == 1
        assert second.new_ids == [sample_tools[2].id]
        assert second.cached_count == 2


class TestTrending:
    async def test_keeps_only_ai_tools_sorted_by_popularity(self, make_fetcher, sample_tools):
        classifier = FixedClassifier(
            {
                "VectorForge": _result(True, 0.6),
                "PromptBench": _result(True, 0.9),
                "WebFramework": _result(False, 0.05),
            }
        )
        source = FakeSource("a", sample_tools)
        async with make_fetcher({"a": source}, classifier) as fetcher:
            result = await fetcher.trending(limit=5)

        # VectorForge has the higher popularity score of the two AI tools
        assert [t.name for t in result.tools] == ["VectorForge", "PromptBench"]
        assert result.total_fetched == 3
        assert source.trending_calls == [5]

    async def test_reports_source_errors(self, make_fetcher):
        bad = FakeSource("bad", error=RuntimeError("down"))
        async with make_fetcher({"bad": bad}) as fetcher:
            result = await fetcher.trending()
        assert result.errors == {"bad": "down"}

    async def test_no_sources(self, make_fetcher):
        async with make_fetcher({}) as fetcher:
            result = await fetcher.trending()
        assert result.tools == []


class TestLifecycle:
    async def test_history_returns_cached_tools(self, make_fetcher, sample_tools):
        async with make_fetcher({"a": FakeSource("a", sample_tools)}) as fetcher:
            await fetcher.search("x")
            history = await fetcher.history(limit=10)
        assert {t.name for t in history} == {t.name for t in sample_tools}

    async def test_close_closes_sources(self, make_fetcher):
        source = FakeSource("a")
        async with make_fetcher({"a": source}) as fetcher:
            assert fetcher.available_sources == ["a"]
        assert source.closed

    async def test_prunes_old_entries_on_open(self, make_fetcher, tmp_path, sample_tool):
        async with ToolCache(db_path=str(tmp_path / "cache.db")) as cache:
            await cache.store([sample_tool])
            db = cache._ensure_open()
            await db.execute("UPDATE tools SET created_at = datetime('now', '-10 days')")
            await db.commit()

        async with make_fetcher({}, ttl_days=5) as fetcher:
            assert await fetcher.history() == []


class TestDefaults:
    def test_github_source_is_built_by_default(self):
        sources = engine_module._build_sources(Settings())
        assert list(sources) == ["github"]

    def test_github_can_be_disabled(self):
        from aitoolsfetcher.config import SourceConfig

        settings = Settings(sources={"github": SourceConfig(enabled=False)})
        assert engine_module._build_sources(settings) == {}

    def test_keyword_classifier_without_llm_key(self):
        assert isinstance(engine_module._build_classifier(Settings()), KeywordClassifier)

    def test_llm_classifier_when_key_and_package_available(self, monkeypatch):
        from aitoolsfetcher.classifiers.llm import LLMClassifier

        monkeypatch.setattr(LLMClassifier, "is_available", staticmethod(lambda provider: True))
        classifier = engine_module._build_classifier(
            Settings(llm_api_key="test-key", llm_provider="openai")
        )
        assert isinstance(classifier, LLMClassifier)

    def test_falls_back_to_keyword_when_provider_unavailable(self, caplog):
        classifier = engine_module._build_classifier(
            Settings(llm_api_key="test-key", llm_provider="not-a-provider")
        )
        assert isinstance(classifier, KeywordClassifier)
        assert "keyword classifier" in caplog.text
