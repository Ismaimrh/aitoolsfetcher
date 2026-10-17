"""Core fetch engine — orchestrates sources, classification, caching, and delivery."""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from aitoolsfetcher.classifiers.base import BaseClassifier
from aitoolsfetcher.classifiers.keyword import KeywordClassifier
from aitoolsfetcher.config import Settings, load_settings
from aitoolsfetcher.core.cache import ToolCache
from aitoolsfetcher.core.models import AITool, FetchResult, SearchFilters
from aitoolsfetcher.destinations.base import BaseDestination
from aitoolsfetcher.sources.base import BaseSource

logger = logging.getLogger(__name__)


def _build_sources(settings: Settings) -> dict[str, BaseSource]:
    """Build enabled source adapters from settings."""
    sources: dict[str, BaseSource] = {}

    if settings.is_source_enabled("github"):
        from aitoolsfetcher.sources.github import GitHubSource
        sources["github"] = GitHubSource(token=settings.github_token)

    # Future sources will be added here as they're implemented
    # if settings.is_source_enabled("huggingface"):
    #     from aitoolsfetcher.sources.huggingface import HuggingFaceSource
    #     sources["huggingface"] = HuggingFaceSource(token=settings.huggingface_token)

    return sources


def _build_classifier(settings: Settings) -> BaseClassifier:
    """Build classifier — LLM if key available, otherwise keyword."""
    if settings.has_llm_key:
        try:
            from aitoolsfetcher.classifiers.llm import LLMClassifier
            return LLMClassifier(
                api_key=settings.llm_api_key,
                provider=settings.llm_provider,
            )
        except ImportError:
            logger.warning("LLM packages not installed, falling back to keyword classifier")
    return KeywordClassifier()


class AIToolsFetcher:
    """Main entry point for discovering AI tools.

    Usage:
        async with AIToolsFetcher() as fetcher:
            result = await fetcher.search("AI agents")
            for tool in result.tools:
                print(tool.name, tool.source_code_or_source)
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        sources: Optional[dict[str, BaseSource]] = None,
        classifier: Optional[BaseClassifier] = None,
        cache: Optional[ToolCache] = None,
    ):
        self._settings = settings or load_settings()
        self._sources = sources or _build_sources(self._settings)
        self._classifier = classifier or _build_classifier(self._settings)
        self._cache = cache or ToolCache()
        self._initialized = False

    async def _ensure_init(self) -> None:
        if not self._initialized:
            await self._cache.open()
            self._initialized = True

    async def close(self) -> None:
        """Release all resources."""
        for source in self._sources.values():
            await source.close()
        await self._cache.close()
        self._initialized = False

    async def __aenter__(self) -> AIToolsFetcher:
        await self._ensure_init()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()

    @property
    def available_sources(self) -> list[str]:
        return list(self._sources.keys())

    async def search(
        self,
        query: str = "",
        sources: Optional[list[str]] = None,
        limit: int = 20,
        min_relevance: float = 0.0,
        min_stars: int = 0,
        **kwargs,
    ) -> FetchResult:
        """On-demand search for AI tools.

        Args:
            query: Search query string.
            sources: List of source names to search. None = all enabled.
            limit: Maximum results to return.
            min_relevance: Minimum AI relevance score (0-1).
            min_stars: Minimum stars/popularity threshold.

        Returns:
            FetchResult with discovered tools.
        """
        await self._ensure_init()

        filters = SearchFilters(
            query=query,
            sources=sources or [],
            limit=limit,
            min_relevance=min_relevance,
            min_stars=min_stars,
            **kwargs,
        )

        # Determine which sources to query
        active_sources = self._resolve_sources(filters.sources)
        if not active_sources:
            return FetchResult()

        # Fetch from all sources concurrently
        raw_tools, errors = await self._fetch_all(active_sources, filters)

        # Classify and filter
        classified = await self._classify_tools(raw_tools)
        filtered = [t for t in classified if t.ai_relevance_score >= min_relevance]

        # Sort by relevance, then popularity
        filtered.sort(key=lambda t: (t.ai_relevance_score, t.popularity_score), reverse=True)
        filtered = filtered[:limit]

        # Cache and compute new vs cached counts
        new_tools = await self._cache.get_new(filtered)
        new_count = await self._cache.store(filtered)

        return FetchResult(
            tools=filtered,
            new_count=new_count,
            cached_count=len(filtered) - len(new_tools),
            total_fetched=len(raw_tools),
            errors=errors,
        )

    async def trending(
        self,
        sources: Optional[list[str]] = None,
        limit: int = 20,
    ) -> FetchResult:
        """Fetch currently trending AI tools."""
        await self._ensure_init()

        active_sources = self._resolve_sources(sources or [])
        if not active_sources:
            return FetchResult()

        raw_tools: list[AITool] = []
        errors: dict[str, str] = {}

        tasks = {
            name: asyncio.create_task(source.trending(limit=limit))
            for name, source in active_sources.items()
        }

        for name, task in tasks.items():
            try:
                result = await task
                raw_tools.extend(result)
            except Exception as e:
                errors[name] = str(e)
                logger.error("Source %s failed: %s", name, e)

        classified = await self._classify_tools(raw_tools)
        classified.sort(key=lambda t: (t.popularity_score, t.ai_relevance_score), reverse=True)
        classified = classified[:limit]

        new_tools = await self._cache.get_new(classified)
        new_count = await self._cache.store(classified)

        return FetchResult(
            tools=classified,
            new_count=new_count,
            cached_count=len(classified) - len(new_tools),
            total_fetched=len(raw_tools),
            errors=errors,
        )

    async def history(self, limit: int = 50) -> list[AITool]:
        """Retrieve previously fetched tools from cache."""
        await self._ensure_init()
        return await self._cache.history(limit=limit)

    def _resolve_sources(self, requested: list[str]) -> dict[str, BaseSource]:
        """Resolve requested source names to active source instances."""
        if requested:
            return {
                name: self._sources[name]
                for name in requested
                if name in self._sources
            }
        return dict(self._sources)

    async def _fetch_all(
        self,
        sources: dict[str, BaseSource],
        filters: SearchFilters,
    ) -> tuple[list[AITool], dict[str, str]]:
        """Fetch from all sources concurrently."""
        raw_tools: list[AITool] = []
        errors: dict[str, str] = {}

        tasks = {
            name: asyncio.create_task(source.search(filters))
            for name, source in sources.items()
        }

        for name, task in tasks.items():
            try:
                result = await task
                raw_tools.extend(result)
            except Exception as e:
                errors[name] = str(e)
                logger.error("Source %s failed: %s", name, e)

        return raw_tools, errors

    async def _classify_tools(self, tools: list[AITool]) -> list[AITool]:
        """Run classification pipeline on tools."""
        if not tools:
            return []

        results = await self._classifier.classify_batch(tools)

        classified: list[AITool] = []
        for tool, result in zip(tools, results):
            updated = tool.model_copy(update={
                "category": result.category,
                "ai_relevance_score": result.relevance_score,
            })
            classified.append(updated)

        return classified
