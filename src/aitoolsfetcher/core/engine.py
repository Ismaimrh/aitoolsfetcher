"""Core fetch engine — orchestrates sources, classification and caching."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from aitoolsfetcher.classifiers.keyword import KeywordClassifier
from aitoolsfetcher.config import Settings, load_settings
from aitoolsfetcher.core.cache import ToolCache
from aitoolsfetcher.core.models import AITool, FetchResult, SearchFilters

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from aitoolsfetcher.classifiers.base import BaseClassifier
    from aitoolsfetcher.sources.base import BaseSource

logger = logging.getLogger(__name__)


def _build_sources(settings: Settings) -> dict[str, BaseSource]:
    """Build the enabled source adapters.

    Args:
        settings: Loaded settings.

    Returns:
        Mapping of source name to adapter instance.
    """
    sources: dict[str, BaseSource] = {}

    if settings.is_source_enabled("github"):
        from aitoolsfetcher.sources.github import GitHubSource

        sources["github"] = GitHubSource(token=settings.github_token)

    return sources


def _build_classifier(settings: Settings) -> BaseClassifier:
    """Build the classifier: LLM if a key is configured, otherwise keyword.

    Args:
        settings: Loaded settings.

    Returns:
        A classifier instance.
    """
    if settings.has_llm_key:
        from aitoolsfetcher.classifiers.llm import LLMClassifier

        if LLMClassifier.is_available(settings.llm_provider):
            return LLMClassifier(api_key=settings.llm_api_key, provider=settings.llm_provider)
        logger.warning(
            "LLM key set but provider %r is unsupported or its package is not installed "
            "(pip install 'aitoolsfetcher[llm]'); using the keyword classifier",
            settings.llm_provider,
        )
    return KeywordClassifier()


class AIToolsFetcher:
    """Main entry point for discovering AI tools.

    Example:
        async with AIToolsFetcher() as fetcher:
            result = await fetcher.search("AI agents")
            for tool in result.tools:
                print(tool.name, tool.source_code_or_source)
    """

    def __init__(
        self,
        settings: Settings | None = None,
        sources: dict[str, BaseSource] | None = None,
        classifier: BaseClassifier | None = None,
        cache: ToolCache | None = None,
    ) -> None:
        """Create a fetcher. Every dependency can be injected for testing.

        Args:
            settings: Settings to use. Defaults to ``load_settings()``.
            sources: Source adapters by name. Defaults to the enabled built-ins.
            classifier: Classifier. Defaults to LLM if a key is set, else keyword.
            cache: Tool cache. Defaults to ``~/.aitoolsfetcher/cache.db``.
        """
        self._settings = settings if settings is not None else load_settings()
        self._sources = sources if sources is not None else _build_sources(self._settings)
        self._classifier = (
            classifier if classifier is not None else _build_classifier(self._settings)
        )
        self._cache = cache if cache is not None else ToolCache()
        self._initialized = False

    async def _ensure_init(self) -> None:
        """Open the cache and drop entries older than the configured TTL."""
        if not self._initialized:
            await self._cache.open()
            removed = await self._cache.prune(self._settings.cache_ttl_days)
            if removed:
                logger.info(
                    "Pruned %d cache entries older than %d days",
                    removed,
                    self._settings.cache_ttl_days,
                )
            self._initialized = True

    async def close(self) -> None:
        """Release all resources (HTTP clients and the cache connection)."""
        for source in self._sources.values():
            await source.close()
        await self._cache.close()
        self._initialized = False

    async def __aenter__(self) -> AIToolsFetcher:
        """Open resources on context entry."""
        await self._ensure_init()
        return self

    async def __aexit__(self, *exc: object) -> None:
        """Release resources on context exit."""
        await self.close()

    @property
    def available_sources(self) -> list[str]:
        """Names of the configured source adapters."""
        return list(self._sources.keys())

    async def search(
        self,
        query: str = "",
        sources: list[str] | None = None,
        limit: int = 20,
        min_relevance: float = 0.0,
        min_stars: int = 0,
        **kwargs: Any,
    ) -> FetchResult:
        """Search for AI tools on demand.

        Args:
            query: Search query string.
            sources: Source names to search. None means all enabled sources.
            limit: Maximum number of results to return.
            min_relevance: Minimum AI relevance score (0-1).
            min_stars: Minimum stars/popularity threshold.
            **kwargs: Extra ``SearchFilters`` fields, e.g. ``language`` or ``since``.

        Returns:
            The classified, filtered and cached results.
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

        active_sources = self._resolve_sources(filters.sources)
        if not active_sources:
            return FetchResult()

        raw_tools, errors = await self._gather(
            active_sources, lambda source: source.search(filters)
        )

        classified = await self._classify_tools(raw_tools)
        filtered = [t for t in classified if t.ai_relevance_score >= min_relevance]
        filtered.sort(key=lambda t: (t.ai_relevance_score, t.popularity_score), reverse=True)

        return await self._finalize(filtered[:limit], len(raw_tools), errors)

    async def trending(
        self,
        sources: list[str] | None = None,
        limit: int = 20,
    ) -> FetchResult:
        """Fetch currently trending AI tools.

        Args:
            sources: Source names to query. None means all enabled sources.
            limit: Maximum number of results to return.

        Returns:
            The classified and cached results, most popular first.
        """
        await self._ensure_init()

        active_sources = self._resolve_sources(sources or [])
        if not active_sources:
            return FetchResult()

        raw_tools, errors = await self._gather(
            active_sources, lambda source: source.trending(limit=limit)
        )

        # The trending page is not AI-specific, so keep only what the classifier accepts
        classified = await self._classify_tools(raw_tools, ai_only=True)
        classified.sort(key=lambda t: (t.popularity_score, t.ai_relevance_score), reverse=True)

        return await self._finalize(classified[:limit], len(raw_tools), errors)

    async def history(self, limit: int = 50) -> list[AITool]:
        """Return previously fetched tools from the cache, newest first.

        Args:
            limit: Maximum number of tools.

        Returns:
            Cached tools.
        """
        await self._ensure_init()
        return await self._cache.history(limit=limit)

    def _resolve_sources(self, requested: list[str]) -> dict[str, BaseSource]:
        """Map requested source names to configured adapters; unknown names are skipped."""
        if requested:
            return {name: self._sources[name] for name in requested if name in self._sources}
        return dict(self._sources)

    @staticmethod
    async def _gather(
        sources: dict[str, BaseSource],
        call: Callable[[BaseSource], Awaitable[list[AITool]]],
    ) -> tuple[list[AITool], dict[str, str]]:
        """Run ``call`` on every source concurrently.

        A failing source is recorded in the error map and never stops the others.
        """
        raw_tools: list[AITool] = []
        errors: dict[str, str] = {}

        names = list(sources)
        results = await asyncio.gather(
            *(call(sources[name]) for name in names), return_exceptions=True
        )
        for name, result in zip(names, results, strict=True):
            if isinstance(result, BaseException):
                errors[name] = str(result) or type(result).__name__
                logger.error("Source %s failed: %s", name, result)
            else:
                raw_tools.extend(result)

        return raw_tools, errors

    async def _finalize(
        self, tools: list[AITool], total_fetched: int, errors: dict[str, str]
    ) -> FetchResult:
        """Store tools in the cache and work out which ones are new."""
        new_tools = await self._cache.get_new(tools)
        new_count = await self._cache.store(tools)
        return FetchResult(
            tools=tools,
            new_count=new_count,
            new_ids=[t.id for t in new_tools],
            cached_count=len(tools) - len(new_tools),
            total_fetched=total_fetched,
            errors=errors,
        )

    async def _classify_tools(self, tools: list[AITool], ai_only: bool = False) -> list[AITool]:
        """Classify tools and copy the category and relevance score onto each one.

        Args:
            tools: Tools to classify.
            ai_only: If True, drop tools the classifier says are not AI-related.
        """
        if not tools:
            return []

        results = await self._classifier.classify_batch(tools)

        return [
            tool.model_copy(
                update={"category": result.category, "ai_relevance_score": result.relevance_score}
            )
            for tool, result in zip(tools, results, strict=True)
            if result.is_ai_tool or not ai_only
        ]
