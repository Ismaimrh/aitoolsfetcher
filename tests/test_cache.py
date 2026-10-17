"""Tests for the SQLite cache layer."""

from __future__ import annotations

import pytest

from aitoolsfetcher.core.cache import ToolCache
from aitoolsfetcher.core.models import AICategory, AITool


class TestToolCache:
    async def test_store_and_has(self, cache: ToolCache, sample_tool: AITool):
        assert not await cache.has(sample_tool.id)
        count = await cache.store([sample_tool])
        assert count == 1
        assert await cache.has(sample_tool.id)

    async def test_store_dedup(self, cache: ToolCache, sample_tool: AITool):
        await cache.store([sample_tool])
        count = await cache.store([sample_tool])
        assert count == 0  # No new inserts

    async def test_get_new(self, cache: ToolCache, sample_tools: list[AITool]):
        # Store first tool
        await cache.store([sample_tools[0]])

        # get_new should exclude the cached one
        new = await cache.get_new(sample_tools)
        new_names = {t.name for t in new}
        assert sample_tools[0].name not in new_names
        assert sample_tools[1].name in new_names

    async def test_search_fts(self, cache: ToolCache, sample_tools: list[AITool]):
        await cache.store(sample_tools)
        results = await cache.search("vector database")
        assert any(t.name == "VectorForge" for t in results)

    async def test_history(self, cache: ToolCache, sample_tools: list[AITool]):
        await cache.store(sample_tools)
        history = await cache.history(limit=10)
        assert len(history) == len(sample_tools)

    async def test_count(self, cache: ToolCache, sample_tools: list[AITool]):
        assert await cache.count() == 0
        await cache.store(sample_tools)
        assert await cache.count() == len(sample_tools)

    async def test_clear(self, cache: ToolCache, sample_tools: list[AITool]):
        await cache.store(sample_tools)
        await cache.clear()
        assert await cache.count() == 0

    async def test_cross_platform_dedup(self, cache: ToolCache):
        """Same source_code_url from different platforms should merge."""
        tool1 = AITool(
            name="CoolTool",
            description="An AI tool",
            source_platform="github",
            source_url="https://github.com/cool/tool",
            source_code_url="https://github.com/cool/tool",
            stars=100,
            author="cool",
        )
        tool2 = AITool(
            name="CoolTool shared on Reddit",
            description="Found this cool AI tool",
            source_platform="reddit",
            source_url="https://reddit.com/r/ai/cool_tool",
            source_code_url="https://github.com/cool/tool",
            stars=50,
            author="redditor",
        )
        await cache.store([tool1])
        count = await cache.store([tool2])
        assert count == 0  # Should merge, not create new
        assert await cache.count() == 1
