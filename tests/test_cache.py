"""Tests for the SQLite cache layer."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from aitoolsfetcher.core.cache import ToolCache
from aitoolsfetcher.core.models import AITool


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

    async def test_store_updates_existing_entry(self, cache: ToolCache, sample_tool: AITool):
        await cache.store([sample_tool])
        await cache.store([sample_tool.model_copy(update={"stars": 9999})])
        [stored] = await cache.history()
        assert stored.stars == 9999

    async def test_search_treats_punctuation_literally(
        self, cache: ToolCache, sample_tools: list[AITool]
    ):
        await cache.store(sample_tools)
        # "-" and ":" are FTS5 operators; unescaped they raise a syntax error
        results = await cache.search("vector-database: RAG")
        assert [t.name for t in results] == ["VectorForge"]
        assert await cache.search("!!!") == []

    async def test_search_finds_nothing_after_clear(
        self, cache: ToolCache, sample_tools: list[AITool]
    ):
        await cache.store(sample_tools)
        await cache.clear()
        assert await cache.search("vector") == []

    async def test_history_since(self, cache: ToolCache, sample_tool: AITool):
        old = sample_tool.model_copy(
            update={
                "name": "Old",
                "source_code_url": "https://github.com/test/old",
                "discovered_at": datetime(2020, 1, 1, tzinfo=timezone.utc),
            }
        )
        new = sample_tool.model_copy(
            update={
                "name": "New",
                "source_code_url": "https://github.com/test/new",
                "discovered_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
            }
        )
        await cache.store([old, new])
        recent = await cache.history(since=datetime(2025, 1, 1, tzinfo=timezone.utc))
        assert [t.name for t in recent] == ["New"]

    async def test_prune_removes_only_old_entries(
        self, cache: ToolCache, sample_tools: list[AITool]
    ):
        await cache.store(sample_tools)
        db = cache._ensure_open()
        await db.execute(
            "UPDATE tools SET created_at = datetime('now', '-100 days') WHERE name = ?",
            ("VectorForge",),
        )
        # Created today but earlier than "now": must survive a same-day cutoff
        await db.execute(
            "UPDATE tools SET created_at = datetime('now', '-1 minutes') WHERE name = ?",
            ("PromptBench",),
        )
        await db.commit()

        assert await cache.prune(ttl_days=90) == 1
        assert {t.name for t in await cache.history()} == {"PromptBench", "WebFramework"}
        assert await cache.prune(ttl_days=0) == 1  # the 1-minute-old entry

    async def test_use_before_open_raises(self, tmp_path):
        closed = ToolCache(db_path=str(tmp_path / "c.db"))
        with pytest.raises(RuntimeError, match="not open"):
            await closed.count()

    async def test_creates_parent_folder(self, tmp_path):
        path = tmp_path / "nested" / "dir" / "cache.db"
        async with ToolCache(db_path=str(path)) as c:
            assert await c.count() == 0
        assert path.exists()
