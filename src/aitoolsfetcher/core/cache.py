"""SQLite cache layer with FTS5 for deduplication and history."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Optional

import aiosqlite

from aitoolsfetcher.config import CACHE_DB, ensure_config_dir
from aitoolsfetcher.core.models import AITool

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tools (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    category TEXT NOT NULL,
    source_platform TEXT NOT NULL,
    source_url TEXT NOT NULL,
    source_code_url TEXT,
    homepage_url TEXT,
    stars INTEGER,
    language TEXT,
    topics TEXT,
    license TEXT,
    author TEXT,
    discovered_at TEXT NOT NULL,
    last_updated TEXT,
    popularity_score REAL,
    ai_relevance_score REAL,
    metadata TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE VIRTUAL TABLE IF NOT EXISTS tools_fts USING fts5(
    name, description, topics, author,
    content='tools',
    content_rowid='rowid'
);

CREATE TRIGGER IF NOT EXISTS tools_ai AFTER INSERT ON tools BEGIN
    INSERT INTO tools_fts(rowid, name, description, topics, author)
    VALUES (new.rowid, new.name, new.description, new.topics, new.author);
END;

CREATE TRIGGER IF NOT EXISTS tools_ad AFTER DELETE ON tools BEGIN
    INSERT INTO tools_fts(tools_fts, rowid, name, description, topics, author)
    VALUES ('delete', old.rowid, old.name, old.description, old.topics, old.author);
END;

CREATE TRIGGER IF NOT EXISTS tools_au AFTER UPDATE ON tools BEGIN
    INSERT INTO tools_fts(tools_fts, rowid, name, description, topics, author)
    VALUES ('delete', old.rowid, old.name, old.description, old.topics, old.author);
    INSERT INTO tools_fts(rowid, name, description, topics, author)
    VALUES (new.rowid, new.name, new.description, new.topics, new.author);
END;

CREATE INDEX IF NOT EXISTS idx_tools_source ON tools(source_platform);
CREATE INDEX IF NOT EXISTS idx_tools_score ON tools(ai_relevance_score);
CREATE INDEX IF NOT EXISTS idx_tools_discovered ON tools(discovered_at);
CREATE INDEX IF NOT EXISTS idx_tools_source_code ON tools(source_code_url);
"""


def _tool_to_row(tool: AITool) -> dict:
    """Convert an AITool to a database row dict."""
    return {
        "id": tool.id,
        "name": tool.name,
        "description": tool.description,
        "category": tool.category.value,
        "source_platform": tool.source_platform,
        "source_url": tool.source_url,
        "source_code_url": tool.source_code_url,
        "homepage_url": tool.homepage_url,
        "stars": tool.stars,
        "language": tool.language,
        "topics": json.dumps(tool.topics),
        "license": tool.license,
        "author": tool.author,
        "discovered_at": tool.discovered_at.isoformat(),
        "last_updated": tool.last_updated.isoformat() if tool.last_updated else None,
        "popularity_score": tool.popularity_score,
        "ai_relevance_score": tool.ai_relevance_score,
        "metadata": json.dumps(tool.metadata),
    }


def _row_to_tool(row: aiosqlite.Row) -> AITool:
    """Convert a database row to an AITool."""
    data = dict(row)
    data["topics"] = json.loads(data["topics"]) if data["topics"] else []
    data["metadata"] = json.loads(data["metadata"]) if data["metadata"] else {}
    data["discovered_at"] = datetime.fromisoformat(data["discovered_at"])
    if data.get("last_updated"):
        data["last_updated"] = datetime.fromisoformat(data["last_updated"])
    data.pop("created_at", None)
    return AITool(**data)


class ToolCache:
    """Async SQLite cache for tool deduplication and history."""

    def __init__(self, db_path: Optional[str] = None):
        self._db_path = db_path or str(CACHE_DB)
        self._db: Optional[aiosqlite.Connection] = None

    async def open(self) -> None:
        """Initialize database connection and schema."""
        ensure_config_dir()
        self._db = await aiosqlite.connect(self._db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(_SCHEMA)
        await self._db.commit()

    async def close(self) -> None:
        """Close database connection."""
        if self._db:
            await self._db.close()
            self._db = None

    async def __aenter__(self) -> ToolCache:
        await self.open()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()

    def _ensure_open(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("Cache not open. Use 'async with ToolCache()' or call open().")
        return self._db

    async def has(self, tool_id: str) -> bool:
        """Check if a tool exists in cache."""
        db = self._ensure_open()
        async with db.execute("SELECT 1 FROM tools WHERE id = ?", (tool_id,)) as cursor:
            return await cursor.fetchone() is not None

    async def has_source_code(self, source_code_url: str) -> Optional[str]:
        """Check if a tool with the same source code URL exists. Returns existing tool ID."""
        if not source_code_url:
            return None
        db = self._ensure_open()
        async with db.execute(
            "SELECT id FROM tools WHERE source_code_url = ? LIMIT 1", (source_code_url,)
        ) as cursor:
            row = await cursor.fetchone()
            return row["id"] if row else None

    async def store(self, tools: list[AITool]) -> int:
        """Store tools in cache. Returns count of newly inserted tools."""
        db = self._ensure_open()
        new_count = 0
        for tool in tools:
            row = _tool_to_row(tool)
            # Check for cross-platform duplicate via source_code_url
            existing = await self.has_source_code(tool.source_code_url) if tool.source_code_url else None
            if existing and existing != tool.id:
                # Merge: update metadata of existing entry if new one has more stars
                await db.execute(
                    "UPDATE tools SET stars = MAX(COALESCE(stars, 0), ?) WHERE id = ?",
                    (tool.stars or 0, existing),
                )
                continue

            if await self.has(tool.id):
                # Update existing entry with fresh data
                await db.execute(
                    """UPDATE tools SET stars = ?, popularity_score = ?, ai_relevance_score = ?,
                       last_updated = ?, metadata = ? WHERE id = ?""",
                    (tool.stars, tool.popularity_score, tool.ai_relevance_score,
                     row["last_updated"], row["metadata"], tool.id),
                )
            else:
                columns = ", ".join(row.keys())
                placeholders = ", ".join(["?"] * len(row))
                await db.execute(
                    f"INSERT INTO tools ({columns}) VALUES ({placeholders})",  # noqa: S608
                    tuple(row.values()),
                )
                new_count += 1
        await db.commit()
        return new_count

    async def get_new(self, tools: list[AITool]) -> list[AITool]:
        """Filter list to only tools not already in cache."""
        result = []
        for tool in tools:
            if not await self.has(tool.id):
                # Also check cross-platform duplicate
                if tool.source_code_url and await self.has_source_code(tool.source_code_url):
                    continue
                result.append(tool)
        return result

    async def search(self, query: str, limit: int = 50) -> list[AITool]:
        """Full-text search across cached tools."""
        db = self._ensure_open()
        async with db.execute(
            """SELECT tools.* FROM tools_fts
               JOIN tools ON tools.rowid = tools_fts.rowid
               WHERE tools_fts MATCH ?
               ORDER BY rank LIMIT ?""",
            (query, limit),
        ) as cursor:
            rows = await cursor.fetchall()
        return [_row_to_tool(row) for row in rows]

    async def history(
        self, since: Optional[datetime] = None, limit: int = 100
    ) -> list[AITool]:
        """Retrieve tools from cache history."""
        db = self._ensure_open()
        if since:
            sql = "SELECT * FROM tools WHERE discovered_at >= ? ORDER BY discovered_at DESC LIMIT ?"
            params: tuple = (since.isoformat(), limit)
        else:
            sql = "SELECT * FROM tools ORDER BY discovered_at DESC LIMIT ?"
            params = (limit,)
        async with db.execute(sql, params) as cursor:
            rows = await cursor.fetchall()
        return [_row_to_tool(row) for row in rows]

    async def prune(self, ttl_days: int = 90) -> int:
        """Remove entries older than TTL. Returns count removed."""
        db = self._ensure_open()
        cutoff = (datetime.now(tz=timezone.utc) - timedelta(days=ttl_days)).isoformat()
        cursor = await db.execute("DELETE FROM tools WHERE created_at < ?", (cutoff,))
        await db.commit()
        return cursor.rowcount

    async def count(self) -> int:
        """Total tools in cache."""
        db = self._ensure_open()
        async with db.execute("SELECT COUNT(*) FROM tools") as cursor:
            row = await cursor.fetchone()
            return row[0] if row else 0

    async def clear(self) -> None:
        """Remove all entries from cache."""
        db = self._ensure_open()
        await db.executescript("DELETE FROM tools; DELETE FROM tools_fts;")
        await db.commit()
