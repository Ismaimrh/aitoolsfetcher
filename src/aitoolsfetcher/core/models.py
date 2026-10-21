"""Core data models for AIToolsFetcher."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, computed_field


class AICategory(str, Enum):
    """Classification categories for AI tools."""

    AGENT = "agent"
    FRAMEWORK = "framework"
    MODEL = "model"
    DATASET = "dataset"
    INFRA = "infra"
    DEV_TOOL = "dev_tool"
    LIBRARY = "library"
    CLI_TOOL = "cli_tool"
    OTHER = "other"


class AITool(BaseModel):
    """Represents a discovered AI tool from any platform."""

    name: str
    description: str
    category: AICategory = AICategory.OTHER
    source_platform: str
    source_url: str
    source_code_url: str | None = None
    homepage_url: str | None = None
    stars: int | None = None
    language: str | None = None
    topics: list[str] = Field(default_factory=list)
    license: str | None = None
    author: str = ""
    discovered_at: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    last_updated: datetime | None = None
    popularity_score: float = 0.0
    ai_relevance_score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def id(self) -> str:
        """Content-based unique identifier for deduplication."""
        key = f"{self.name.lower().strip()}:{self.source_platform}:{self.author.lower().strip()}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def source_code_or_source(self) -> str:
        """Best available link — prefers source code, falls back to discovery URL."""
        return self.source_code_url or self.source_url


class SearchFilters(BaseModel):
    """Filters applied to tool searches."""

    query: str = ""
    sources: list[str] = Field(default_factory=list)
    categories: list[AICategory] = Field(default_factory=list)
    min_stars: int = 0
    min_relevance: float = 0.5
    since: datetime | None = None
    language: str | None = None
    limit: int = 20


class FetchResult(BaseModel):
    """Aggregated result from a fetch operation."""

    tools: list[AITool] = Field(default_factory=list)
    new_count: int = 0
    new_ids: list[str] = Field(
        default_factory=list, description="IDs of tools in `tools` that were not cached before"
    )
    cached_count: int = 0
    total_fetched: int = 0
    errors: dict[str, str] = Field(default_factory=dict)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def summary(self) -> str:
        """Human-readable summary of the fetch."""
        parts = [f"{self.new_count} new AI tools found"]
        parts.append(f"({self.total_fetched} fetched, {self.cached_count} cached)")
        if self.errors:
            failed = ", ".join(self.errors.keys())
            parts.append(f"[errors: {failed}]")
        return " ".join(parts)
