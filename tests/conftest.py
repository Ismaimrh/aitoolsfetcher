"""Shared test fixtures."""

from __future__ import annotations

from datetime import datetime

import pytest

from aitoolsfetcher import config as config_module
from aitoolsfetcher.core import cache as cache_module
from aitoolsfetcher.core.cache import ToolCache
from aitoolsfetcher.core.models import AICategory, AITool


@pytest.fixture
def sample_tool() -> AITool:
    """A sample AI tool for testing."""
    return AITool(
        name="AutoAgent",
        description="Build autonomous AI agents with MCP tool use and multi-step planning",
        category=AICategory.AGENT,
        source_platform="github",
        source_url="https://github.com/test/autoagent",
        source_code_url="https://github.com/test/autoagent",
        stars=1500,
        language="Python",
        topics=["ai-agent", "llm", "mcp", "autonomous"],
        license="MIT",
        author="testuser",
        discovered_at=datetime(2025, 1, 15),
        popularity_score=0.015,
        ai_relevance_score=0.85,
    )


@pytest.fixture
def sample_tools() -> list[AITool]:
    """Multiple sample tools for testing."""
    return [
        AITool(
            name="VectorForge",
            description="High-performance vector database for RAG pipelines",
            category=AICategory.INFRA,
            source_platform="github",
            source_url="https://github.com/vf/vectorforge",
            source_code_url="https://github.com/vf/vectorforge",
            stars=3200,
            language="Rust",
            topics=["vector-database", "rag", "embedding"],
            author="vfteam",
            popularity_score=0.032,
            ai_relevance_score=0.9,
        ),
        AITool(
            name="PromptBench",
            description="Evaluate and compare LLM prompts with automated testing",
            category=AICategory.DEV_TOOL,
            source_platform="huggingface",
            source_url="https://huggingface.co/spaces/promptbench",
            stars=800,
            language="Python",
            topics=["prompt-engineering", "evaluation", "llm"],
            author="pbench",
            popularity_score=0.008,
            ai_relevance_score=0.8,
        ),
        AITool(
            name="WebFramework",
            description="A fast web framework for building APIs",
            category=AICategory.OTHER,
            source_platform="github",
            source_url="https://github.com/web/framework",
            source_code_url="https://github.com/web/framework",
            stars=5000,
            language="Go",
            topics=["web", "api", "http"],
            author="webdev",
            popularity_score=0.05,
            ai_relevance_score=0.1,
        ),
    ]


_SECRET_ENV_VARS = (
    "GITHUB_TOKEN",
    "AITF_GITHUB_TOKEN",
    "HF_TOKEN",
    "AITF_HF_TOKEN",
    "AITF_NOTION_KEY",
    "AITF_LLM_KEY",
    "AITF_LLM_PROVIDER",
)


@pytest.fixture(autouse=True)
def isolated_user_dir(tmp_path, monkeypatch):
    """Point config and cache at a temp folder and clear token env vars.

    No test can read the developer's real ~/.aitoolsfetcher/config.toml or
    cache, or pick up a real API key from the environment.
    """
    home = tmp_path / "aitf-home"
    monkeypatch.setattr(config_module, "CONFIG_DIR", home)
    monkeypatch.setattr(config_module, "CONFIG_FILE", home / "config.toml")
    monkeypatch.setattr(config_module, "CACHE_DB", home / "cache.db")
    monkeypatch.setattr(cache_module, "CACHE_DB", home / "cache.db")
    for var in _SECRET_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    return home


@pytest.fixture
async def cache(tmp_path):
    """Temporary cache instance for testing."""
    db_path = str(tmp_path / "test_cache.db")
    cache = ToolCache(db_path=db_path)
    await cache.open()
    yield cache
    await cache.close()
