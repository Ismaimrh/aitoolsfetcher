"""Tests for GitHub source adapter."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from aitoolsfetcher.core.models import SearchFilters
from aitoolsfetcher.sources.github import GitHubSource


@pytest.fixture
def github_api_response():
    """Mock GitHub API search response."""
    return {
        "total_count": 2,
        "items": [
            {
                "name": "langchain",
                "full_name": "langchain-ai/langchain",
                "description": "Build context-aware reasoning applications",
                "html_url": "https://github.com/langchain-ai/langchain",
                "homepage": "https://langchain.com",
                "stargazers_count": 95000,
                "forks_count": 15000,
                "language": "Python",
                "topics": ["llm", "ai", "langchain", "framework"],
                "license": {"spdx_id": "MIT"},
                "owner": {"login": "langchain-ai"},
                "pushed_at": "2025-01-10T12:00:00Z",
                "archived": False,
                "size": 50000,
                "open_issues_count": 500,
                "watchers_count": 95000,
            },
            {
                "name": "autogen",
                "full_name": "microsoft/autogen",
                "description": "Multi-agent conversation framework",
                "html_url": "https://github.com/microsoft/autogen",
                "homepage": "",
                "stargazers_count": 35000,
                "forks_count": 5000,
                "language": "Python",
                "topics": ["multi-agent", "llm", "ai-agent"],
                "license": {"spdx_id": "MIT"},
                "owner": {"login": "microsoft"},
                "pushed_at": "2025-01-09T08:00:00Z",
                "archived": False,
                "size": 30000,
                "open_issues_count": 200,
                "watchers_count": 35000,
            },
        ],
    }


class TestGitHubSource:
    @respx.mock
    async def test_search_returns_tools(self, github_api_response):
        respx.get("https://api.github.com/search/repositories").mock(
            return_value=httpx.Response(200, json=github_api_response)
        )

        async with GitHubSource() as source:
            filters = SearchFilters(query="AI agents", limit=10)
            tools = await source.search(filters)

        assert len(tools) == 2
        assert tools[0].name == "langchain"
        assert tools[0].source_platform == "github"
        assert tools[0].stars == 95000
        assert tools[0].author == "langchain-ai"

    @respx.mock
    async def test_search_handles_rate_limit(self):
        respx.get("https://api.github.com/search/repositories").mock(
            return_value=httpx.Response(403, json={"message": "rate limit exceeded"})
        )

        async with GitHubSource() as source:
            filters = SearchFilters(query="test", limit=5)
            tools = await source.search(filters)

        assert tools == []

    @respx.mock
    async def test_search_handles_empty_response(self):
        respx.get("https://api.github.com/search/repositories").mock(
            return_value=httpx.Response(200, json={"total_count": 0, "items": []})
        )

        async with GitHubSource() as source:
            filters = SearchFilters(query="nonexistent", limit=5)
            tools = await source.search(filters)

        assert tools == []

    @respx.mock
    async def test_search_respects_min_stars(self, github_api_response):
        respx.get("https://api.github.com/search/repositories").mock(
            return_value=httpx.Response(200, json=github_api_response)
        )

        async with GitHubSource() as source:
            filters = SearchFilters(query="AI", limit=10, min_stars=50000)
            tools = await source.search(filters)

        assert len(tools) == 1
        assert tools[0].name == "langchain"

    def test_infer_category_from_topics(self):
        assert GitHubSource._infer_category(["ai-agent", "llm"], "test", "") == "agent"
        assert GitHubSource._infer_category(["vector-database"], "test", "") == "infra"
        assert GitHubSource._infer_category(["deep-learning", "pytorch"], "test", "") == "framework"

    def test_infer_category_from_description(self):
        assert GitHubSource._infer_category([], "test", "an autonomous agent") == "agent"
        assert GitHubSource._infer_category([], "my-cli", "command-line tool") == "cli_tool"
