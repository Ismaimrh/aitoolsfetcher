"""Tests for GitHub source adapter."""

from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pytest
import respx

from aitoolsfetcher.core.models import SearchFilters
from aitoolsfetcher.sources.base import USER_AGENT
from aitoolsfetcher.sources.github import GitHubRateLimitError, GitHubSource, _rate_limit_wait


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
    async def test_search_raises_when_rate_limited_from_the_start(self):
        respx.get("https://api.github.com/search/repositories").mock(
            return_value=httpx.Response(403, json={"message": "API rate limit exceeded"})
        )
        waits: list[float] = []

        async def fake_sleep(seconds: float) -> None:
            waits.append(seconds)

        async with GitHubSource(sleep=fake_sleep, max_wait=1000) as source:
            with pytest.raises(GitHubRateLimitError, match="GITHUB_TOKEN"):
                await source.search(SearchFilters(query="test", limit=5))

        # No headers: wait at least a minute, doubling each retry (GitHub guidance)
        assert waits == [60.0, 120.0, 240.0]

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


def _repo(i: int) -> dict:
    return {
        "name": f"repo{i}",
        "full_name": f"owner/repo{i}",
        "html_url": f"https://github.com/owner/repo{i}",
        "description": "an llm tool",
        "stargazers_count": 1000 - i,
        "topics": ["llm"],
        "owner": {"login": "owner"},
    }


class TestGitHubRequests:
    @respx.mock
    async def test_sends_user_agent_and_api_headers_without_token(self):
        route = respx.get("https://api.github.com/search/repositories").mock(
            return_value=httpx.Response(200, json={"items": []})
        )
        async with GitHubSource() as source:
            await source.search(SearchFilters(query="llm", limit=5))

        headers = route.calls.last.request.headers
        assert headers["user-agent"] == USER_AGENT
        assert headers["accept"] == "application/vnd.github+json"
        assert "authorization" not in headers

    @respx.mock
    async def test_sends_token_when_configured(self):
        route = respx.get("https://api.github.com/search/repositories").mock(
            return_value=httpx.Response(200, json={"items": []})
        )
        async with GitHubSource(token="test-token") as source:
            await source.search(SearchFilters(query="llm", limit=5))

        assert route.calls.last.request.headers["authorization"] == "Bearer test-token"

    @respx.mock
    async def test_paginates_with_a_fixed_page_size(self):
        def page(request: httpx.Request) -> httpx.Response:
            n = int(request.url.params["page"])
            per_page = int(request.url.params["per_page"])
            start = (n - 1) * per_page
            return httpx.Response(
                200, json={"items": [_repo(i) for i in range(start, start + per_page)]}
            )

        route = respx.get("https://api.github.com/search/repositories").mock(side_effect=page)
        async with GitHubSource() as source:
            tools = await source.search(SearchFilters(query="llm", limit=150))

        assert len(tools) == 150
        assert len({t.name for t in tools}) == 150  # no duplicates across pages
        assert [c.request.url.params["per_page"] for c in route.calls] == ["100", "100"]

    @respx.mock
    async def test_retries_after_retry_after_header(self):
        route = respx.get("https://api.github.com/search/repositories").mock(
            side_effect=[
                httpx.Response(429, headers={"retry-after": "7"}),
                httpx.Response(200, json={"items": [_repo(1)]}),
            ]
        )
        waits: list[float] = []

        async def fake_sleep(seconds: float) -> None:
            waits.append(seconds)

        async with GitHubSource(sleep=fake_sleep) as source:
            tools = await source.search(SearchFilters(query="llm", limit=1))

        assert waits == [7.0]
        assert route.call_count == 2
        assert [t.name for t in tools] == ["repo1"]

    @respx.mock
    async def test_gives_up_instead_of_waiting_too_long(self):
        respx.get("https://api.github.com/search/repositories").mock(
            return_value=httpx.Response(403, headers={"retry-after": "3600"})
        )
        waits: list[float] = []

        async def fake_sleep(seconds: float) -> None:
            waits.append(seconds)

        async with GitHubSource(token="t", sleep=fake_sleep, max_wait=60) as source:
            with pytest.raises(GitHubRateLimitError, match="3600s"):
                await source.search(SearchFilters(query="llm", limit=1))
        assert waits == []

    @respx.mock
    async def test_keeps_partial_results_when_a_later_page_fails(self):
        respx.get("https://api.github.com/search/repositories").mock(
            side_effect=[
                httpx.Response(200, json={"items": [_repo(i) for i in range(100)]}),
                httpx.Response(500),
            ]
        )
        async with GitHubSource() as source:
            tools = await source.search(SearchFilters(query="llm", limit=150))
        assert len(tools) == 100

    @respx.mock
    async def test_non_rate_limit_403_is_raised(self):
        respx.get("https://api.github.com/search/repositories").mock(
            return_value=httpx.Response(403, json={"message": "Forbidden"})
        )
        async with GitHubSource() as source:
            with pytest.raises(httpx.HTTPStatusError):
                await source.search(SearchFilters(query="llm", limit=1))


class TestRateLimitWait:
    def test_uses_reset_header_when_quota_is_exhausted(self):
        resp = httpx.Response(
            403, headers={"x-ratelimit-remaining": "0", "x-ratelimit-reset": "1030"}
        )
        assert _rate_limit_wait(resp, attempt=0, now=1000.0) == 31.0

    def test_falls_back_to_exponential_minutes(self):
        resp = httpx.Response(429)
        assert _rate_limit_wait(resp, attempt=0, now=0) == 60.0
        assert _rate_limit_wait(resp, attempt=2, now=0) == 240.0


class TestSearchQuery:
    def test_adds_ai_topic_for_generic_queries(self):
        query = GitHubSource()._build_search_query(SearchFilters(query="email parser"))
        assert "topic:ai" in query

    def test_skips_ai_topic_when_query_is_already_about_ai(self):
        query = GitHubSource()._build_search_query(SearchFilters(query="AI agents"))
        assert "topic:ai" not in query

    def test_includes_filters(self):
        since = datetime(2026, 1, 2, tzinfo=timezone.utc)
        query = GitHubSource()._build_search_query(
            SearchFilters(query="llm", language="python", min_stars=50, since=since)
        )
        assert "language:python" in query
        assert "stars:>=50" in query
        assert "pushed:>=2026-01-02" in query


_TRENDING_HTML = """
<html><body>
<article class="Box-row">
  <h2><a href="/acme/agent-kit">acme / agent-kit</a></h2>
  <p>An LLM agent toolkit</p>
  <span itemprop="programmingLanguage">Python</span>
  <a href="/acme/agent-kit/stargazers">1,234</a>
</article>
<article class="Box-row">
  <h2><a href="/broken">broken</a></h2>
</article>
<article class="Box-row">
  <h2><a href="/acme/agent-kit">duplicate</a></h2>
  <p>Same repo listed twice</p>
</article>
</body></html>
"""


class TestTrending:
    @respx.mock
    async def test_parses_trending_page_and_deduplicates(self):
        respx.get(url__startswith="https://github.com/trending").mock(
            return_value=httpx.Response(200, text=_TRENDING_HTML)
        )
        async with GitHubSource() as source:
            tools = await source.trending(limit=10)

        assert len(tools) == 1
        tool = tools[0]
        assert tool.name == "agent-kit"
        assert tool.author == "acme"
        assert tool.stars == 1234
        assert tool.language == "Python"
        assert tool.source_url == "https://github.com/acme/agent-kit"

    @respx.mock
    async def test_failed_pages_are_skipped(self):
        respx.get(url__startswith="https://github.com/trending").mock(
            return_value=httpx.Response(500)
        )
        async with GitHubSource() as source:
            assert await source.trending(limit=5) == []
