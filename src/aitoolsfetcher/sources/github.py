"""GitHub source adapter — primary and most reliable data source."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import time
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

import httpx
from bs4 import BeautifulSoup

from aitoolsfetcher.core.models import AICategory, AITool, SearchFilters
from aitoolsfetcher.sources.base import BaseSource

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)

_API_BASE = "https://api.github.com"
_TRENDING_URL = "https://github.com/trending"
_SEARCH_PAGE_SIZE = 100  # GitHub search API maximum

# Topic-to-category mapping
_TOPIC_CATEGORIES: dict[str, AICategory] = {
    "agent": AICategory.AGENT,
    "ai-agent": AICategory.AGENT,
    "autonomous-agent": AICategory.AGENT,
    "llm-agent": AICategory.AGENT,
    "multi-agent": AICategory.AGENT,
    "framework": AICategory.FRAMEWORK,
    "ai-framework": AICategory.FRAMEWORK,
    "llm-framework": AICategory.FRAMEWORK,
    "machine-learning": AICategory.FRAMEWORK,
    "deep-learning": AICategory.FRAMEWORK,
    "model": AICategory.MODEL,
    "language-model": AICategory.MODEL,
    "large-language-model": AICategory.MODEL,
    "diffusion-model": AICategory.MODEL,
    "foundation-model": AICategory.MODEL,
    "dataset": AICategory.DATASET,
    "ai-dataset": AICategory.DATASET,
    "benchmark": AICategory.DATASET,
    "vector-database": AICategory.INFRA,
    "rag": AICategory.INFRA,
    "inference": AICategory.INFRA,
    "model-serving": AICategory.INFRA,
    "fine-tuning": AICategory.INFRA,
    "quantization": AICategory.INFRA,
    "mlops": AICategory.INFRA,
    "prompt-engineering": AICategory.DEV_TOOL,
    "evaluation": AICategory.DEV_TOOL,
    "llm-evaluation": AICategory.DEV_TOOL,
    "ai-testing": AICategory.DEV_TOOL,
    "cli": AICategory.CLI_TOOL,
    "ai-cli": AICategory.CLI_TOOL,
}

# If the query already contains one of these words, no extra topic:ai filter is added
_AI_TOPICS = frozenset(
    {
        "llm",
        "ai",
        "machine-learning",
        "deep-learning",
        "gpt",
        "transformer",
        "langchain",
        "llamaindex",
        "autogen",
        "crewai",
        "agent",
        "agents",
        "rag",
        "vector-database",
        "fine-tuning",
        "prompt-engineering",
        "diffusion",
        "neural-network",
        "nlp",
        "computer-vision",
        "generative-ai",
        "large-language-model",
        "embedding",
        "huggingface",
        "pytorch",
        "tensorflow",
        "mcp",
        "function-calling",
        "tool-use",
    }
)


class GitHubRateLimitError(RuntimeError):
    """Raised when GitHub's rate limit would require waiting longer than allowed."""


def _is_rate_limited(resp: httpx.Response) -> bool:
    """Tell a rate-limit response apart from other 403/429 errors."""
    if resp.status_code == 429:
        return True
    if resp.status_code != 403:
        return False
    if resp.headers.get("x-ratelimit-remaining") == "0" or "retry-after" in resp.headers:
        return True
    return "rate limit" in resp.text.lower()


def _rate_limit_wait(resp: httpx.Response, attempt: int, now: float) -> float:
    """Seconds to wait before retrying, following GitHub's REST API guidance.

    1. ``retry-after`` header: wait that many seconds.
    2. ``x-ratelimit-remaining: 0``: wait until ``x-ratelimit-reset`` (epoch seconds).
    3. Otherwise: wait at least one minute, doubling on each retry.
    """
    retry_after = resp.headers.get("retry-after")
    if retry_after is not None:
        with contextlib.suppress(ValueError):
            return max(float(retry_after), 0.0)
    reset = resp.headers.get("x-ratelimit-reset")
    if resp.headers.get("x-ratelimit-remaining") == "0" and reset is not None:
        with contextlib.suppress(ValueError):
            return max(float(reset) - now, 0.0) + 1.0
    return 60.0 * float(2**attempt)


class GitHubSource(BaseSource):
    """Fetch AI tools from GitHub.

    ``search`` uses the official REST search API. ``trending`` parses the public
    github.com/trending page, because GitHub has no trending API.
    """

    name = "github"
    requires_auth = False  # Works without auth, better with token

    def __init__(
        self,
        token: str = "",
        client: httpx.AsyncClient | None = None,
        max_retries: int = 3,
        max_wait: float = 60.0,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        """Create the GitHub adapter.

        Args:
            token: Optional GitHub token, for a higher search rate limit.
            client: Optional shared HTTP client.
            max_retries: Retries after a rate-limit response.
            max_wait: Longest single wait (seconds) the adapter accepts. If GitHub
                asks for longer, it stops and raises ``GitHubRateLimitError``.
            sleep: Async sleep function (replaceable in tests).
        """
        super().__init__(client=client)
        self._token = token
        self._max_retries = max_retries
        self._max_wait = max_wait
        self._sleep = sleep

    async def get_client(self) -> httpx.AsyncClient:
        """Get the HTTP client with GitHub API headers and, if set, the token.

        Returns:
            The configured client.
        """
        client = await super().get_client()
        # httpx sends "Accept: */*" by default, so set GitHub's media type explicitly
        client.headers["Accept"] = "application/vnd.github+json"
        client.headers["X-GitHub-Api-Version"] = "2022-11-28"
        if self._token and "Authorization" not in client.headers:
            client.headers["Authorization"] = f"Bearer {self._token}"
        return client

    async def _get(self, url: str, params: dict[str, Any]) -> httpx.Response:
        """GET a URL, waiting and retrying on rate limits as GitHub asks.

        Raises:
            GitHubRateLimitError: If retries run out or the wait is too long.
            httpx.HTTPError: For other HTTP or network errors.
        """
        client = await self.get_client()
        for attempt in range(self._max_retries + 1):
            resp = await client.get(url, params=params)
            if not _is_rate_limited(resp):
                resp.raise_for_status()
                return resp

            wait = _rate_limit_wait(resp, attempt, time.time())
            if attempt >= self._max_retries or wait > self._max_wait:
                hint = "" if self._token else " Set GITHUB_TOKEN for a higher limit."
                raise GitHubRateLimitError(
                    f"GitHub rate limit reached; retry in about {int(wait)}s.{hint}"
                )
            logger.warning("GitHub rate limit hit; waiting %.0fs before retrying", wait)
            await self._sleep(wait)

        raise AssertionError("unreachable")  # pragma: no cover

    async def search(self, filters: SearchFilters) -> list[AITool]:
        """Search GitHub repositories with the REST search API.

        Args:
            filters: Query, limit, language, minimum stars and date filters.

        Returns:
            Matching repositories, most-starred first. If a later page fails,
            the results collected so far are returned.

        Raises:
            GitHubRateLimitError: If rate-limited before any result was collected.
            httpx.HTTPError: If the first request fails.
        """
        query = self._build_search_query(filters)
        per_page = max(1, min(filters.limit, _SEARCH_PAGE_SIZE))
        tools: list[AITool] = []
        page = 1

        while len(tools) < filters.limit:
            try:
                resp = await self._get(
                    f"{_API_BASE}/search/repositories",
                    {
                        "q": query,
                        "sort": "stars",
                        "order": "desc",
                        "per_page": per_page,
                        "page": page,
                    },
                )
            except (GitHubRateLimitError, httpx.HTTPError) as e:
                if not tools:
                    raise
                logger.warning("GitHub search stopped early, keeping partial results: %s", e)
                break

            items = resp.json().get("items", [])
            for item in items:
                tool = self._repo_to_tool(item)
                if tool and (tool.stars or 0) >= filters.min_stars:
                    tools.append(tool)

            if len(items) < per_page:
                break
            page += 1

        return tools[: filters.limit]

    async def trending(self, limit: int = 20) -> list[AITool]:
        """Parse the github.com/trending page (weekly) for popular repositories.

        The page lists all trending repos; the engine's classifier decides which
        ones are AI-related.

        Args:
            limit: Maximum number of repositories.

        Returns:
            Trending repositories, de-duplicated.
        """
        tools: list[AITool] = []

        for language in ("", "python", "javascript", "typescript", "rust", "go"):
            url = f"{_TRENDING_URL}/{language}" if language else _TRENDING_URL
            try:
                resp = await self._get(url, {"since": "weekly"})
            except (GitHubRateLimitError, httpx.HTTPError) as e:
                logger.warning("Failed to fetch trending page (%s): %s", language or "all", e)
                continue

            tools.extend(self._parse_trending_html(resp.text))
            if len(tools) >= limit * 2:  # fetch extra; non-AI repos are filtered later
                break

        seen: set[str] = set()
        unique: list[AITool] = []
        for tool in tools:
            if tool.source_url not in seen:
                seen.add(tool.source_url)
                unique.append(tool)

        return unique[: limit * 2]

    def _build_search_query(self, filters: SearchFilters) -> str:
        """Build the GitHub search query string from filters."""
        parts: list[str] = []

        if filters.query:
            parts.append(filters.query)

        # Narrow to AI repos unless the query already says so
        words = set(re.findall(r"[a-z0-9-]+", filters.query.lower()))
        if not words & _AI_TOPICS:
            parts.append("topic:ai")

        if filters.language:
            parts.append(f"language:{filters.language}")

        if filters.min_stars:
            parts.append(f"stars:>={filters.min_stars}")

        if filters.since:
            parts.append(f"pushed:>={filters.since.strftime('%Y-%m-%d')}")
        else:
            cutoff = (datetime.now(tz=timezone.utc) - timedelta(days=90)).strftime("%Y-%m-%d")
            parts.append(f"pushed:>={cutoff}")

        return " ".join(parts)

    def _repo_to_tool(self, repo: dict[str, Any]) -> AITool | None:
        """Convert a GitHub API repository object to an AITool."""
        name = repo.get("name", "")
        description = repo.get("description") or ""
        topics = repo.get("topics", [])
        full_name = repo.get("full_name", "")

        if not name:
            return None

        category = self._infer_category(topics, name, description)

        last_updated = None
        if repo.get("pushed_at"):
            with contextlib.suppress(ValueError, TypeError):
                last_updated = datetime.fromisoformat(repo["pushed_at"].replace("Z", "+00:00"))

        stars = repo.get("stargazers_count", 0)
        popularity = min(stars / 100_000, 1.0) if stars else 0.0

        return AITool(
            name=name,
            description=description[:500],
            category=category,
            source_platform="github",
            source_url=repo.get("html_url", f"https://github.com/{full_name}"),
            source_code_url=repo.get("html_url", f"https://github.com/{full_name}"),
            homepage_url=repo.get("homepage") or None,
            stars=stars,
            language=repo.get("language"),
            topics=topics,
            license=(repo.get("license") or {}).get("spdx_id"),
            author=repo.get("owner", {}).get("login", ""),
            last_updated=last_updated,
            popularity_score=popularity,
            metadata={
                "forks": repo.get("forks_count", 0),
                "open_issues": repo.get("open_issues_count", 0),
                "watchers": repo.get("watchers_count", 0),
                "full_name": full_name,
                "archived": repo.get("archived", False),
                "size_kb": repo.get("size", 0),
            },
        )

    def _parse_trending_html(self, html: str) -> list[AITool]:
        """Parse the GitHub trending page HTML into AITool objects."""
        soup = BeautifulSoup(html, "html.parser")
        tools: list[AITool] = []

        for article in soup.select("article.Box-row"):
            link = article.select_one("h2 a")
            href = link.get("href") if link else None
            if not isinstance(href, str):
                continue
            parts = [p.strip() for p in href.strip().strip("/").split("/") if p.strip()]
            if len(parts) < 2:
                continue
            owner, repo_name = parts[0], parts[1]

            p_tag = article.select_one("p")
            description = p_tag.get_text(strip=True) if p_tag else ""

            stars = 0
            stars_el = article.select_one("a[href$='/stargazers']")
            if stars_el:
                stars_text = stars_el.get_text(strip=True).replace(",", "")
                stars = int(stars_text) if stars_text.isdigit() else 0

            lang_el = article.select_one("[itemprop='programmingLanguage']")
            language = lang_el.get_text(strip=True) if lang_el else None

            url = f"https://github.com/{owner}/{repo_name}"
            tools.append(
                AITool(
                    name=repo_name,
                    description=description[:500],
                    category=self._infer_category([], repo_name, description),
                    source_platform="github",
                    source_url=url,
                    source_code_url=url,
                    stars=stars,
                    language=language,
                    author=owner,
                    popularity_score=min(stars / 100_000, 1.0),
                )
            )

        return tools

    @staticmethod
    def _infer_category(topics: list[str], name: str, description: str) -> AICategory:
        """Infer a category from repository topics, then name and description."""
        all_text = " ".join(topics) + " " + name.lower() + " " + description.lower()

        for topic in topics:
            normalized = topic.lower().strip()
            if normalized in _TOPIC_CATEGORIES:
                return _TOPIC_CATEGORIES[normalized]

        patterns: list[tuple[str, AICategory]] = [
            (r"\bagent\b", AICategory.AGENT),
            (r"\bframework\b", AICategory.FRAMEWORK),
            (r"\bmodel\b", AICategory.MODEL),
            (r"\bdataset\b", AICategory.DATASET),
            (r"\bvector.?db\b|\brag\b|\bserving\b|\binference\b|\bdeploy\b", AICategory.INFRA),
            (r"\bcli\b|\bcommand.?line\b", AICategory.CLI_TOOL),
            (r"\blibrary\b|\bsdk\b|\bpackage\b", AICategory.LIBRARY),
            (r"\btool\b|\butil\b|\beval\b|\bbench\b", AICategory.DEV_TOOL),
        ]
        for pattern, cat in patterns:
            if re.search(pattern, all_text):
                return cat

        return AICategory.OTHER
