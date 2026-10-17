"""GitHub source adapter — primary and most reliable data source."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import httpx
from bs4 import BeautifulSoup

from aitoolsfetcher.core.models import AICategory, AITool, SearchFilters
from aitoolsfetcher.sources.base import BaseSource

logger = logging.getLogger(__name__)

_API_BASE = "https://api.github.com"
_TRENDING_URL = "https://github.com/trending"

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

# AI-related search qualifiers for GitHub API
_AI_TOPICS = [
    "llm", "ai", "machine-learning", "deep-learning", "gpt", "transformer",
    "langchain", "llamaindex", "autogen", "crewai", "agent", "rag",
    "vector-database", "fine-tuning", "prompt-engineering", "diffusion",
    "neural-network", "nlp", "computer-vision", "generative-ai",
    "large-language-model", "embedding", "huggingface", "pytorch", "tensorflow",
    "mcp", "function-calling", "tool-use",
]


class GitHubSource(BaseSource):
    """Fetch AI tools from GitHub via API and trending page scraping."""

    name = "github"
    requires_auth = False  # Works without auth, better with token

    def __init__(
        self,
        token: str = "",
        client: Optional[httpx.AsyncClient] = None,
    ):
        super().__init__(client=client)
        self._token = token

    async def get_client(self) -> httpx.AsyncClient:
        """Get client with GitHub auth headers if token available."""
        client = await super().get_client()
        if self._token and "Authorization" not in client.headers:
            client.headers["Authorization"] = f"Bearer {self._token}"
            client.headers["Accept"] = "application/vnd.github+json"
            client.headers["X-GitHub-Api-Version"] = "2022-11-28"
        return client

    async def search(self, filters: SearchFilters) -> list[AITool]:
        """Search GitHub repos matching query with AI topic filtering."""
        client = await self.get_client()
        query = self._build_search_query(filters)
        tools: list[AITool] = []
        page = 1
        remaining = filters.limit

        while remaining > 0:
            per_page = min(remaining, 30)  # GitHub max per page
            try:
                resp = await client.get(
                    f"{_API_BASE}/search/repositories",
                    params={
                        "q": query,
                        "sort": "stars",
                        "order": "desc",
                        "per_page": per_page,
                        "page": page,
                    },
                )
                resp.raise_for_status()
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 403:
                    logger.warning("GitHub API rate limit reached")
                    break
                logger.error("GitHub API error: %s", e)
                break
            except httpx.HTTPError as e:
                logger.error("GitHub request failed: %s", e)
                break

            data = resp.json()
            items = data.get("items", [])
            if not items:
                break

            for item in items:
                tool = self._repo_to_tool(item)
                if tool and (not filters.min_stars or (tool.stars or 0) >= filters.min_stars):
                    tools.append(tool)

            remaining -= len(items)
            page += 1

            if len(items) < per_page:
                break

        return tools[:filters.limit]

    async def trending(self, limit: int = 20) -> list[AITool]:
        """Scrape GitHub trending page for AI-related repos."""
        client = await self.get_client()
        tools: list[AITool] = []

        for language in ("", "python", "javascript", "typescript", "rust", "go"):
            try:
                params: dict[str, str] = {"since": "weekly"}
                if language:
                    params["spoken_language_code"] = ""
                url = f"{_TRENDING_URL}/{language}" if language else _TRENDING_URL
                resp = await client.get(url, params=params)
                resp.raise_for_status()
            except httpx.HTTPError as e:
                logger.warning("Failed to fetch trending page (%s): %s", language or "all", e)
                continue

            parsed = self._parse_trending_html(resp.text)
            tools.extend(parsed)

            if len(tools) >= limit * 2:  # Fetch extra, will filter later
                break

        # Deduplicate by repo full name
        seen: set[str] = set()
        unique: list[AITool] = []
        for tool in tools:
            key = tool.source_url
            if key not in seen:
                seen.add(key)
                unique.append(tool)

        return unique[:limit]

    def _build_search_query(self, filters: SearchFilters) -> str:
        """Build GitHub search query string from filters."""
        parts: list[str] = []

        if filters.query:
            parts.append(filters.query)

        # Add AI topic qualifier if query doesn't already contain AI terms
        query_lower = filters.query.lower()
        has_ai_term = any(t in query_lower for t in _AI_TOPICS)
        if not has_ai_term:
            # Use a single broad topic to avoid complex OR logic issues
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

    def _repo_to_tool(self, repo: dict[str, Any]) -> Optional[AITool]:
        """Convert GitHub API repo object to AITool."""
        name = repo.get("name", "")
        description = repo.get("description") or ""
        topics = repo.get("topics", [])
        full_name = repo.get("full_name", "")

        if not name:
            return None

        category = self._infer_category(topics, name, description)

        last_updated = None
        if repo.get("pushed_at"):
            try:
                last_updated = datetime.fromisoformat(repo["pushed_at"].replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass

        stars = repo.get("stargazers_count", 0)
        max_stars = 100_000  # Normalize against a high baseline
        popularity = min(stars / max_stars, 1.0) if stars else 0.0

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
        """Parse GitHub trending page HTML into AITool objects."""
        soup = BeautifulSoup(html, "html.parser")
        tools: list[AITool] = []

        for article in soup.select("article.Box-row"):
            try:
                # Repo name and link
                h2 = article.select_one("h2 a")
                if not h2:
                    continue
                href = h2.get("href", "").strip()
                parts = [p.strip() for p in href.strip("/").split("/") if p.strip()]
                if len(parts) < 2:
                    continue
                owner, repo_name = parts[0], parts[1]

                # Description
                p_tag = article.select_one("p")
                description = p_tag.get_text(strip=True) if p_tag else ""

                # Stars
                stars_el = article.select_one("a[href$='/stargazers']")
                stars = 0
                if stars_el:
                    stars_text = stars_el.get_text(strip=True).replace(",", "")
                    stars = int(stars_text) if stars_text.isdigit() else 0

                # Language
                lang_el = article.select_one("[itemprop='programmingLanguage']")
                language = lang_el.get_text(strip=True) if lang_el else None

                url = f"https://github.com/{owner}/{repo_name}"

                tools.append(AITool(
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
                ))
            except (AttributeError, ValueError, IndexError) as e:
                logger.debug("Failed to parse trending item: %s", e)
                continue

        return tools

    @staticmethod
    def _infer_category(topics: list[str], name: str, description: str) -> AICategory:
        """Infer AI category from repo metadata."""
        all_text = " ".join(topics) + " " + name.lower() + " " + description.lower()

        # Check topics first (most reliable)
        for topic in topics:
            normalized = topic.lower().strip()
            if normalized in _TOPIC_CATEGORIES:
                return _TOPIC_CATEGORIES[normalized]

        # Keyword-based fallback on name + description
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
