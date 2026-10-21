"""Abstract base class for all source adapters."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from aitoolsfetcher.core.models import AITool, SearchFilters

logger = logging.getLogger(__name__)

try:
    _VERSION = version("aitoolsfetcher")
except PackageNotFoundError:  # pragma: no cover - running from an uninstalled checkout
    _VERSION = "0.0.0"

#: Identifies the client to every API it calls, with a link back to the project.
USER_AGENT = f"aitoolsfetcher/{_VERSION} (+https://github.com/Ismaimrh/aitoolsfetcher)"


class BaseSource(ABC):
    """Base class for platform source adapters.

    Each adapter is responsible for:
    - Fetching tools from its platform
    - Handling its own rate limiting and error recovery
    - Converting platform-specific data into AITool objects
    """

    name: str = "base"
    requires_auth: bool = False

    def __init__(self, client: httpx.AsyncClient | None = None, **kwargs: Any) -> None:
        """Create the adapter.

        Args:
            client: Optional shared HTTP client. If omitted, the adapter creates
                and owns its own client.
            **kwargs: Ignored; lets subclasses accept extra options.
        """
        self._client = client
        self._owns_client = client is None

    async def get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client.

        Returns:
            An ``httpx.AsyncClient`` with a timeout and User-Agent set.
        """
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=30.0,
                follow_redirects=True,
                headers={"User-Agent": USER_AGENT},
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client if this adapter created it."""
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    @abstractmethod
    async def search(self, filters: SearchFilters) -> list[AITool]:
        """Search for AI tools matching the given filters.

        Args:
            filters: Query, limit and other search filters.

        Returns:
            Tools found on this platform.
        """

    @abstractmethod
    async def trending(self, limit: int = 20) -> list[AITool]:
        """Fetch currently trending AI tools from this platform.

        Args:
            limit: Maximum number of tools to return.

        Returns:
            Trending tools on this platform.
        """

    async def traceback_source_code(self, tool: AITool) -> str | None:
        """Find a source code URL for a tool found on a non-code platform.

        Args:
            tool: The tool to trace back.

        Returns:
            A source code URL, or None. The default implementation returns None.
        """
        return None

    async def __aenter__(self) -> BaseSource:
        """Enter the async context."""
        return self

    async def __aexit__(self, *exc: object) -> None:
        """Close the client on exit."""
        await self.close()
