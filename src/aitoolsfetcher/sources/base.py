"""Abstract base class for all source adapters."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

import httpx

from aitoolsfetcher.core.models import AITool, SearchFilters

logger = logging.getLogger(__name__)


class BaseSource(ABC):
    """Base class for platform source adapters.

    Each adapter is responsible for:
    - Fetching tools from its platform
    - Handling its own rate limiting and error recovery
    - Converting platform-specific data into AITool objects
    """

    name: str = "base"
    requires_auth: bool = False

    def __init__(self, client: Optional[httpx.AsyncClient] = None, **kwargs):
        self._client = client
        self._owns_client = client is None

    async def get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=30.0,
                follow_redirects=True,
                headers={"User-Agent": "AIToolsFetcher/0.1.0"},
            )
        return self._client

    async def close(self) -> None:
        """Close owned HTTP client."""
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    @abstractmethod
    async def search(self, filters: SearchFilters) -> list[AITool]:
        """Search for AI tools matching the given filters."""
        ...

    @abstractmethod
    async def trending(self, limit: int = 20) -> list[AITool]:
        """Fetch currently trending AI tools from this platform."""
        ...

    async def traceback_source_code(self, tool: AITool) -> Optional[str]:
        """Attempt to find source code URL for a tool discovered on a non-code platform.

        Default implementation returns None. Override in scrapers.
        """
        return None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.close()
