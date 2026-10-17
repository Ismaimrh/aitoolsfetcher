"""Abstract base class for destination adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod

from aitoolsfetcher.core.models import AITool


class BaseDestination(ABC):
    """Base class for output destination adapters."""

    name: str = "base"

    @abstractmethod
    async def deliver(self, tools: list[AITool], **kwargs) -> None:
        """Deliver tools to this destination."""
        ...
