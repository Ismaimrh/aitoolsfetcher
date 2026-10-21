"""Abstract base class for destination adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aitoolsfetcher.core.models import AITool


class BaseDestination(ABC):
    """Base class for output destination adapters."""

    name: str = "base"

    @abstractmethod
    async def deliver(self, tools: list[AITool], **kwargs: Any) -> None:
        """Deliver tools to this destination.

        Args:
            tools: Tools to deliver.
            **kwargs: Destination-specific options.
        """
