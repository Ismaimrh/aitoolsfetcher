"""Abstract base class for AI tool classifiers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from aitoolsfetcher.core.models import AICategory, AITool


class ClassificationResult(NamedTuple):
    """Result of classifying a tool."""

    is_ai_tool: bool
    category: AICategory
    relevance_score: float
    summary: str


class BaseClassifier(ABC):
    """Base class for tool classifiers."""

    @abstractmethod
    async def classify(self, tool: AITool) -> ClassificationResult:
        """Classify a single tool."""
        ...

    async def classify_batch(self, tools: list[AITool]) -> list[ClassificationResult]:
        """Classify multiple tools. Default: sequential. Override for batch optimization."""
        return [await self.classify(tool) for tool in tools]
