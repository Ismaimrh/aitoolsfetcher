"""JSON file destination — save results to a JSON file."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from aitoolsfetcher.core.models import AITool
from aitoolsfetcher.destinations.base import BaseDestination

logger = logging.getLogger(__name__)


class JSONFileDestination(BaseDestination):
    """Save tools as a JSON file."""

    name = "json"

    def __init__(self, path: str | Path, append: bool = True):
        self._path = Path(path).expanduser().resolve()
        self._append = append

    async def deliver(self, tools: list[AITool], **kwargs) -> None:
        """Write tools to JSON file."""
        if not tools:
            return

        existing: list[dict] = []
        if self._append and self._path.exists():
            try:
                with open(self._path) as f:
                    existing = json.load(f)
                if not isinstance(existing, list):
                    existing = [existing]
            except (json.JSONDecodeError, OSError):
                existing = []

        new_data = [tool.model_dump(mode="json") for tool in tools]

        if self._append:
            # Deduplicate by ID
            existing_ids = {item.get("id") for item in existing if isinstance(item, dict)}
            new_data = [d for d in new_data if d.get("id") not in existing_ids]
            existing.extend(new_data)
            output = existing
        else:
            output = new_data

        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w") as f:
            json.dump(output, f, indent=2, default=str)

        logger.info("Saved %d tools to %s", len(new_data), self._path)
