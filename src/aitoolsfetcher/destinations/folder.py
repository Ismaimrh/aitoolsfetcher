"""Folder destination — save each fetch as a timestamped file."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from aitoolsfetcher.core.models import AITool
from aitoolsfetcher.destinations.base import BaseDestination

logger = logging.getLogger(__name__)


class FolderDestination(BaseDestination):
    """Save tools to a folder, one file per fetch."""

    name = "folder"

    def __init__(self, path: str | Path, format: str = "json"):
        self._path = Path(path).expanduser().resolve()
        self._format = format

    async def deliver(self, tools: list[AITool], **kwargs) -> None:
        """Save tools as a timestamped file in the destination folder."""
        if not tools:
            return

        self._path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"ai_tools_{timestamp}.{self._format}"
        filepath = self._path / filename

        if self._format == "json":
            data = [tool.model_dump(mode="json") for tool in tools]
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2, default=str)
        elif self._format == "md":
            lines = [f"# AI Tools — {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M')}\n"]
            for tool in tools:
                lines.append(f"## [{tool.name}]({tool.source_code_or_source})")
                lines.append(f"**Category:** {tool.category.value} | "
                             f"**Stars:** {tool.stars or 'N/A'} | "
                             f"**Language:** {tool.language or 'N/A'}")
                lines.append(f"\n{tool.description}\n")
            with open(filepath, "w") as f:
                f.write("\n".join(lines))
        elif self._format == "csv":
            import csv
            with open(filepath, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["name", "category", "description", "stars", "source", "language"])
                for tool in tools:
                    writer.writerow([
                        tool.name, tool.category.value, tool.description[:200],
                        tool.stars or "", tool.source_code_or_source, tool.language or "",
                    ])
        else:
            raise ValueError(f"Unsupported format: {self._format}")

        logger.info("Saved %d tools to %s", len(tools), filepath)
