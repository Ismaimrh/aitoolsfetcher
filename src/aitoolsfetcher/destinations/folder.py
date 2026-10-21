"""Folder destination — save each fetch as a timestamped file."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aitoolsfetcher.destinations.base import BaseDestination
from aitoolsfetcher.formatters import FILE_FORMATS, render

if TYPE_CHECKING:
    from aitoolsfetcher.core.models import AITool

logger = logging.getLogger(__name__)


class FolderDestination(BaseDestination):
    """Save tools to a folder, one timestamped file per delivery."""

    name = "folder"

    def __init__(self, path: str | Path, fmt: str = "json") -> None:
        """Create a folder destination.

        Args:
            path: Target directory. ``~`` is expanded; it is created on first write.
            fmt: File format, one of ``json``, ``csv`` or ``md``.

        Raises:
            ValueError: If the format is not supported.
        """
        if fmt not in FILE_FORMATS:
            raise ValueError(f"Unsupported format: {fmt}. Use one of: {', '.join(FILE_FORMATS)}")
        self._path = Path(path).expanduser().resolve()
        self._format = fmt
        self.last_written: Path | None = None

    async def deliver(self, tools: list[AITool], **kwargs: Any) -> None:
        """Save tools as a new timestamped file in the folder.

        Does nothing when ``tools`` is empty.

        Args:
            tools: Tools to save.
            **kwargs: Unused.
        """
        if not tools:
            return

        self._path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        filepath = self._path / f"ai_tools_{timestamp}.{self._format}"
        # Never overwrite an earlier delivery made within the same second
        counter = 1
        while filepath.exists():
            filepath = self._path / f"ai_tools_{timestamp}_{counter}.{self._format}"
            counter += 1

        filepath.write_text(render(tools, self._format), encoding="utf-8")
        self.last_written = filepath
        logger.info("Saved %d tools to %s", len(tools), filepath)
