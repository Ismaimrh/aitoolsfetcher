"""JSON file destination — save results to a JSON file."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aitoolsfetcher.destinations.base import BaseDestination

if TYPE_CHECKING:
    from aitoolsfetcher.core.models import AITool

logger = logging.getLogger(__name__)


class JSONFileDestination(BaseDestination):
    """Save tools to a single JSON file, appending or overwriting."""

    name = "json"

    def __init__(self, path: str | Path, append: bool = True) -> None:
        """Create a JSON file destination.

        Args:
            path: Target file. ``~`` is expanded; parent folders are created.
            append: If True, merge into the existing file and skip tools whose
                ``id`` is already there. If False, overwrite the file.
        """
        self._path = Path(path).expanduser().resolve()
        self._append = append

    async def deliver(self, tools: list[AITool], **kwargs: Any) -> None:
        """Write tools to the JSON file.

        Does nothing when ``tools`` is empty.

        Args:
            tools: Tools to write.
            **kwargs: Unused.

        Raises:
            ValueError: In append mode, if the existing file is not valid JSON.
                The file is left untouched rather than overwritten.
        """
        if not tools:
            return

        existing: list[Any] = []
        if self._append and self._path.exists():
            try:
                loaded = json.loads(self._path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as e:
                raise ValueError(f"{self._path} is not valid JSON; refusing to overwrite it") from e
            existing = loaded if isinstance(loaded, list) else [loaded]

        new_data: list[dict[str, Any]] = [tool.model_dump(mode="json") for tool in tools]

        if self._append:
            existing_ids = {item.get("id") for item in existing if isinstance(item, dict)}
            new_data = [d for d in new_data if d.get("id") not in existing_ids]
            output = existing + new_data
        else:
            output = list(new_data)

        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
        logger.info("Saved %d tools to %s", len(new_data), self._path)
