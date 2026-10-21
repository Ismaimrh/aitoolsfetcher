"""Render tools as JSON, CSV or Markdown text.

Shared by the CLI (``--format``, ``export``) and the file-based destinations so
every output path produces the same content.
"""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aitoolsfetcher.core.models import AITool

#: Output formats that can be written to a file or stdout.
FILE_FORMATS = ("json", "csv", "md")

CSV_COLUMNS = ["name", "category", "description", "stars", "language", "source", "relevance"]


def to_json(tools: list[AITool]) -> str:
    """Serialize tools as a pretty-printed JSON array.

    Args:
        tools: Tools to serialize.

    Returns:
        JSON text containing every field of each tool.
    """
    data = [tool.model_dump(mode="json") for tool in tools]
    return json.dumps(data, indent=2, default=str)


def to_csv(tools: list[AITool]) -> str:
    """Serialize tools as CSV with one row per tool.

    Args:
        tools: Tools to serialize.

    Returns:
        CSV text with a header row.
    """
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(CSV_COLUMNS)
    for tool in tools:
        writer.writerow(
            [
                tool.name,
                tool.category.value,
                tool.description[:200],
                tool.stars if tool.stars is not None else "",
                tool.language or "",
                tool.source_code_or_source,
                f"{tool.ai_relevance_score:.3f}",
            ]
        )
    return buf.getvalue()


def to_markdown(tools: list[AITool], now: datetime | None = None) -> str:
    """Render tools as a Markdown list with links.

    Args:
        tools: Tools to render.
        now: Timestamp for the heading. Defaults to the current UTC time.

    Returns:
        Markdown text.
    """
    stamp = (now or datetime.now(tz=timezone.utc)).strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"# AI Tools — {stamp}", ""]
    for tool in tools:
        stars = f"{tool.stars:,}" if tool.stars is not None else "N/A"
        lines.append(f"## [{tool.name}]({tool.source_code_or_source})")
        lines.append("")
        lines.append(
            f"**Category:** {tool.category.value} | **Stars:** {stars} | "
            f"**Language:** {tool.language or 'N/A'}"
        )
        lines.append("")
        if tool.description:
            lines.append(tool.description)
            lines.append("")
    return "\n".join(lines)


def render(tools: list[AITool], fmt: str) -> str:
    """Render tools in the given file format.

    Args:
        tools: Tools to render.
        fmt: One of ``"json"``, ``"csv"`` or ``"md"``.

    Returns:
        The rendered text.

    Raises:
        ValueError: If the format is not supported.
    """
    if fmt == "json":
        return to_json(tools)
    if fmt == "csv":
        return to_csv(tools)
    if fmt == "md":
        return to_markdown(tools)
    raise ValueError(f"Unsupported format: {fmt}. Use one of: {', '.join(FILE_FORMATS)}")
