"""Console destination — rich table output for CLI."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from aitoolsfetcher.destinations.base import BaseDestination

if TYPE_CHECKING:
    from aitoolsfetcher.core.models import AITool, FetchResult


class ConsoleDestination(BaseDestination):
    """Display tools as a rich table in the terminal.

    All text that comes from a source (names, descriptions, topics) is escaped,
    so square brackets in a repo description can't break or restyle the output.
    """

    name = "console"

    def __init__(self, console: Console | None = None) -> None:
        """Create a console destination.

        Args:
            console: Rich console to print to. Defaults to a new stdout console.
        """
        self._console = console or Console()

    async def deliver(self, tools: list[AITool], **kwargs: Any) -> None:
        """Render tools as a table, or as one panel per tool.

        Args:
            tools: Tools to display.
            **kwargs: ``detail=True`` switches to the expanded panel view.
        """
        detail = bool(kwargs.get("detail", False))

        if not tools:
            self._console.print("[dim]No tools found.[/dim]")
            return

        if detail:
            self._render_detail(tools)
        else:
            self._render_table(tools)

    def _render_table(self, tools: list[AITool]) -> None:
        """Compact summary table with full links."""
        table = Table(
            title="AI Tools",
            show_header=True,
            header_style="bold cyan",
            border_style="dim",
            pad_edge=True,
            expand=True,
        )

        # Names and links wrap ("fold") instead of being cut off with an ellipsis
        table.add_column("Name", style="bold white", overflow="fold", ratio=1)
        table.add_column("Category", style="yellow", no_wrap=True, ratio=1)
        table.add_column("What it does", style="white", ratio=3)
        table.add_column("Stars", style="green", justify="right", no_wrap=True, ratio=1)
        table.add_column("Source", style="blue", overflow="fold", ratio=3)

        for tool in tools:
            desc = (
                tool.description[:100] + "..." if len(tool.description) > 100 else tool.description
            )
            stars = f"{tool.stars:,}" if tool.stars else "-"
            table.add_row(
                escape(tool.name),
                tool.category.value,
                escape(desc),
                stars,
                escape(tool.source_code_or_source),
            )

        self._console.print(table)

    def _render_detail(self, tools: list[AITool]) -> None:
        """Expanded detail view — one panel per tool with full info."""
        for i, tool in enumerate(tools, 1):
            stars = f"{tool.stars:,}" if tool.stars else "N/A"
            lang = escape(tool.language or "N/A")
            license_ = escape(tool.license or "N/A")
            topics = escape(", ".join(tool.topics[:8])) if tool.topics else "N/A"

            lines = [
                f"[bold white]{escape(tool.description)}[/bold white]",
                "",
                f"[cyan]Category:[/cyan]    {tool.category.value}",
                f"[cyan]Author:[/cyan]      {escape(tool.author)}",
                f"[cyan]Stars:[/cyan]       {stars}",
                f"[cyan]Language:[/cyan]    {lang}",
                f"[cyan]License:[/cyan]     {license_}",
                f"[cyan]Topics:[/cyan]      {topics}",
                f"[cyan]Source:[/cyan]      {escape(tool.source_code_or_source)}",
            ]

            if tool.homepage_url:
                lines.append(f"[cyan]Homepage:[/cyan]    {escape(tool.homepage_url)}")

            if tool.source_code_url and tool.source_code_url != tool.source_url:
                lines.append(f"[cyan]Code:[/cyan]        {escape(tool.source_code_url)}")

            lines.append(f"[cyan]Relevance:[/cyan]   {tool.ai_relevance_score:.0%}")

            panel = Panel(
                "\n".join(lines),
                title=f"[bold]{escape(tool.name)}[/bold]",
                subtitle=f"[dim]{escape(tool.source_platform)} | {i}/{len(tools)}[/dim]",
                border_style="bright_blue",
                padding=(1, 2),
            )
            self._console.print(panel)

    def print_summary(self, result: FetchResult) -> None:
        """Print the fetch summary line and any per-source errors.

        Args:
            result: The fetch result to summarize.
        """
        self._console.print(f"\n[bold]{escape(result.summary)}[/bold]")
        if result.errors:
            for source, error in result.errors.items():
                self._console.print(f"  [red]✗ {escape(source)}: {escape(error)}[/red]")
