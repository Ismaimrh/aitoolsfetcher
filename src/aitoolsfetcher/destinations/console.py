"""Console destination — rich table output for CLI."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from aitoolsfetcher.core.models import AITool, FetchResult
from aitoolsfetcher.destinations.base import BaseDestination


class ConsoleDestination(BaseDestination):
    """Display tools as a rich table in the terminal."""

    name = "console"

    def __init__(self, console: Console | None = None):
        self._console = console or Console()

    async def deliver(self, tools: list[AITool], **kwargs) -> None:
        """Render tools as a summary table with full clickable links."""
        detail = kwargs.get("detail", False)

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

        table.add_column("Name", style="bold white", no_wrap=True, ratio=1)
        table.add_column("Category", style="yellow", no_wrap=True, ratio=1)
        table.add_column("What it does", style="white", ratio=3)
        table.add_column("Stars", style="green", justify="right", no_wrap=True, ratio=1)
        table.add_column("Source", style="blue", no_wrap=False, ratio=3)

        for tool in tools:
            desc = tool.description[:100] + "..." if len(tool.description) > 100 else tool.description
            stars = f"{tool.stars:,}" if tool.stars else "-"
            # Full URL — rich/terminal handles clickability
            source = tool.source_code_or_source

            table.add_row(tool.name, tool.category.value, desc, stars, source)

        self._console.print(table)

    def _render_detail(self, tools: list[AITool]) -> None:
        """Expanded detail view — one panel per tool with full info."""
        for i, tool in enumerate(tools, 1):
            stars = f"{tool.stars:,}" if tool.stars else "N/A"
            lang = tool.language or "N/A"
            license_ = tool.license or "N/A"
            topics = ", ".join(tool.topics[:8]) if tool.topics else "N/A"

            lines = [
                f"[bold white]{tool.description}[/bold white]",
                "",
                f"[cyan]Category:[/cyan]    {tool.category.value}",
                f"[cyan]Author:[/cyan]      {tool.author}",
                f"[cyan]Stars:[/cyan]       {stars}",
                f"[cyan]Language:[/cyan]    {lang}",
                f"[cyan]License:[/cyan]     {license_}",
                f"[cyan]Topics:[/cyan]      {topics}",
                f"[cyan]Source:[/cyan]      {tool.source_code_or_source}",
            ]

            if tool.homepage_url:
                lines.append(f"[cyan]Homepage:[/cyan]    {tool.homepage_url}")

            if tool.source_code_url and tool.source_code_url != tool.source_url:
                lines.append(f"[cyan]Code:[/cyan]        {tool.source_code_url}")

            lines.append(f"[cyan]Relevance:[/cyan]   {tool.ai_relevance_score:.0%}")

            panel = Panel(
                "\n".join(lines),
                title=f"[bold]{tool.name}[/bold]",
                subtitle=f"[dim]{tool.source_platform} | {i}/{len(tools)}[/dim]",
                border_style="bright_blue",
                padding=(1, 2),
            )
            self._console.print(panel)

    def print_summary(self, result: FetchResult) -> None:
        """Print fetch summary line below the table."""
        self._console.print(f"\n[bold]{result.summary}[/bold]")
        if result.errors:
            for source, error in result.errors.items():
                self._console.print(f"  [red]✗ {source}: {error}[/red]")
