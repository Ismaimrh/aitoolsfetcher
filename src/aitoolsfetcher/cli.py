"""CLI entry point for AIToolsFetcher."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import click
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from aitoolsfetcher.config import load_settings
from aitoolsfetcher.core.engine import AIToolsFetcher
from aitoolsfetcher.destinations.console import ConsoleDestination
from aitoolsfetcher.formatters import FILE_FORMATS, render

if TYPE_CHECKING:
    from collections.abc import Coroutine

    from aitoolsfetcher.core.models import AITool, FetchResult
    from aitoolsfetcher.destinations.base import BaseDestination

T = TypeVar("T")

console = Console()

_FORMAT_CHOICE = click.Choice(["table", *FILE_FORMATS])


def _run_async(coro: Coroutine[Any, Any, T]) -> T:
    """Run a coroutine from the synchronous CLI, even inside a running loop (e.g. Jupyter)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor() as pool:
        return pool.submit(asyncio.run, coro).result()


def _setup_logging(verbose: bool) -> None:
    """Configure stdlib logging: debug with --verbose, warnings otherwise."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _split_sources(sources: str | None) -> list[str] | None:
    """Turn ``"github, reddit"`` into ``["github", "reddit"]``."""
    if not sources:
        return None
    return [s.strip() for s in sources.split(",") if s.strip()]


def _resolve_destination(dest: str | None, fmt: str = "json") -> BaseDestination | None:
    """Parse a ``--dest`` value such as ``folder:~/results`` into a destination.

    Args:
        dest: The option value, or None.
        fmt: Output format; used by the folder destination when it is a file format.

    Returns:
        The destination, or None if ``dest`` is empty.

    Raises:
        click.BadParameter: If the value is malformed or the type is unknown.
    """
    if not dest:
        return None

    if ":" not in dest:
        raise click.BadParameter(
            f"Invalid destination: {dest}. Use type:path, e.g. folder:~/results",
            param_hint="--dest",
        )

    dtype, dpath = dest.split(":", 1)
    dtype = dtype.strip().lower()

    if dtype == "folder":
        from aitoolsfetcher.destinations.folder import FolderDestination

        return FolderDestination(dpath, fmt=fmt if fmt in FILE_FORMATS else "json")
    if dtype == "json":
        from aitoolsfetcher.destinations.json_file import JSONFileDestination

        return JSONFileDestination(dpath)
    raise click.BadParameter(
        f"Unknown destination type: {dtype}. Available: folder, json", param_hint="--dest"
    )


async def _show(result: FetchResult, fmt: str, detail: bool) -> None:
    """Print a fetch result as a table (with summary) or as json/csv/md text."""
    if fmt == "table":
        con = ConsoleDestination(console)
        await con.deliver(result.tools, detail=detail)
        con.print_summary(result)
    else:
        click.echo(render(result.tools, fmt))
        for source, error in result.errors.items():
            click.echo(f"error: {source}: {error}", err=True)


async def _deliver(
    destination: BaseDestination | None, dest: str | None, tools: list[AITool]
) -> None:
    """Send tools to the --dest destination, if one was given."""
    if destination:
        await destination.deliver(tools)
        console.print(f"[green]Results saved to {escape(dest or '')}[/green]", highlight=False)


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging.")
@click.version_option(package_name="aitoolsfetcher")
@click.pass_context
def main(ctx: click.Context, verbose: bool) -> None:
    """AIToolsFetcher — discover trending AI tools on GitHub."""
    _setup_logging(verbose)
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose


@main.command()
@click.argument("query")
@click.option(
    "--sources", "-s", default=None, help="Comma-separated sources (default: all enabled)."
)
@click.option("--limit", "-l", default=20, type=click.IntRange(min=1), help="Max results.")
@click.option("--min-stars", default=0, type=click.IntRange(min=0), help="Minimum stars.")
@click.option(
    "--min-score",
    default=0.0,
    type=click.FloatRange(0, 1),
    help="Minimum AI relevance score (0-1).",
)
@click.option("--format", "-f", "fmt", default="table", type=_FORMAT_CHOICE, help="Output format.")
@click.option("--dest", default=None, help="Also save to a destination: folder:PATH or json:FILE.")
@click.option("--detail", "-d", is_flag=True, help="Show one panel per tool instead of a table.")
def search(
    query: str,
    sources: str | None,
    limit: int,
    min_stars: int,
    min_score: float,
    fmt: str,
    dest: str | None,
    detail: bool,
) -> None:
    """Search for AI tools matching QUERY."""
    destination = _resolve_destination(dest, fmt)

    async def _search() -> None:
        async with AIToolsFetcher(settings=load_settings()) as fetcher:
            result = await fetcher.search(
                query=query,
                sources=_split_sources(sources),
                limit=limit,
                min_stars=min_stars,
                min_relevance=min_score,
            )
            await _show(result, fmt, detail)
            await _deliver(destination, dest, result.tools)

    _run_async(_search())


@main.command()
@click.option("--sources", "-s", default=None, help="Comma-separated sources.")
@click.option("--limit", "-l", default=20, type=click.IntRange(min=1), help="Max results.")
@click.option("--format", "-f", "fmt", default="table", type=_FORMAT_CHOICE, help="Output format.")
@click.option("--dest", default=None, help="Also save to a destination: folder:PATH or json:FILE.")
@click.option("--detail", "-d", is_flag=True, help="Show one panel per tool instead of a table.")
def trending(sources: str | None, limit: int, fmt: str, dest: str | None, detail: bool) -> None:
    """Show this week's trending AI repositories on GitHub."""
    destination = _resolve_destination(dest, fmt)

    async def _trending() -> None:
        async with AIToolsFetcher(settings=load_settings()) as fetcher:
            result = await fetcher.trending(sources=_split_sources(sources), limit=limit)
            await _show(result, fmt, detail)
            await _deliver(destination, dest, result.tools)

    _run_async(_trending())


@main.command()
@click.option("--limit", "-l", default=50, type=click.IntRange(min=1), help="Max results.")
@click.option("--detail", "-d", is_flag=True, help="Show one panel per tool instead of a table.")
def history(limit: int, detail: bool) -> None:
    """Show tools saved in the local cache, newest first."""

    async def _history() -> None:
        async with AIToolsFetcher(settings=load_settings()) as fetcher:
            tools = await fetcher.history(limit=limit)
            await ConsoleDestination(console).deliver(tools, detail=detail)
            console.print(f"\n[dim]{len(tools)} tools shown from cache[/dim]")

    _run_async(_history())


def _run_scheduler(coro: Coroutine[Any, Any, None]) -> None:
    """Run a blocking scheduler coroutine and exit quietly on Ctrl+C."""
    try:
        _run_async(coro)
    except KeyboardInterrupt:
        console.print("\n[dim]Scheduler stopped.[/dim]")


@main.command()
@click.argument("query")
@click.option("--every", required=True, help="Interval, e.g. 30m, 6h, 1d.")
@click.option("--sources", "-s", default=None, help="Comma-separated sources.")
@click.option("--limit", "-l", default=20, type=click.IntRange(min=1), help="Max results per run.")
@click.option("--dest", required=True, help="Destination: folder:PATH or json:FILE.")
def periodic(query: str, every: str, sources: str | None, limit: int, dest: str) -> None:
    """Run QUERY now and then every interval; deliver only new tools."""
    from aitoolsfetcher.core.scheduler import FetchJob, Scheduler, parse_interval

    try:
        parse_interval(every)
    except ValueError as e:
        raise click.BadParameter(str(e), param_hint="--every") from e
    destination = _resolve_destination(dest)

    async def _periodic() -> None:
        async with AIToolsFetcher(settings=load_settings()) as fetcher:
            job = FetchJob(
                query=query, sources=_split_sources(sources), limit=limit, destination=destination
            )
            scheduler = Scheduler(fetcher)

            console.print(f"[bold]Periodic fetch: '{escape(query)}' every {escape(every)}[/bold]")
            result = await job.execute(fetcher)
            await _show(result, "table", detail=False)

            await scheduler.add_periodic(job, every)
            console.print(f"\n[dim]Next fetch in {escape(every)}. Press Ctrl+C to stop.[/dim]")
            await scheduler.start()

    _run_scheduler(_periodic())


@main.command()
@click.argument("query")
@click.option(
    "--cron", required=True, help="5-field cron expression, e.g. '0 9 * * MON' (local time)."
)
@click.option("--sources", "-s", default=None, help="Comma-separated sources.")
@click.option("--limit", "-l", default=20, type=click.IntRange(min=1), help="Max results per run.")
@click.option("--dest", required=True, help="Destination: folder:PATH or json:FILE.")
def schedule(query: str, cron: str, sources: str | None, limit: int, dest: str) -> None:
    """Run QUERY on a cron schedule; deliver only new tools."""
    from aitoolsfetcher.core.scheduler import FetchJob, Scheduler, parse_cron

    try:
        parse_cron(cron)
    except ValueError as e:
        raise click.BadParameter(str(e), param_hint="--cron") from e
    destination = _resolve_destination(dest)

    async def _schedule() -> None:
        async with AIToolsFetcher(settings=load_settings()) as fetcher:
            job = FetchJob(
                query=query, sources=_split_sources(sources), limit=limit, destination=destination
            )
            scheduler = Scheduler(fetcher)
            await scheduler.add_cron(job, cron)
            console.print(f"[bold]Scheduled: '{escape(query)}' at '{escape(cron)}'[/bold]")
            console.print("[dim]Press Ctrl+C to stop.[/dim]")
            await scheduler.start()

    _run_scheduler(_schedule())


@main.command(name="sources")
def list_sources() -> None:
    """List sources and whether they are available."""
    settings = load_settings()

    all_sources = {
        "github": ("GitHub API", "Primary source, most reliable"),
        "huggingface": ("HuggingFace API", "Models and spaces"),
        "reddit": ("Reddit (scraping)", "r/MachineLearning, r/LocalLLaMA, etc."),
        "arxiv": ("arXiv API", "Research papers with code"),
        "producthunt": ("Product Hunt (scraping)", "New product launches"),
        "twitter": ("X/Twitter (scraping)", "Fragile, disabled by default"),
        "linkedin": ("LinkedIn (scraping)", "Fragile, disabled by default"),
    }

    table = Table(title="Available Sources", header_style="bold cyan")
    table.add_column("Source", style="bold")
    table.add_column("Type")
    table.add_column("Status")
    table.add_column("Notes")

    for name, (label, notes) in all_sources.items():
        if name == "github":
            status = "[green]Active[/green]"
        elif not settings.is_source_enabled(name):
            status = "[dim]Disabled[/dim]"
        else:
            status = "[yellow]Coming soon[/yellow]"
        table.add_row(name, label, status, notes)

    console.print(table)


@main.command()
@click.option("--format", "-f", "fmt", default="json", type=click.Choice(FILE_FORMATS))
@click.option("--output", "-o", required=True, type=click.Path(dir_okay=False), help="Output file.")
@click.option("--limit", "-l", default=100, type=click.IntRange(min=1), help="Max tools to export.")
def export(fmt: str, output: str, limit: int) -> None:
    """Export cached tools to a JSON, CSV or Markdown file."""

    async def _export() -> None:
        async with AIToolsFetcher(settings=load_settings()) as fetcher:
            tools = await fetcher.history(limit=limit)
        if not tools:
            console.print("[dim]No tools in cache to export.[/dim]")
            return
        path = Path(output).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render(tools, fmt), encoding="utf-8")
        console.print(
            f"[green]Exported {len(tools)} tools to {escape(str(path))}[/green]", highlight=False
        )

    _run_async(_export())


if __name__ == "__main__":
    main()
