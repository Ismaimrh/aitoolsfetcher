"""CLI entry point for AIToolsFetcher."""

from __future__ import annotations

import asyncio
import logging
import sys
from typing import Optional

import click
from rich.console import Console

from aitoolsfetcher.config import load_settings
from aitoolsfetcher.core.engine import AIToolsFetcher
from aitoolsfetcher.destinations.console import ConsoleDestination

console = Console()


def _run_async(coro):
    """Run async code from sync CLI context."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Fallback for nested event loops (e.g., Jupyter)
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _resolve_destination(dest: Optional[str]):
    """Parse --dest flag into a destination adapter."""
    if not dest:
        return None

    if ":" not in dest:
        raise click.BadParameter(f"Invalid destination format: {dest}. Use type:path (e.g., folder:~/results)")

    dtype, dpath = dest.split(":", 1)
    dtype = dtype.strip().lower()

    if dtype == "folder":
        from aitoolsfetcher.destinations.folder import FolderDestination
        return FolderDestination(dpath)
    elif dtype == "json":
        from aitoolsfetcher.destinations.json_file import JSONFileDestination
        return JSONFileDestination(dpath)
    else:
        raise click.BadParameter(f"Unknown destination type: {dtype}. Available: folder, json")


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging.")
@click.version_option(package_name="aitoolsfetcher")
@click.pass_context
def main(ctx: click.Context, verbose: bool) -> None:
    """AIToolsFetcher — Discover trending AI tools across multiple platforms."""
    _setup_logging(verbose)
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose


@main.command()
@click.argument("query")
@click.option("--sources", "-s", default=None, help="Comma-separated sources (e.g., github,reddit).")
@click.option("--limit", "-l", default=20, type=int, help="Max results.")
@click.option("--min-stars", default=0, type=int, help="Minimum stars.")
@click.option("--min-score", default=0.0, type=float, help="Minimum AI relevance score (0-1).")
@click.option("--format", "-f", "fmt", default="table", type=click.Choice(["table", "json", "csv", "md"]))
@click.option("--dest", default=None, help="Destination (e.g., folder:~/results, json:~/tools.json).")
@click.option("--no-cache", is_flag=True, help="Skip cache, fetch fresh results.")
@click.option("--detail", "-d", is_flag=True, help="Show expanded detail view per tool.")
def search(
    query: str,
    sources: Optional[str],
    limit: int,
    min_stars: int,
    min_score: float,
    fmt: str,
    dest: Optional[str],
    no_cache: bool,
    detail: bool,
) -> None:
    """Search for AI tools matching a query."""
    source_list = [s.strip() for s in sources.split(",")] if sources else None
    destination = _resolve_destination(dest)

    async def _search():
        settings = load_settings()
        async with AIToolsFetcher(settings=settings) as fetcher:
            result = await fetcher.search(
                query=query,
                sources=source_list,
                limit=limit,
                min_stars=min_stars,
                min_relevance=min_score,
            )

            # Display results
            if fmt == "table" and not dest:
                con = ConsoleDestination(console)
                await con.deliver(result.tools, detail=detail)
                con.print_summary(result)
            elif fmt == "json":
                import json
                data = [t.model_dump(mode="json") for t in result.tools]
                console.print_json(json.dumps(data, default=str))

            # Also deliver to destination if specified
            if destination:
                await destination.deliver(result.tools)
                console.print(f"[green]Results saved to destination: {dest}[/green]")

    _run_async(_search())


@main.command()
@click.option("--sources", "-s", default=None, help="Comma-separated sources.")
@click.option("--limit", "-l", default=20, type=int, help="Max results.")
@click.option("--dest", default=None, help="Destination.")
@click.option("--detail", "-d", is_flag=True, help="Show expanded detail view per tool.")
def trending(sources: Optional[str], limit: int, dest: Optional[str], detail: bool) -> None:
    """Fetch currently trending AI tools."""
    source_list = [s.strip() for s in sources.split(",")] if sources else None
    destination = _resolve_destination(dest)

    async def _trending():
        settings = load_settings()
        async with AIToolsFetcher(settings=settings) as fetcher:
            result = await fetcher.trending(sources=source_list, limit=limit)

            con = ConsoleDestination(console)
            await con.deliver(result.tools, detail=detail)
            con.print_summary(result)

            if destination:
                await destination.deliver(result.tools)
                console.print(f"[green]Results saved to destination: {dest}[/green]")

    _run_async(_trending())


@main.command()
@click.option("--limit", "-l", default=50, type=int, help="Max results.")
@click.option("--detail", "-d", is_flag=True, help="Show expanded detail view per tool.")
def history(limit: int, detail: bool) -> None:
    """View previously fetched tools from cache."""
    async def _history():
        settings = load_settings()
        async with AIToolsFetcher(settings=settings) as fetcher:
            tools = await fetcher.history(limit=limit)
            con = ConsoleDestination(console)
            await con.deliver(tools, detail=detail)
            console.print(f"\n[dim]{len(tools)} tools in cache[/dim]")

    _run_async(_history())


@main.command()
@click.argument("query")
@click.option("--every", required=True, help="Interval (e.g., 6h, 30m, 1d).")
@click.option("--sources", "-s", default=None, help="Comma-separated sources.")
@click.option("--limit", "-l", default=20, type=int, help="Max results per fetch.")
@click.option("--dest", required=True, help="Destination (e.g., folder:~/results).")
def periodic(
    query: str,
    every: str,
    sources: Optional[str],
    limit: int,
    dest: str,
) -> None:
    """Set up interval-based fetching."""
    source_list = [s.strip() for s in sources.split(",")] if sources else None
    destination = _resolve_destination(dest)
    if not destination:
        console.print("[red]Destination required for periodic fetching[/red]")
        sys.exit(1)

    from aitoolsfetcher.core.scheduler import FetchJob, Scheduler

    async def _periodic():
        settings = load_settings()
        async with AIToolsFetcher(settings=settings) as fetcher:
            job = FetchJob(query=query, sources=source_list, limit=limit, destination=destination)
            scheduler = Scheduler(fetcher)

            # Execute once immediately
            console.print(f"[bold]Starting periodic fetch: '{query}' every {every}[/bold]")
            result = await job.execute(fetcher)
            con = ConsoleDestination(console)
            await con.deliver(result.tools)
            con.print_summary(result)

            # Start periodic scheduling
            await scheduler.add_periodic(job, every)
            console.print(f"\n[dim]Next fetch in {every}. Press Ctrl+C to stop.[/dim]")
            try:
                await scheduler.start()
            except KeyboardInterrupt:
                await scheduler.stop()
                console.print("\n[dim]Scheduler stopped.[/dim]")

    _run_async(_periodic())


@main.command()
@click.argument("query")
@click.option("--cron", required=True, help="Cron expression (e.g., '0 9 * * MON').")
@click.option("--sources", "-s", default=None, help="Comma-separated sources.")
@click.option("--limit", "-l", default=20, type=int, help="Max results per fetch.")
@click.option("--dest", required=True, help="Destination.")
def schedule(
    query: str,
    cron: str,
    sources: Optional[str],
    limit: int,
    dest: str,
) -> None:
    """Set up cron-based scheduled fetching."""
    source_list = [s.strip() for s in sources.split(",")] if sources else None
    destination = _resolve_destination(dest)
    if not destination:
        console.print("[red]Destination required for scheduled fetching[/red]")
        sys.exit(1)

    from aitoolsfetcher.core.scheduler import FetchJob, Scheduler

    async def _schedule():
        settings = load_settings()
        async with AIToolsFetcher(settings=settings) as fetcher:
            job = FetchJob(query=query, sources=source_list, limit=limit, destination=destination)
            scheduler = Scheduler(fetcher)

            console.print(f"[bold]Scheduled: '{query}' at '{cron}'[/bold]")
            await scheduler.add_cron(job, cron)
            console.print("[dim]Press Ctrl+C to stop.[/dim]")
            try:
                await scheduler.start()
            except KeyboardInterrupt:
                await scheduler.stop()
                console.print("\n[dim]Scheduler stopped.[/dim]")

    _run_async(_schedule())


@main.command(name="sources")
def list_sources() -> None:
    """List available sources and their status."""
    settings = load_settings()

    all_sources = {
        "github": ("GitHub API", "Primary source, most reliable", not settings.github_token),
        "huggingface": ("HuggingFace API", "Models and spaces", True),
        "reddit": ("Reddit (scraping)", "r/MachineLearning, r/LocalLLaMA, etc.", True),
        "arxiv": ("arXiv API", "Research papers with code", True),
        "producthunt": ("Product Hunt (scraping)", "New product launches", True),
        "twitter": ("X/Twitter (scraping)", "Fragile, disabled by default", True),
        "linkedin": ("LinkedIn (scraping)", "Fragile, disabled by default", True),
    }

    from rich.table import Table
    table = Table(title="Available Sources", header_style="bold cyan")
    table.add_column("Source", style="bold")
    table.add_column("Type")
    table.add_column("Status")
    table.add_column("Notes")

    for name, (label, notes, needs_impl) in all_sources.items():
        enabled = settings.is_source_enabled(name)
        if name == "github":
            status = "[green]Active[/green]"
        elif not enabled:
            status = "[dim]Disabled[/dim]"
        else:
            status = "[yellow]Coming soon[/yellow]"
        table.add_row(name, label, status, notes)

    console.print(table)


@main.command()
@click.option("--format", "-f", "fmt", default="json", type=click.Choice(["json", "csv", "md"]))
@click.option("--output", "-o", required=True, help="Output file path.")
@click.option("--limit", "-l", default=100, type=int, help="Max results to export.")
def export(fmt: str, output: str, limit: int) -> None:
    """Export cached tools to a file."""
    async def _export():
        settings = load_settings()
        async with AIToolsFetcher(settings=settings) as fetcher:
            tools = await fetcher.history(limit=limit)
            if not tools:
                console.print("[dim]No tools in cache to export.[/dim]")
                return

            if fmt == "json":
                from aitoolsfetcher.destinations.json_file import JSONFileDestination
                dest = JSONFileDestination(output, append=False)
            else:
                from aitoolsfetcher.destinations.folder import FolderDestination
                import os
                dest = FolderDestination(os.path.dirname(output) or ".", format=fmt)

            await dest.deliver(tools)
            console.print(f"[green]Exported {len(tools)} tools to {output}[/green]")

    _run_async(_export())


if __name__ == "__main__":
    main()
