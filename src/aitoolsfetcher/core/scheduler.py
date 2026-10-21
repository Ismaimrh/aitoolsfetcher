"""Scheduler for periodic and cron-based fetching."""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from apscheduler.triggers.cron import CronTrigger

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine
    from typing import Any

    from aitoolsfetcher.core.engine import AIToolsFetcher
    from aitoolsfetcher.core.models import FetchResult
    from aitoolsfetcher.destinations.base import BaseDestination

logger = logging.getLogger(__name__)

_UNITS = {"s": "seconds", "m": "minutes", "h": "hours", "d": "days", "w": "weeks"}
# Standard cron numbers days 0-7 from Sunday; APScheduler numbers them from Monday.
_CRON_DAYS = ["sun", "mon", "tue", "wed", "thu", "fri", "sat", "sun"]


def parse_interval(interval: str) -> timedelta:
    """Parse an interval string such as ``"6h"``, ``"30m"`` or ``"1d"``.

    Args:
        interval: A positive integer followed by s, m, h, d or w.

    Returns:
        The interval as a timedelta.

    Raises:
        ValueError: If the string is empty, malformed or not positive.
    """
    interval = interval.strip().lower()
    if not interval:
        raise ValueError("Empty interval string")

    unit = interval[-1]
    if unit not in _UNITS:
        raise ValueError(f"Unknown interval unit '{unit}'. Use: s, m, h, d, w")

    try:
        value = int(interval[:-1])
    except ValueError:
        raise ValueError(f"Invalid interval number: {interval[:-1]}") from None

    if value <= 0:
        raise ValueError("Interval must be positive")

    return timedelta(**{_UNITS[unit]: value})


def parse_cron(expr: str) -> CronTrigger:
    """Parse a standard 5-field cron expression.

    Numeric day-of-week values follow standard cron (0 or 7 = Sunday,
    1 = Monday), not APScheduler's Monday-based numbering.

    Args:
        expr: Expression like ``"0 9 * * MON"`` (minute hour day month weekday).

    Returns:
        An APScheduler cron trigger in the local timezone.

    Raises:
        ValueError: If the expression does not have 5 fields or a field is invalid.
    """
    parts = expr.split()
    if len(parts) != 5:
        raise ValueError(f"Expected 5-field cron expression, got {len(parts)} fields")

    def _day_name(match: re.Match[str]) -> str:
        number = int(match.group())
        if number > 7:
            raise ValueError(f"Invalid day of week: {number}")
        return _CRON_DAYS[number]

    day_of_week = re.sub(r"\d+", _day_name, parts[4])
    return CronTrigger(
        minute=parts[0], hour=parts[1], day=parts[2], month=parts[3], day_of_week=day_of_week
    )


class FetchJob:
    """A search that runs on a schedule and delivers new tools to a destination."""

    def __init__(
        self,
        query: str,
        sources: list[str] | None = None,
        limit: int = 20,
        destination: BaseDestination | None = None,
    ) -> None:
        """Create a job.

        Args:
            query: Search query.
            sources: Source names, or None for all enabled sources.
            limit: Maximum results per run.
            destination: Where new tools are delivered. Optional.
        """
        self.query = query
        self.sources = sources
        self.limit = limit
        self.destination = destination

    async def execute(self, fetcher: AIToolsFetcher) -> FetchResult:
        """Run the search once and deliver only tools that were not cached before.

        Args:
            fetcher: The fetcher to search with.

        Returns:
            The full fetch result.
        """
        result = await fetcher.search(query=self.query, sources=self.sources, limit=self.limit)

        if self.destination:
            new_ids = set(result.new_ids)
            new_tools = [t for t in result.tools if t.id in new_ids]
            if new_tools:
                await self.destination.deliver(new_tools)
                logger.info("Delivered %d new tools to %s", len(new_tools), self.destination.name)

        return result


class Scheduler:
    """Runs periodic and cron-based fetch jobs on the current event loop.

    Jobs can be added before or after ``start()``. ``start()`` blocks until
    ``stop()`` is called (or the task is cancelled, e.g. by Ctrl+C).
    """

    def __init__(self, fetcher: AIToolsFetcher) -> None:
        """Create a scheduler.

        Args:
            fetcher: Fetcher passed to every job run.
        """
        self._fetcher = fetcher
        self._pending: list[Callable[[], Coroutine[Any, Any, None]]] = []
        self._tasks: list[asyncio.Task[None]] = []
        self._running = False
        self._stopped = asyncio.Event()

    def _add(self, loop: Callable[[], Coroutine[Any, Any, None]]) -> None:
        """Start a job loop now if running, otherwise when ``start()`` is called."""
        if self._running:
            self._tasks.append(asyncio.create_task(loop()))
        else:
            self._pending.append(loop)

    async def _run_job(self, job: FetchJob, kind: str) -> None:
        """Run one job, logging instead of raising on failure."""
        try:
            result = await job.execute(self._fetcher)
            logger.info("%s fetch [%s]: %s", kind, job.query, result.summary)
        except Exception as e:  # a failed run must not kill the schedule
            logger.error("%s fetch [%s] failed: %s", kind, job.query, e)

    async def add_periodic(
        self, job: FetchJob, interval: str | timedelta, run_immediately: bool = False
    ) -> None:
        """Add a job that runs at a fixed interval.

        Args:
            job: The job to run.
            interval: A timedelta or a string such as ``"6h"``.
            run_immediately: If True, the first run happens at once; otherwise
                after one interval.
        """
        delay = parse_interval(interval) if isinstance(interval, str) else interval
        seconds = delay.total_seconds()

        async def _loop() -> None:
            if not run_immediately:
                await asyncio.sleep(seconds)
            while self._running:
                await self._run_job(job, "Periodic")
                await asyncio.sleep(seconds)

        self._add(_loop)

    async def add_cron(self, job: FetchJob, cron_expr: str) -> None:
        """Add a job that runs on a cron schedule.

        Args:
            job: The job to run.
            cron_expr: 5-field cron expression, e.g. ``"0 9 * * MON"``.

        Raises:
            ValueError: If the expression is invalid.
        """
        trigger = parse_cron(cron_expr)

        async def _loop() -> None:
            previous: datetime | None = None
            while self._running:
                now = datetime.now(trigger.timezone)
                next_fire = trigger.get_next_fire_time(previous, now)
                if next_fire is None:
                    return
                await asyncio.sleep(max((next_fire - now).total_seconds(), 0.0))
                if not self._running:
                    return
                await self._run_job(job, "Cron")
                previous = next_fire

        self._add(_loop)

    async def start(self) -> None:
        """Start all jobs and block until ``stop()`` is called."""
        self._running = True
        self._stopped.clear()
        for loop in self._pending:
            self._tasks.append(asyncio.create_task(loop()))
        self._pending.clear()
        logger.info("Scheduler started with %d jobs", len(self._tasks))
        try:
            await self._stopped.wait()
        finally:
            await self._shutdown()

    async def stop(self) -> None:
        """Stop the scheduler and cancel all running jobs."""
        self._running = False
        self._stopped.set()
        await self._shutdown()

    async def _shutdown(self) -> None:
        """Cancel job tasks and wait for them to finish."""
        self._running = False
        tasks, self._tasks = self._tasks, []
        current = asyncio.current_task()
        for task in tasks:
            if task is not current:
                task.cancel()
        others = [t for t in tasks if t is not current]
        if others:
            await asyncio.gather(*others, return_exceptions=True)
        logger.info("Scheduler stopped")
