"""Scheduler for periodic and cron-based fetching."""

from __future__ import annotations

import asyncio
import logging
from datetime import timedelta
from typing import Optional

from aitoolsfetcher.core.models import FetchResult
from aitoolsfetcher.destinations.base import BaseDestination

logger = logging.getLogger(__name__)


def _parse_interval(interval: str) -> timedelta:
    """Parse interval string like '6h', '30m', '1d' into timedelta."""
    unit_map = {"s": "seconds", "m": "minutes", "h": "hours", "d": "days", "w": "weeks"}
    interval = interval.strip().lower()
    if not interval:
        raise ValueError("Empty interval string")

    unit = interval[-1]
    if unit not in unit_map:
        raise ValueError(f"Unknown interval unit '{unit}'. Use: s, m, h, d, w")

    try:
        value = int(interval[:-1])
    except ValueError:
        raise ValueError(f"Invalid interval number: {interval[:-1]}")

    if value <= 0:
        raise ValueError("Interval must be positive")

    return timedelta(**{unit_map[unit]: value})


class FetchJob:
    """A scheduled or periodic fetch job."""

    def __init__(
        self,
        query: str,
        sources: Optional[list[str]] = None,
        limit: int = 20,
        destination: Optional[BaseDestination] = None,
    ):
        self.query = query
        self.sources = sources
        self.limit = limit
        self.destination = destination
        self._running = False

    async def execute(self, fetcher) -> FetchResult:
        """Execute the fetch and deliver results."""
        result = await fetcher.search(
            query=self.query,
            sources=self.sources,
            limit=self.limit,
        )

        if self.destination and result.tools:
            new_tools = [t for t in result.tools][:result.new_count] if result.new_count else []
            if new_tools:
                await self.destination.deliver(new_tools)
                logger.info(
                    "Delivered %d new tools to %s", len(new_tools), self.destination.name
                )

        return result


class Scheduler:
    """Manages periodic and cron-based fetch jobs.

    Uses asyncio for scheduling rather than APScheduler to keep things
    simple for v1. APScheduler can be added later for cron expressions.
    """

    def __init__(self, fetcher):
        self._fetcher = fetcher
        self._tasks: list[asyncio.Task] = []
        self._running = False

    async def add_periodic(
        self,
        job: FetchJob,
        interval: str | timedelta,
    ) -> None:
        """Add a periodic fetch job."""
        if isinstance(interval, str):
            interval = _parse_interval(interval)

        async def _run_periodic():
            while self._running:
                try:
                    result = await job.execute(self._fetcher)
                    logger.info(
                        "Periodic fetch [%s]: %s", job.query, result.summary
                    )
                except Exception as e:
                    logger.error("Periodic fetch failed: %s", e)
                await asyncio.sleep(interval.total_seconds())

        if self._running:
            task = asyncio.create_task(_run_periodic())
            self._tasks.append(task)

    async def add_cron(
        self,
        job: FetchJob,
        cron_expr: str,
    ) -> None:
        """Add a cron-based fetch job.

        For v1, this uses APScheduler's CronTrigger for parsing.
        """
        try:
            from apscheduler.triggers.cron import CronTrigger
        except ImportError:
            raise ImportError("apscheduler required for cron scheduling")

        parts = cron_expr.split()
        if len(parts) != 5:
            raise ValueError(f"Expected 5-field cron expression, got {len(parts)} fields")

        trigger = CronTrigger(
            minute=parts[0],
            hour=parts[1],
            day=parts[2],
            month=parts[3],
            day_of_week=parts[4],
        )

        async def _run_cron():
            while self._running:
                now = asyncio.get_event_loop().time()
                next_fire = trigger.get_next_fire_time(None, None)
                if next_fire is None:
                    break

                from datetime import datetime
                wait_seconds = (next_fire - datetime.now(next_fire.tzinfo)).total_seconds()
                if wait_seconds > 0:
                    await asyncio.sleep(wait_seconds)

                if not self._running:
                    break

                try:
                    result = await job.execute(self._fetcher)
                    logger.info("Cron fetch [%s]: %s", job.query, result.summary)
                except Exception as e:
                    logger.error("Cron fetch failed: %s", e)

        if self._running:
            task = asyncio.create_task(_run_cron())
            self._tasks.append(task)

    async def start(self) -> None:
        """Start the scheduler. Blocks until stop() is called."""
        self._running = True
        logger.info("Scheduler started with %d pending jobs", len(self._tasks))
        try:
            # Keep running until stopped
            while self._running:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        finally:
            await self.stop()

    async def stop(self) -> None:
        """Stop all scheduled jobs."""
        self._running = False
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        logger.info("Scheduler stopped")
