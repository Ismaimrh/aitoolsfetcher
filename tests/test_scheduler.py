"""Tests for interval/cron parsing, fetch jobs and the scheduler loop."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Any

import pytest

from aitoolsfetcher.core import scheduler as scheduler_module
from aitoolsfetcher.core.models import AITool, FetchResult
from aitoolsfetcher.core.scheduler import FetchJob, Scheduler, parse_cron, parse_interval
from aitoolsfetcher.destinations.base import BaseDestination


class RecordingDestination(BaseDestination):
    name = "recording"

    def __init__(self) -> None:
        self.deliveries: list[list[str]] = []

    async def deliver(self, tools: list[AITool], **kwargs: Any) -> None:
        self.deliveries.append([t.name for t in tools])


class FakeFetcher:
    """Stands in for AIToolsFetcher; returns queued results."""

    def __init__(self, results: list[FetchResult] | None = None, error: bool = False) -> None:
        self.results = list(results or [])
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def search(self, **kwargs: Any) -> FetchResult:
        self.calls.append(kwargs)
        if self.error:
            raise RuntimeError("source down")
        return self.results.pop(0) if self.results else FetchResult()


class TestParseInterval:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("30s", timedelta(seconds=30)),
            ("15m", timedelta(minutes=15)),
            (" 6H ", timedelta(hours=6)),
            ("1d", timedelta(days=1)),
            ("2w", timedelta(weeks=2)),
        ],
    )
    def test_valid(self, text, expected):
        assert parse_interval(text) == expected

    @pytest.mark.parametrize(
        ("text", "message"),
        [("", "Empty"), ("5x", "Unknown interval unit"), ("abch", "Invalid"), ("0m", "positive")],
    )
    def test_invalid(self, text, message):
        with pytest.raises(ValueError, match=message):
            parse_interval(text)


class TestParseCron:
    @staticmethod
    def next_fire(expr: str, after: datetime) -> datetime:
        trigger = parse_cron(expr)
        start = after.replace(tzinfo=trigger.timezone)
        fire = trigger.get_next_fire_time(None, start)
        assert fire is not None
        return fire

    def test_numeric_weekday_follows_standard_cron(self):
        # 2026-10-17 is a Saturday. In standard cron, 1 = Monday.
        fire = self.next_fire("0 9 * * 1", datetime(2026, 10, 17, 12, 0))
        assert fire.strftime("%A %H:%M") == "Monday 09:00"

    def test_zero_and_seven_mean_sunday(self):
        start = datetime(2026, 10, 17, 12, 0)
        assert self.next_fire("0 9 * * 0", start).strftime("%A") == "Sunday"
        assert self.next_fire("0 9 * * 7", start).strftime("%A") == "Sunday"

    def test_named_weekdays_and_ranges(self):
        fire = self.next_fire("30 8 * * MON-FRI", datetime(2026, 10, 17, 12, 0))
        assert fire.strftime("%A %H:%M") == "Monday 08:30"
        fire = self.next_fire("0 9 * * 1-5", datetime(2026, 10, 17, 12, 0))
        assert fire.strftime("%A") == "Monday"

    @pytest.mark.parametrize("expr", ["0 9 * *", "0 9 * * 1 extra", "0 9 * * 8", "99 9 * * *"])
    def test_invalid(self, expr):
        with pytest.raises(ValueError):
            parse_cron(expr)


class TestFetchJob:
    async def test_delivers_only_new_tools(self, sample_tools):
        # The second tool is new; the first and third were already cached
        result = FetchResult(tools=sample_tools, new_count=1, new_ids=[sample_tools[1].id])
        dest = RecordingDestination()
        job = FetchJob("llm", sources=["github"], limit=5, destination=dest)

        returned = await job.execute(FakeFetcher([result]))  # type: ignore[arg-type]

        assert returned is result
        assert dest.deliveries == [["PromptBench"]]

    async def test_passes_query_options(self):
        fetcher = FakeFetcher()
        await FetchJob("llm", sources=["github"], limit=5).execute(fetcher)  # type: ignore[arg-type]
        assert fetcher.calls == [{"query": "llm", "sources": ["github"], "limit": 5}]

    async def test_nothing_new_means_no_delivery(self, sample_tools):
        dest = RecordingDestination()
        job = FetchJob("llm", destination=dest)
        await job.execute(FakeFetcher([FetchResult(tools=sample_tools)]))  # type: ignore[arg-type]
        assert dest.deliveries == []


async def _run_until(scheduler: Scheduler, condition, timeout: float = 2.0) -> None:
    """Start the scheduler, stop it once ``condition()`` is true."""
    task = asyncio.create_task(scheduler.start())

    async def _wait() -> None:
        while not condition():
            await asyncio.sleep(0.005)

    try:
        await asyncio.wait_for(_wait(), timeout)
    finally:
        await scheduler.stop()
        await task


class TestScheduler:
    async def test_job_added_before_start_runs(self, sample_tools):
        """Regression: jobs registered before start() used to be dropped silently."""
        results = [
            FetchResult(tools=sample_tools[:1], new_ids=[sample_tools[0].id]),
            FetchResult(tools=sample_tools[1:2], new_ids=[sample_tools[1].id]),
        ]
        fetcher = FakeFetcher(results)
        dest = RecordingDestination()
        scheduler = Scheduler(fetcher)  # type: ignore[arg-type]
        await scheduler.add_periodic(
            FetchJob("llm", destination=dest), timedelta(milliseconds=10), run_immediately=True
        )

        await _run_until(scheduler, lambda: len(fetcher.calls) >= 2)

        assert dest.deliveries[:2] == [["VectorForge"], ["PromptBench"]]

    async def test_job_added_while_running_starts_immediately(self):
        fetcher = FakeFetcher()
        scheduler = Scheduler(fetcher)  # type: ignore[arg-type]
        task = asyncio.create_task(scheduler.start())
        await asyncio.sleep(0)
        await scheduler.add_periodic(FetchJob("llm"), timedelta(milliseconds=5))
        for _ in range(200):
            if fetcher.calls:
                break
            await asyncio.sleep(0.005)
        await scheduler.stop()
        await task
        assert fetcher.calls

    async def test_failing_run_does_not_stop_the_schedule(self, caplog):
        fetcher = FakeFetcher(error=True)
        scheduler = Scheduler(fetcher)  # type: ignore[arg-type]
        await scheduler.add_periodic(
            FetchJob("llm"), timedelta(milliseconds=5), run_immediately=True
        )
        await _run_until(scheduler, lambda: len(fetcher.calls) >= 3)
        assert "source down" in caplog.text

    async def test_cron_job_waits_for_next_fire_time(self, monkeypatch):
        waits: list[float] = []
        real_sleep = asyncio.sleep

        async def fake_sleep(seconds: float) -> None:
            waits.append(seconds)
            await real_sleep(0)

        monkeypatch.setattr(scheduler_module.asyncio, "sleep", fake_sleep)
        fetcher = FakeFetcher()
        scheduler = Scheduler(fetcher)  # type: ignore[arg-type]
        await scheduler.add_cron(FetchJob("llm"), "0 9 * * MON")

        task = asyncio.create_task(scheduler.start())
        for _ in range(1000):
            if len(fetcher.calls) >= 2:
                break
            await real_sleep(0)
        await scheduler.stop()
        await task

        assert len(fetcher.calls) >= 2
        assert 0 < waits[0] <= 7 * 24 * 3600  # never more than a week away

    async def test_invalid_cron_is_rejected(self):
        with pytest.raises(ValueError):
            await Scheduler(FakeFetcher()).add_cron(FetchJob("x"), "bad")  # type: ignore[arg-type]
