"""CLI tests with Click's CliRunner. Sources are fakes; config and cache live in tmp_path."""

from __future__ import annotations

import csv
import io
import json

import pytest
from click.testing import CliRunner

from aitoolsfetcher import cli
from aitoolsfetcher.classifiers.keyword import KeywordClassifier
from aitoolsfetcher.config import Settings
from aitoolsfetcher.core import scheduler as scheduler_module
from aitoolsfetcher.core.cache import ToolCache
from aitoolsfetcher.core.engine import AIToolsFetcher
from tests.test_engine import FakeSource


@pytest.fixture
def fake_source(sample_tools):
    return FakeSource("github", sample_tools)


@pytest.fixture
def runner(monkeypatch, tmp_path, fake_source):
    """CliRunner whose fetcher uses a fake GitHub source and a temp cache."""
    db_path = str(tmp_path / "cli-cache.db")

    def make_fetcher(settings: Settings | None = None) -> AIToolsFetcher:
        return AIToolsFetcher(
            settings=settings or Settings(),
            sources={"github": fake_source},
            classifier=KeywordClassifier(),
            cache=ToolCache(db_path=db_path),
        )

    monkeypatch.setattr(cli, "AIToolsFetcher", make_fetcher)
    # Wide terminal so table cells aren't truncated in assertions
    monkeypatch.setattr(cli, "console", cli.Console(width=200, color_system=None))
    return CliRunner()


def invoke(runner: CliRunner, *args: str):
    result = runner.invoke(cli.main, list(args), catch_exceptions=False)
    return result


class TestSearch:
    def test_table_output(self, runner):
        result = invoke(runner, "search", "vector database")
        assert result.exit_code == 0
        assert "VectorForge" in result.output
        assert "https://github.com/vf/vectorforge" in result.output
        assert "3 new AI tools found (3 fetched, 0 cached)" in result.output

    def test_second_run_reports_cached(self, runner):
        invoke(runner, "search", "x")
        result = invoke(runner, "search", "x")
        assert "0 new AI tools found (3 fetched, 3 cached)" in result.output

    def test_passes_options_to_source(self, runner, fake_source):
        invoke(runner, "search", "agents", "--limit", "2", "--min-stars", "1000")
        [filters] = fake_source.search_calls
        assert (filters.query, filters.limit, filters.min_stars) == ("agents", 2, 1000)

    def test_detail_view(self, runner):
        result = invoke(runner, "search", "x", "--detail")
        assert "Relevance:" in result.output

    def test_json_output_is_valid_json(self, runner):
        result = invoke(runner, "search", "x", "--format", "json")
        data = json.loads(result.output)
        assert {d["name"] for d in data} == {"VectorForge", "PromptBench", "WebFramework"}

    def test_csv_output(self, runner):
        result = invoke(runner, "search", "x", "-f", "csv")
        rows = list(csv.DictReader(io.StringIO(result.output)))
        assert len(rows) == 3

    def test_markdown_output(self, runner):
        result = invoke(runner, "search", "x", "-f", "md")
        assert "## [VectorForge](https://github.com/vf/vectorforge)" in result.output

    def test_min_score_filters(self, runner):
        result = invoke(runner, "search", "x", "--min-score", "0.2", "-f", "json")
        names = {d["name"] for d in json.loads(result.output)}
        assert "WebFramework" not in names

    def test_dest_folder_uses_chosen_format(self, runner, tmp_path):
        out = tmp_path / "results"
        result = invoke(runner, "search", "x", "-f", "md", "--dest", f"folder:{out}")
        assert result.exit_code == 0
        [file] = list(out.iterdir())
        assert file.suffix == ".md"
        assert "Results saved to" in result.output

    def test_dest_json(self, runner, tmp_path):
        out = tmp_path / "tools.json"
        invoke(runner, "search", "x", "--dest", f"json:{out}")
        assert len(json.loads(out.read_text())) == 3

    @pytest.mark.parametrize("dest", ["nocolon", "ftp:somewhere"])
    def test_bad_dest_is_a_usage_error(self, runner, dest):
        result = runner.invoke(cli.main, ["search", "x", "--dest", dest])
        assert result.exit_code == 2
        assert "--dest" in result.output

    def test_source_error_is_shown(self, runner, fake_source):
        fake_source._error = RuntimeError("GitHub rate limit reached")
        result = invoke(runner, "search", "x")
        assert "github: GitHub rate limit reached" in result.output

    def test_source_error_goes_to_stderr_for_file_formats(self, runner, fake_source):
        fake_source._error = RuntimeError("down")
        result = invoke(runner, "search", "x", "-f", "json")
        assert json.loads(result.stdout) == []
        assert "error: github: down" in result.stderr


class TestOtherCommands:
    def test_trending_shows_only_ai_tools(self, runner):
        result = invoke(runner, "trending")
        assert result.exit_code == 0
        assert "VectorForge" in result.output
        assert "WebFramework" not in result.output

    def test_history_after_search(self, runner):
        invoke(runner, "search", "x")
        result = invoke(runner, "history", "--limit", "10")
        assert "PromptBench" in result.output
        assert "3 tools shown from cache" in result.output

    def test_history_empty(self, runner):
        result = invoke(runner, "history")
        assert "No tools found" in result.output

    def test_sources(self, runner):
        result = invoke(runner, "sources")
        assert "github" in result.output
        assert "Active" in result.output

    def test_version(self, runner):
        result = invoke(runner, "--version")
        assert result.exit_code == 0
        assert "version" in result.output


class TestExport:
    @pytest.mark.parametrize("fmt", ["json", "csv", "md"])
    def test_writes_exact_output_path(self, runner, tmp_path, fmt):
        invoke(runner, "search", "x")
        out = tmp_path / "exports" / f"tools.{fmt}"
        result = invoke(runner, "export", "-f", fmt, "-o", str(out))
        assert result.exit_code == 0
        assert out.exists()
        assert "VectorForge" in out.read_text()
        assert "Exported 3 tools" in result.output

    def test_empty_cache(self, runner, tmp_path):
        out = tmp_path / "tools.json"
        result = invoke(runner, "export", "-o", str(out))
        assert "No tools in cache" in result.output
        assert not out.exists()


class TestScheduling:
    @pytest.fixture
    def no_loop(self, monkeypatch):
        """Make Scheduler.start return at once instead of blocking forever."""
        started: list[bool] = []

        async def fake_start(self) -> None:
            started.append(True)

        monkeypatch.setattr(scheduler_module.Scheduler, "start", fake_start)
        return started

    def test_periodic_runs_once_then_starts_scheduler(self, runner, tmp_path, no_loop):
        out = tmp_path / "periodic"
        result = invoke(runner, "periodic", "llm", "--every", "6h", "--dest", f"folder:{out}")
        assert result.exit_code == 0
        assert "every 6h" in result.output
        assert len(list(out.iterdir())) == 1  # first run delivered the new tools
        assert no_loop == [True]

    def test_schedule_starts_scheduler(self, runner, tmp_path, no_loop):
        result = invoke(
            runner, "schedule", "llm", "--cron", "0 9 * * MON", "--dest", f"json:{tmp_path}/t.json"
        )
        assert result.exit_code == 0
        assert "Scheduled: 'llm'" in result.output
        assert no_loop == [True]

    def test_ctrl_c_stops_cleanly(self, runner, tmp_path, monkeypatch):
        async def interrupted(self) -> None:
            raise KeyboardInterrupt

        monkeypatch.setattr(scheduler_module.Scheduler, "start", interrupted)
        result = invoke(
            runner, "schedule", "llm", "--cron", "0 9 * * *", "--dest", f"json:{tmp_path}/t.json"
        )
        assert result.exit_code == 0
        assert "Scheduler stopped" in result.output

    def test_bad_interval_is_a_usage_error(self, runner, tmp_path):
        result = runner.invoke(cli.main, ["periodic", "x", "--every", "soon", "--dest", "json:t"])
        assert result.exit_code == 2
        assert "--every" in result.output

    def test_bad_cron_is_a_usage_error(self, runner):
        result = runner.invoke(cli.main, ["schedule", "x", "--cron", "daily", "--dest", "json:t"])
        assert result.exit_code == 2
        assert "--cron" in result.output
