"""Tests for destination adapters and output formatters (all writes go to tmp_path)."""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timezone

import pytest
from rich.console import Console

from aitoolsfetcher.core.models import AITool, FetchResult
from aitoolsfetcher.destinations.console import ConsoleDestination
from aitoolsfetcher.destinations.folder import FolderDestination
from aitoolsfetcher.destinations.json_file import JSONFileDestination
from aitoolsfetcher.formatters import render, to_csv, to_markdown


class TestFormatters:
    def test_json_contains_every_tool(self, sample_tools):
        data = json.loads(render(sample_tools, "json"))
        assert [d["name"] for d in data] == [t.name for t in sample_tools]
        assert data[0]["id"] == sample_tools[0].id

    def test_csv_has_header_and_rows(self, sample_tools):
        rows = list(csv.DictReader(io.StringIO(to_csv(sample_tools))))
        assert len(rows) == 3
        assert rows[0]["name"] == "VectorForge"
        assert rows[0]["stars"] == "3200"
        assert rows[1]["source"] == "https://huggingface.co/spaces/promptbench"

    def test_markdown_links_and_heading(self, sample_tool):
        md = to_markdown([sample_tool], now=datetime(2026, 10, 17, 9, 30, tzinfo=timezone.utc))
        assert md.startswith("# AI Tools — 2026-10-17 09:30 UTC")
        assert "## [AutoAgent](https://github.com/test/autoagent)" in md
        assert "**Stars:** 1,500" in md

    def test_unknown_format_raises(self, sample_tools):
        with pytest.raises(ValueError, match="Unsupported format"):
            render(sample_tools, "xml")


class TestFolderDestination:
    @pytest.mark.parametrize("fmt", ["json", "csv", "md"])
    async def test_writes_one_file_per_delivery(self, tmp_path, sample_tools, fmt):
        dest = FolderDestination(tmp_path / "out", fmt=fmt)
        await dest.deliver(sample_tools)
        await dest.deliver(sample_tools)

        files = sorted((tmp_path / "out").iterdir())
        assert len(files) == 2  # same-second deliveries don't overwrite each other
        assert all(f.suffix == f".{fmt}" for f in files)
        assert "VectorForge" in files[0].read_text()
        assert dest.last_written in files

    async def test_empty_delivery_writes_nothing(self, tmp_path):
        await FolderDestination(tmp_path / "out").deliver([])
        assert not (tmp_path / "out").exists()

    def test_rejects_unknown_format(self, tmp_path):
        with pytest.raises(ValueError, match="Unsupported format"):
            FolderDestination(tmp_path, fmt="xml")


class TestJSONFileDestination:
    async def test_append_skips_existing_ids(self, tmp_path, sample_tools):
        path = tmp_path / "sub" / "tools.json"
        dest = JSONFileDestination(path)
        await dest.deliver(sample_tools[:2])
        await dest.deliver(sample_tools)  # two duplicates, one new

        data = json.loads(path.read_text())
        assert [d["name"] for d in data] == ["VectorForge", "PromptBench", "WebFramework"]

    async def test_overwrite_mode_replaces_content(self, tmp_path, sample_tools):
        path = tmp_path / "tools.json"
        dest = JSONFileDestination(path, append=False)
        await dest.deliver(sample_tools)
        await dest.deliver(sample_tools[:1])
        assert [d["name"] for d in json.loads(path.read_text())] == ["VectorForge"]

    async def test_append_wraps_a_single_object(self, tmp_path, sample_tools):
        path = tmp_path / "tools.json"
        path.write_text(json.dumps({"id": "manual", "name": "Manual entry"}))
        await JSONFileDestination(path).deliver(sample_tools[:1])
        assert [d["name"] for d in json.loads(path.read_text())] == ["Manual entry", "VectorForge"]

    async def test_refuses_to_overwrite_invalid_json(self, tmp_path, sample_tools):
        path = tmp_path / "tools.json"
        path.write_text("{ not json")
        with pytest.raises(ValueError, match="not valid JSON"):
            await JSONFileDestination(path).deliver(sample_tools)
        assert path.read_text() == "{ not json"

    async def test_empty_delivery_writes_nothing(self, tmp_path):
        path = tmp_path / "tools.json"
        await JSONFileDestination(path).deliver([])
        assert not path.exists()


def _console() -> tuple[ConsoleDestination, Console]:
    console = Console(record=True, width=200, color_system=None)
    return ConsoleDestination(console), console


class TestConsoleDestination:
    async def test_table_shows_name_stars_and_link(self, sample_tool):
        dest, console = _console()
        await dest.deliver([sample_tool])
        out = console.export_text()
        assert "AutoAgent" in out
        assert "1,500" in out
        assert "https://github.com/test/autoagent" in out

    async def test_detail_view(self, sample_tool):
        dest, console = _console()
        await dest.deliver([sample_tool], detail=True)
        out = console.export_text()
        assert "Category:" in out
        assert "ai-agent, llm, mcp, autonomous" in out
        assert "1/1" in out

    async def test_empty_list(self):
        dest, console = _console()
        await dest.deliver([])
        assert "No tools found" in console.export_text()

    async def test_markup_in_source_text_is_escaped(self):
        tool = AITool(
            name="[bold]sneaky[/bold]",
            description="closing [/foo] tag and [red]color[/red]",
            source_platform="github",
            source_url="https://github.com/x/y",
        )
        dest, console = _console()
        await dest.deliver([tool])
        await dest.deliver([tool], detail=True)
        out = console.export_text()
        assert "[bold]sneaky[/bold]" in out
        assert "closing [/foo] tag" in out

    def test_summary_lists_errors(self):
        dest, console = _console()
        dest.print_summary(FetchResult(new_count=1, total_fetched=2, errors={"github": "429"}))
        out = console.export_text()
        assert "1 new AI tools found (2 fetched, 0 cached)" in out
        assert "github: 429" in out
