# AIToolsFetcher — Project Instructions

## What This Is
Open-source Python package (PyPI) that aggregates and surfaces trending/new AI tools from multiple platforms. Target: developers keeping up with the AI ecosystem.

## Architecture Principles
- **Modular source adapters**: each data source (GitHub, X, LinkedIn, Reddit, HuggingFace, Product Hunt, arXiv) is an independent adapter behind a common interface
- **Classification pipeline**: keyword-based by default, LLM-enhanced when API key provided
- **Cache-first**: SQLite local cache to deduplicate and avoid re-fetching known tools
- **Three fetch modes**: on-demand, scheduled (cron-like), periodic (interval-based)
- **Output pipeline**: CLI summary table + detailed results (JSON default, extensible to Notion/X/email)

## Tech Stack
- Python 3.10+ with type hints everywhere
- `httpx` for async HTTP, `beautifulsoup4` for scraping
- `click` for CLI, `rich` for terminal tables
- `pydantic` for data models
- `apscheduler` for scheduling
- SQLite via `aiosqlite` for cache
- GitHub API (primary, structured), scraping for X/LinkedIn/Reddit/Product Hunt
- HuggingFace API, arXiv API

## Code Standards
- All public functions must have docstrings (Google style)
- Type hints on all function signatures
- Tests with pytest, aim for 80%+ coverage on core logic
- Use async/await throughout the fetching layer
- Keep adapters independent — no cross-adapter imports
- Pydantic models for all data structures
- Follow src layout: `src/aitoolsfetcher/`

## Key Decisions
- GitHub is the primary and most reliable source; other sources are best-effort via scraping
- X/LinkedIn scraping is legally grey — implement cautiously, make optional, document risks
- LLM classification is opt-in (requires user API key), keyword matching is the default
- All output destinations (file, Notion, X feed, email) are "destination adapters" — same pattern as source adapters
- Cache uses content hash for dedup, not just URL (same tool may appear on multiple platforms)

## File Structure
```
AIToolsFetcher/
├── CLAUDE.md
├── pyproject.toml
├── README.md
├── LICENSE
├── src/
│   └── aitoolsfetcher/
│       ├── __init__.py
│       ├── cli.py              # Click CLI entry point
│       ├── core/
│       │   ├── __init__.py
│       │   ├── engine.py       # Main fetch orchestrator
│       │   ├── models.py       # Pydantic data models
│       │   ├── cache.py        # SQLite cache layer
│       │   └── scheduler.py    # APScheduler wrapper
│       ├── sources/
│       │   ├── __init__.py
│       │   ├── base.py         # Abstract source adapter
│       │   ├── github.py
│       │   ├── twitter.py      # X/Twitter scraper
│       │   ├── linkedin.py
│       │   ├── reddit.py
│       │   ├── huggingface.py
│       │   ├── producthunt.py
│       │   └── arxiv.py
│       ├── classifiers/
│       │   ├── __init__.py
│       │   ├── base.py         # Abstract classifier
│       │   ├── keyword.py      # Keyword-based classifier
│       │   └── llm.py          # LLM-based classifier (optional)
│       ├── destinations/
│       │   ├── __init__.py
│       │   ├── base.py         # Abstract destination adapter
│       │   ├── json_file.py
│       │   ├── folder.py       # Save to local folder
│       │   ├── notion.py       # Notion integration
│       │   ├── email.py        # Email delivery
│       │   └── social.py       # Post to X/LinkedIn
│       └── config.py           # Configuration management
└── tests/
    ├── conftest.py
    ├── test_engine.py
    ├── test_cache.py
    ├── test_classifiers/
    └── test_sources/
```

## When Implementing
- Build source adapters one at a time; GitHub first, then expand
- Each adapter must handle its own rate limiting and error recovery
- Never store API keys in code — use env vars or config file (~/.aitoolsfetcher/config.toml)
- CLI should work with zero config for basic GitHub fetching
- Keep the package dependency footprint minimal — optional deps for optional features
