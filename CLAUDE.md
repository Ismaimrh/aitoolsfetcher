# AIToolsFetcher — Project Instructions

Guidance for contributors and AI coding assistants working on this repository. Design details
live in [SPECS.md](SPECS.md), and usage is covered in [README.md](README.md).

## What this is
An open-source Python package and CLI that finds new and trending AI tools, classifies them,
caches them in SQLite and delivers what is new. Its audience is developers keeping up with the
AI ecosystem. GitHub is the only source today; HuggingFace and arXiv are next.

## Architecture principles
- **Independent adapters**: sources, classifiers and destinations each sit behind a small
  abstract base class (`sources/base.py`, `classifiers/base.py`, `destinations/base.py`).
  Adapters never import each other.
- **Classification**: the offline keyword classifier is the default. An LLM classifier is used
  only when the user configures a key.
- **Cache-first**: a local SQLite cache deduplicates across runs (content hash plus source-code
  URL), supports FTS5 search and expires old entries.
- **Three fetch modes**: on demand, periodic (interval) and scheduled (cron).
- **Resilience**: one failing source never stops the others; partial results are returned.

## Tech stack
- Python 3.10+ with type hints everywhere
- `httpx` (async HTTP), `beautifulsoup4` (only for the github.com/trending page)
- `click` (CLI), `rich` (terminal output), `pydantic` (models)
- `apscheduler` (scheduling), `aiosqlite` (cache)

## Code standards
- Google-style docstrings on all public functions; type hints on all signatures
- Async/await throughout the fetching layer
- Pydantic models for data structures
- src layout: `src/aitoolsfetcher/`
- Must pass `ruff check`, `ruff format --check`, `mypy --strict` and `pytest --cov` with at least
  80% coverage (CI enforces this on Python 3.10–3.13)

## Rules
- **Sources**: use official APIs, and check the platform's current terms before adding a
  source. Send the package User-Agent and back off on rate limits (429 and rate-limit 403s).
  Do not collect anything from X/Twitter or LinkedIn.
- **Free by default**: everything must work with zero config and no paid service. LLM features
  should support free options (Groq free tier, local Ollama) through an OpenAI-compatible
  `base_url`; Anthropic and OpenAI stay optional.
- **Secrets**: never put keys in code, tests or fixtures. Keys come from env vars or
  `~/.aitoolsfetcher/config.toml`, and `config.example.toml` documents the options.
- **Tests**: mock every network call (`respx`, fake sources). Tests must never touch the real
  user config or cache; the autouse fixture in `tests/conftest.py` handles this.
- **Privacy**: store only public project metadata. Authors appear only as the account or
  organisation handle the platform shows.
- **Numbers**: any figure in the README must come from a run of this project, with its date and
  method stated.

## File structure
```
├── README.md, SPECS.md, CLAUDE.md, LICENSE
├── pyproject.toml
├── config.example.toml
├── .github/workflows/ci.yml
├── src/aitoolsfetcher/
│   ├── cli.py              # Click CLI
│   ├── config.py           # Settings from config.toml and env vars
│   ├── formatters.py       # json / csv / md rendering
│   ├── core/
│   │   ├── engine.py       # AIToolsFetcher orchestrator
│   │   ├── models.py       # AITool, AICategory, SearchFilters, FetchResult
│   │   ├── cache.py        # SQLite cache (dedup, FTS5, TTL)
│   │   └── scheduler.py    # periodic and cron jobs
│   ├── sources/
│   │   ├── base.py         # BaseSource, User-Agent
│   │   └── github.py       # REST search API + trending page
│   ├── classifiers/
│   │   ├── base.py
│   │   ├── keyword.py      # default, offline
│   │   └── llm.py          # optional
│   └── destinations/
│       ├── base.py
│       ├── console.py
│       ├── folder.py
│       └── json_file.py
└── tests/                  # mirrors src/, all network mocked
```

## When adding a source or destination
- Add one adapter at a time, behind the base class, with its own rate limiting and error handling.
- Add mocked tests next to the existing ones.
- Update the README's "Works today" and "Roadmap" sections and SPECS.md.
