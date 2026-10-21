# AIToolsFetcher — Design Notes

This is the design document for AIToolsFetcher. It describes what is built today and what is
planned. Each section is marked **Built**, **Planned** or **Not planned**. For install and usage,
see the [README](README.md).

## 1. Overview

AIToolsFetcher finds new and trending AI tools, classifies them, keeps them in a local SQLite
cache and delivers what is new. It works with zero configuration and no API keys.

```python
from aitoolsfetcher import AIToolsFetcher

async with AIToolsFetcher() as fetcher:
    result = await fetcher.search("AI code generation agents", limit=10)
```

```bash
aitoolsfetcher search "vector database" --format json
```

The package is installed from source for now. Releasing it on PyPI is planned.

## 2. Fetch modes — Built

| Mode | CLI | Behaviour |
|---|---|---|
| On demand | `search QUERY`, `trending` | Runs once and prints or exports the results |
| Periodic | `periodic QUERY --every 6h --dest ...` | Runs now, then on a fixed interval (`30m`, `6h`, `1d`) |
| Scheduled | `schedule QUERY --cron "0 9 * * MON" --dest ...` | Runs on a 5-field cron expression in local time |

Periodic and scheduled runs deliver only the tools that were not in the cache before that run.
Cron weekday numbers follow standard crontab (0 and 7 are Sunday). In Python, the same jobs go
through `aitoolsfetcher.core.scheduler.Scheduler` with a `FetchJob`.

## 3. Sources

### 3.1 Adapter interface — Built

```python
class BaseSource(ABC):
    name: str
    requires_auth: bool

    async def search(self, filters: SearchFilters) -> list[AITool]: ...
    async def trending(self, limit: int = 20) -> list[AITool]: ...
    async def traceback_source_code(self, tool: AITool) -> str | None: ...
```

Each adapter handles its own errors and rate limiting. The engine queries sources concurrently,
so a source that fails is reported in `FetchResult.errors` and never stops the others. Adapters
never import each other.

### 3.2 Source status

| Source | Access method | Status |
|---|---|---|
| GitHub search | Official REST search API | **Built** |
| GitHub trending | Public github.com/trending page (GitHub has no trending API) | **Built**, under review: may move to API search for recently created repos |
| HuggingFace | Official Hub API (trending models and Spaces) | Planned |
| arXiv | Official API (recent cs.AI / cs.CL / cs.LG papers that link to code) | Planned |
| Product Hunt | Official API only | Planned, only if its current terms allow this use |
| Reddit | Official API only | Planned, only if its current terms allow this use |
| X / Twitter | — | **Not planned**: no data is collected from X |
| LinkedIn | — | **Not planned**: no data is collected from LinkedIn |

Rules for every source:

- Prefer official APIs over scraping, and check the vendor's current terms before adding one.
- Send a descriptive User-Agent: `aitoolsfetcher/<version> (+https://github.com/Ismaimrh/aitoolsfetcher)`.
- Back off on HTTP 429 and on rate-limit 403s.
- Keep only public project metadata, with the author shown only as the account or organisation
  handle the platform displays.

### 3.3 GitHub adapter — Built

- **Search**: the REST search API with the user's query, plus a `topic:ai` filter when the query
  contains no AI term. Results are sorted by stars and paginated with a fixed page size.
- **Trending**: this week's github.com/trending page, filtered to AI repos by the classifier.
- **Metadata**: name, description, topics, stars, language, licence, owner handle, homepage,
  last push.
- **Rate limiting**: waits as the `retry-after` or `x-ratelimit-reset` headers ask (falling back
  to exponential backoff), with a capped wait and a capped number of retries. If nothing was
  collected it raises `GitHubRateLimitError` with a clear message; otherwise it returns partial
  results.
- **Token**: optional (`GITHUB_TOKEN`). It raises GitHub's rate limit; no scopes are needed.

### 3.4 Source-code traceback — Planned

Tools found outside GitHub (HuggingFace, arXiv) should be linked to their code:

1. Extract URLs from the page or paper.
2. Keep links that point to GitHub, GitLab or HuggingFace.
3. Otherwise, search GitHub for the tool name plus keywords from its description.
4. Return the best match, or `None`.

## 4. Data model — Built

```python
class AITool(BaseModel):
    name: str
    description: str
    category: AICategory
    source_platform: str             # "github", ...
    source_url: str                  # where the tool was found
    source_code_url: str | None      # code repository, if known
    homepage_url: str | None
    stars: int | None
    language: str | None
    topics: list[str]
    license: str | None
    author: str                      # account or organisation handle only
    discovered_at: datetime
    last_updated: datetime | None
    popularity_score: float
    ai_relevance_score: float        # 0-1
    metadata: dict[str, Any]

    id: str                          # computed: hash of (name, source_platform, author)

class AICategory(str, Enum):
    AGENT, FRAMEWORK, MODEL, DATASET, INFRA, DEV_TOOL, LIBRARY, CLI_TOOL, OTHER
```

`FetchResult` carries the tools, `new_ids` (the tools not seen in the cache before), counts and
per-source errors.

## 5. Classification

### 5.1 Keyword classifier (default) — Built

Offline and free. A weighted keyword taxonomy is matched on whole words in the name, description
and topics, with a few penalty words. It returns an AI-relevance score (0-1) and the best-matching
category. Its accuracy has not been measured yet; a labelled evaluation set is planned.

### 5.2 LLM classifier (optional) — Built

Used only when an LLM API key is configured (`AITF_LLM_KEY` or `llm_api_key`). It sends each
tool's name, description and topics to the model and gets back `is_ai_tool`, `category`,
`confidence` and a one-line summary. When the model says a tool is *not* AI, its relevance
becomes `1 - confidence` and its category `other`.

- Providers today: Anthropic and OpenAI, which bill per call.
- If the provider's SDK isn't installed, the engine falls back to the keyword classifier with a
  warning.
- **Planned**: free providers through an OpenAI-compatible `base_url` (Groq free tier, local
  Ollama), model names in config, and batching to cut the number of calls.

### 5.3 Pipeline

```
Raw results → classify (keyword, or LLM if configured) → filter by relevance → cache → output
```

## 6. Cache — Built

`ToolCache` in `core/cache.py`, stored at `~/.aitoolsfetcher/cache.db`:

- `has`, `store`, `get_new`, `search`, `history`, `prune`, `count`, `clear`
- Dedup key: the tool's `id`. A tool whose `source_code_url` is already cached is not added a
  second time.
- Entries older than `cache_ttl_days` (default 90) are removed when the engine opens the cache.
- Full-text search on name and description with SQLite FTS5.
- **Planned**: merge metadata when the same code repository is found on several platforms.

## 7. Destinations and formats

### 7.1 Interface — Built

```python
class BaseDestination(ABC):
    name: str

    async def deliver(self, tools: list[AITool], **kwargs: Any) -> None: ...
```

### 7.2 Destinations

| Destination | CLI | Status |
|---|---|---|
| Console | default | **Built**: table, or one panel per tool with `--detail` |
| Folder | `folder:PATH` | **Built**: one timestamped file per run, in json, csv or md |
| JSON file | `json:FILE` | **Built**: appends without duplicates; refuses to overwrite a corrupt file |
| Notion | — | Planned |
| Email (SMTP) | — | Planned |
| Posting to X or LinkedIn | — | **Not planned** |

### 7.3 Output formats — Built

`json` (full model), `csv` (flat, for spreadsheets) and `md` (Markdown list with links), from
`aitoolsfetcher.formatters`. They are available on `search`, `trending` and `export`, and in the
folder destination.

## 8. CLI — Built

```
aitoolsfetcher [--verbose] COMMAND [OPTIONS]

  search QUERY   [--sources] [--limit] [--min-stars] [--min-score] [--format] [--dest] [--detail]
  trending       [--sources] [--limit] [--format] [--dest] [--detail]
  history        [--limit] [--detail]
  export         --output FILE [--format json|csv|md] [--limit]
  periodic QUERY --every INTERVAL --dest DEST [--sources] [--limit]
  schedule QUERY --cron EXPR --dest DEST [--sources] [--limit]
  sources        list sources and whether they are available
```

`--format` takes `table`, `json`, `csv` or `md`. `--dest` takes `folder:PATH` or `json:FILE`.
See the README for real example output.

## 9. Configuration — Built

Optional file at `~/.aitoolsfetcher/config.toml`; see [`config.example.toml`](config.example.toml).
Environment variables take precedence: `GITHUB_TOKEN` (or `AITF_GITHUB_TOKEN`), `AITF_LLM_KEY`,
`AITF_LLM_PROVIDER`. Keys never go in the repository.

## 10. Dependencies

Core: `httpx`, `pydantic`, `click`, `rich`, `apscheduler`, `aiosqlite`, `beautifulsoup4`, and
`tomli` on Python 3.10.

Optional extras: `llm` (Anthropic and OpenAI SDKs) is used today. `notion` and `email` are
reserved for the planned destinations.

## 11. Error handling — Built

- Each source catches its own errors; partial results are always returned.
- Rate-limit detection with backoff, as described in 3.3.
- Configurable request timeout (`request_timeout`).
- Logging via the standard library; `--verbose` turns on debug output.

## 12. Data, privacy and terms

- Only public project metadata is read. Authors appear only as the account or organisation
  handle the platform shows.
- Everything is stored locally in the user's cache, and nowhere else.
- Nothing is collected from X/Twitter or LinkedIn.
- Every source follows its platform's current terms and rate limits, as listed in 3.2.
- No API keys in code or defaults. They come from env vars or the user's config file.
- MIT licence.
