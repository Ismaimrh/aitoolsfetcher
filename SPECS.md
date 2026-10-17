# AIToolsFetcher — Technical Specifications

## 1. Overview

**AIToolsFetcher** is an open-source Python package that aggregates, classifies, and delivers trending AI tools from multiple platforms. It helps developers stay current with the rapidly evolving AI tool ecosystem.

```bash
pip install aitoolsfetcher
```

```python
from aitoolsfetcher import AIToolsFetcher

fetcher = AIToolsFetcher()
tools = await fetcher.search("AI code generation agents")
```

```bash
aitoolsfetcher search "vector database" --sources github,huggingface --format json
```

---

## 2. Fetch Modes

### 2.1 On-Demand (User-Triggered)
```python
# Programmatic
tools = await fetcher.search(query="AI agents", sources=["github", "reddit"])

# CLI
aitoolsfetcher search "AI agents" --sources github,reddit
```
- Executes immediately, returns results
- Supports filtering by source, category, date range, stars/popularity
- Results are cached automatically

### 2.2 Scheduled (Cron-Like)
```python
fetcher.schedule(
    query="new AI tools",
    cron="0 9 * * MON",  # Every Monday at 9am
    destination=FolderDestination("~/ai-tools-weekly/"),
    # OR
    destination=NotionDestination(database_id="xxx"),
    # OR
    destination=EmailDestination(to="dev@example.com"),
)
fetcher.run()  # Blocks and runs scheduler
```

```bash
aitoolsfetcher schedule "new AI tools" \
  --cron "0 9 * * MON" \
  --dest folder:~/ai-tools-weekly/
```
- User defines cron expression + destination
- Runs as a background daemon or foreground process
- Only delivers NEW tools (cache-aware dedup)

### 2.3 Periodic (Interval-Based)
```python
fetcher.periodic(
    query="AI tools",
    interval="6h",  # or timedelta(hours=6)
    destination=JSONFileDestination("~/ai-tools.json"),
)
```

```bash
aitoolsfetcher periodic "AI tools" --every 6h --dest json:~/ai-tools.json
```
- Simpler than scheduled — just a fixed interval
- Same destination and dedup behavior as scheduled

---

## 3. Data Sources

### 3.1 Source Adapter Interface
```python
class BaseSource(ABC):
    name: str
    requires_auth: bool

    @abstractmethod
    async def search(self, query: str, **filters) -> list[AITool]: ...

    @abstractmethod
    async def trending(self, **filters) -> list[AITool]: ...

    async def traceback_source_code(self, tool: AITool) -> Optional[str]:
        """Attempt to find source code URL for tools discovered on non-code platforms."""
        ...
```

### 3.2 Sources

| Source | Method | Auth Required | Reliability |
|---|---|---|---|
| **GitHub** | REST API v3 + GraphQL | Optional (higher rate limits with token) | High |
| **HuggingFace** | HF API | Optional | High |
| **arXiv** | arXiv API (OAI-PMH) | No | High |
| **Reddit** | Scraping (r/MachineLearning, r/artificial, r/LocalLLaMA, etc.) | No | Medium |
| **Product Hunt** | Scraping | No | Medium |
| **X / Twitter** | Scraping (nitter instances / search scraping) | No | Low (fragile) |
| **LinkedIn** | Scraping (public posts) | No | Low (fragile) |

### 3.3 GitHub Adapter (Primary)
- **Trending**: scrape github.com/trending + API search for recently created repos with AI topics
- **Search**: GitHub Search API with qualifiers (`topic:ai`, `topic:llm`, `topic:agent`, etc.)
- **Metadata**: stars, forks, language, last commit, description, topics, license
- **Rate limiting**: respect API limits, exponential backoff, optional token for 5000 req/hr

### 3.4 Source Code Traceback
For tools discovered on non-code platforms (X, LinkedIn, Reddit, Product Hunt):
1. Extract URLs from post content
2. Check if any URL points to GitHub/GitLab/HuggingFace
3. If not, search GitHub for the tool name + keywords from description
4. Return best match or `None`

---

## 4. Data Model

```python
class AITool(BaseModel):
    """Core data model for a discovered AI tool."""
    id: str                          # Content hash for dedup
    name: str
    description: str
    category: AICategory             # Enum: AGENT, FRAMEWORK, MODEL, DATASET, INFRA, DEV_TOOL, OTHER
    source_platform: str             # "github", "reddit", etc.
    source_url: str                  # Original discovery URL
    source_code_url: Optional[str]   # GitHub/GitLab URL (traced back if needed)
    homepage_url: Optional[str]
    stars: Optional[int]             # GitHub stars or equivalent metric
    language: Optional[str]          # Primary programming language
    topics: list[str]                # Tags/topics
    license: Optional[str]
    author: str
    discovered_at: datetime
    last_updated: Optional[datetime]
    popularity_score: float          # Normalized 0-1 score across platforms
    ai_relevance_score: float        # Classification confidence 0-1
    metadata: dict                   # Platform-specific extra data

class AICategory(str, Enum):
    AGENT = "agent"
    FRAMEWORK = "framework"
    MODEL = "model"
    DATASET = "dataset"
    INFRA = "infra"
    DEV_TOOL = "dev_tool"
    LIBRARY = "library"
    CLI_TOOL = "cli_tool"
    OTHER = "other"
```

---

## 5. Classification Pipeline

### 5.1 Keyword Classifier (Default)
```python
class KeywordClassifier(BaseClassifier):
    """Fast, zero-dependency classifier using curated keyword lists."""
```
- Curated taxonomy of AI-related keywords organized by category
- TF-IDF-like scoring on repo name + description + README (first 500 chars) + topics
- Thresholds tuned to minimize false positives (precision > recall)
- Returns `ai_relevance_score` (0-1) and best-match `AICategory`

**Keyword domains:**
- Frameworks: pytorch, tensorflow, jax, langchain, llamaindex, autogen, crewai...
- Models: llm, transformer, diffusion, gpt, claude, llama, mistral, embedding...
- Agents: agent, autonomous, tool-use, function-calling, mcp, a2a...
- Infra: vector-db, rag, fine-tuning, deployment, inference, serving, quantization...
- Dev tools: prompt engineering, evaluation, testing, monitoring, observability...

### 5.2 LLM Classifier (Optional Enhancement)
```python
class LLMClassifier(BaseClassifier):
    """High-accuracy classifier using Claude/OpenAI API."""

    def __init__(self, api_key: str, provider: str = "anthropic"): ...
```
- Activated only when user provides an LLM API key
- Sends tool name + description + README excerpt to LLM
- LLM returns: `is_ai_tool` (bool), `category`, `confidence`, `one_line_summary`
- Cached per tool (never re-classifies the same content)
- Supports Anthropic (Claude) and OpenAI as providers
- Batch classification to minimize API calls

### 5.3 Pipeline Flow
```
Raw results → Keyword filter (fast reject non-AI) → LLM classify (if available) → Final scored results
```

---

## 6. Cache Layer

### 6.1 SQLite Cache
```python
class ToolCache:
    """Local SQLite cache for deduplication and history."""

    async def has(self, tool_id: str) -> bool: ...
    async def store(self, tools: list[AITool]) -> int: ...  # returns new count
    async def get_new(self, tools: list[AITool]) -> list[AITool]: ...
    async def search(self, query: str) -> list[AITool]: ...
    async def history(self, since: datetime) -> list[AITool]: ...
    async def clear(self) -> None: ...
```

- Stored at `~/.aitoolsfetcher/cache.db`
- Dedup key: content hash of `(name, source_platform, author)` — same tool on different platforms gets separate entries but linked via `source_code_url`
- Cross-platform dedup: if same `source_code_url` exists, merge metadata instead of creating duplicate
- TTL-based expiry: entries older than 90 days auto-pruned (configurable)
- Full-text search on name + description via SQLite FTS5

---

## 7. Destination Adapters

### 7.1 Interface
```python
class BaseDestination(ABC):
    @abstractmethod
    async def deliver(self, tools: list[AITool], format: str = "json") -> None: ...
```

### 7.2 Destinations (v1)

| Destination | Description | Config |
|---|---|---|
| **Folder** | Save results as files to a local directory | `path` |
| **JSON File** | Append/write results to a JSON file | `path`, `mode` (append/overwrite) |
| **Console** | Rich table in terminal (default for CLI) | - |
| **Notion** | Add entries to a Notion database | `api_key`, `database_id` |
| **Email** | Send digest email | `smtp_config`, `recipients` |
| **Social (X)** | Post curated tools to X feed | `api_keys` |

### 7.3 Output Formats
- **JSON**: Full `AITool` model serialized
- **CSV**: Flat export for spreadsheets
- **Markdown**: Formatted list with links (great for Notion/GitHub)
- **CLI Table**: `rich` table with columns: Name | Category | Description (truncated) | Source Link

---

## 8. CLI Interface

```
aitoolsfetcher [COMMAND] [OPTIONS]

Commands:
  search      On-demand search for AI tools
  trending    Fetch currently trending AI tools
  schedule    Set up scheduled fetching with destination
  periodic    Set up interval-based fetching
  history     View previously fetched tools from cache
  config      Manage configuration (API keys, defaults)
  sources     List available sources and their status
  export      Export cache to file (json/csv/md)

Global Options:
  --sources       Comma-separated source list (default: all enabled)
  --format        Output format: json, csv, md, table (default: table)
  --limit         Max results per source (default: 20)
  --min-score     Minimum AI relevance score 0-1 (default: 0.5)
  --category      Filter by category
  --since         Only tools discovered after this date
  --dest          Destination adapter (folder:/path, json:/path, notion:db_id, email:addr)
  --verbose       Show detailed output
  --no-cache      Skip cache, fetch fresh
```

### CLI Table Output Example
```
┌─────────────────────┬───────────┬──────────────────────────────────────┬────────────────────────────────────┐
│ Name                │ Category  │ What it does                         │ Source                             │
├─────────────────────┼───────────┼──────────────────────────────────────┼────────────────────────────────────┤
│ AutoAgent           │ Agent     │ Build autonomous AI agents with MCP  │ github.com/auto/autoagent          │
│ VectorForge         │ Infra     │ High-perf vector DB for RAG pipes    │ github.com/vf/vectorforge          │
│ PromptBench         │ Dev Tool  │ Evaluate and compare LLM prompts     │ huggingface.co/spaces/promptbench  │
│ NeuralDeploy        │ Infra     │ One-click LLM deployment to cloud    │ producthunt.com/posts/neuraldeploy │
└─────────────────────┴───────────┴──────────────────────────────────────┴────────────────────────────────────┘

4 new AI tools found (12 total fetched, 8 cached)
```

---

## 9. Configuration

Stored at `~/.aitoolsfetcher/config.toml`:

```toml
[general]
default_sources = ["github", "huggingface", "reddit", "arxiv", "producthunt"]
default_format = "table"
cache_ttl_days = 90
min_relevance_score = 0.5
max_results_per_source = 20

[api_keys]
github_token = ""          # Optional, increases rate limits
huggingface_token = ""     # Optional
notion_api_key = ""        # Required for Notion destination
llm_api_key = ""           # Optional, enables LLM classifier
llm_provider = "anthropic" # "anthropic" or "openai"

[sources.twitter]
enabled = false            # Fragile, disabled by default
# nitter_instance = ""

[sources.linkedin]
enabled = false            # Fragile, disabled by default

[destinations.email]
smtp_host = ""
smtp_port = 587
smtp_user = ""
smtp_password = ""
from_address = ""
```

---

## 10. Dependency Strategy

### Core (always installed)
- `httpx` — async HTTP client
- `pydantic` — data validation/models
- `click` — CLI framework
- `rich` — terminal formatting
- `apscheduler` — scheduling
- `aiosqlite` — async SQLite
- `beautifulsoup4` — HTML parsing for scrapers
- `tomli` / `tomllib` — config parsing

### Optional (extras)
```bash
pip install aitoolsfetcher[notion]    # notion-client
pip install aitoolsfetcher[email]     # aiosmtplib
pip install aitoolsfetcher[llm]       # anthropic, openai
pip install aitoolsfetcher[all]       # everything
```

---

## 11. Implementation Phases

### Phase 1 — Core + GitHub (MVP)
- [ ] Project scaffolding (pyproject.toml, src layout, CI)
- [ ] Data models (`AITool`, `AICategory`)
- [ ] Cache layer (SQLite + FTS5)
- [ ] GitHub source adapter (API search + trending)
- [ ] Keyword classifier
- [ ] CLI: `search`, `trending`, `history` commands
- [ ] Console table output
- [ ] JSON file destination
- [ ] Folder destination
- [ ] Configuration management

### Phase 2 — More Sources + Classification
- [ ] Reddit adapter (scraping)
- [ ] HuggingFace adapter (API)
- [ ] arXiv adapter (API)
- [ ] Product Hunt adapter (scraping)
- [ ] LLM classifier (Anthropic + OpenAI)
- [ ] Source code traceback for non-GitHub tools
- [ ] CSV + Markdown output formats

### Phase 3 — Scheduling + Destinations
- [ ] Scheduled fetching (cron)
- [ ] Periodic fetching (interval)
- [ ] Notion destination
- [ ] Email destination
- [ ] CLI: `schedule`, `periodic`, `export` commands
- [ ] Cross-platform dedup (merge same tool from different sources)

### Phase 4 — Social + Polish
- [ ] X/Twitter adapter (best-effort scraping)
- [ ] LinkedIn adapter (best-effort scraping)
- [ ] Social posting destination (X)
- [ ] README, docs, examples
- [ ] PyPI publishing
- [ ] GitHub Actions CI/CD

---

## 12. Error Handling & Resilience

- Each source adapter catches its own exceptions — one failing source never crashes the pipeline
- Rate limit detection with automatic backoff (exponential + jitter)
- Scraping adapters detect blocking/CAPTCHAs and gracefully degrade
- All network operations have configurable timeouts (default: 30s)
- Partial results are always returned — if 3/5 sources succeed, return those 3
- Structured logging via `logging` stdlib (no heavy deps)

---

## 13. Security & Legal

- No API keys in code or defaults — always user-provided via config or env vars
- Scraping adapters respect `robots.txt` where applicable
- Rate limiting is conservative by default to avoid IP bans
- X and LinkedIn adapters disabled by default with clear documentation about ToS risks
- All scraped data is publicly available information only
- MIT License for the package itself
