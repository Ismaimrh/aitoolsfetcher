# AIToolsFetcher

Discover and track trending AI tools across GitHub, HuggingFace, Reddit, arXiv, and more.

## Install

```bash
pip install aitoolsfetcher
```

## Quick Start

### CLI

```bash
# Search for AI tools
aitoolsfetcher search "AI agents"

# Fetch trending AI tools
aitoolsfetcher trending

# Periodic fetching every 6 hours
aitoolsfetcher periodic "new AI tools" --every 6h --dest folder:~/ai-tools/

# Export cache to JSON
aitoolsfetcher export -o tools.json
```

### Python API

```python
import asyncio
from aitoolsfetcher import AIToolsFetcher

async def main():
    async with AIToolsFetcher() as fetcher:
        result = await fetcher.search("AI code generation")
        for tool in result.tools:
            print(f"{tool.name} - {tool.source_code_or_source}")

asyncio.run(main())
```

## Features

- **Multi-source**: GitHub (API), HuggingFace, Reddit, arXiv, Product Hunt, X, LinkedIn
- **Smart classification**: Keyword-based (default) + LLM-enhanced (optional)
- **Three fetch modes**: On-demand, scheduled (cron), periodic (interval)
- **Flexible output**: CLI table, JSON, CSV, Markdown
- **Destination adapters**: Local folder, JSON file, Notion, email, social posting
- **Cache & dedup**: SQLite with FTS5, cross-platform deduplication

## Optional Dependencies

```bash
pip install aitoolsfetcher[llm]      # LLM classification (Anthropic/OpenAI)
pip install aitoolsfetcher[notion]   # Notion integration
pip install aitoolsfetcher[email]    # Email delivery
pip install aitoolsfetcher[all]      # Everything
```

## Configuration

Create `~/.aitoolsfetcher/config.toml` or use environment variables:

```bash
export GITHUB_TOKEN="ghp_..."        # Higher rate limits
export AITF_LLM_KEY="sk-..."         # Enable LLM classification
export AITF_LLM_PROVIDER="anthropic" # or "openai"
```

## License

MIT
