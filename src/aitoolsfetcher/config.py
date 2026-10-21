"""Configuration management for AIToolsFetcher."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - only on Python 3.10
    import tomli as tomllib

CONFIG_DIR = Path.home() / ".aitoolsfetcher"
CONFIG_FILE = CONFIG_DIR / "config.toml"
CACHE_DB = CONFIG_DIR / "cache.db"


class SourceConfig(BaseModel):
    """Per-source configuration."""

    enabled: bool = True


class Settings(BaseModel):
    """Application-wide settings."""

    default_sources: list[str] = Field(
        default_factory=lambda: ["github", "huggingface", "reddit", "arxiv", "producthunt"]
    )
    default_format: str = "table"
    cache_ttl_days: int = 90
    min_relevance_score: float = 0.5
    max_results_per_source: int = 20
    request_timeout: int = 30

    # API keys — loaded from config or env vars
    github_token: str = ""
    huggingface_token: str = ""
    notion_api_key: str = ""
    llm_api_key: str = ""
    llm_provider: str = "anthropic"

    # Source overrides
    sources: dict[str, SourceConfig] = Field(default_factory=dict)

    def is_source_enabled(self, name: str) -> bool:
        """Check if a source is enabled in config."""
        if name in self.sources:
            return self.sources[name].enabled
        # X and LinkedIn disabled by default (fragile scrapers)
        if name in ("twitter", "linkedin"):
            return False
        return name in self.default_sources

    @property
    def has_llm_key(self) -> bool:
        return bool(self.llm_api_key)


def _env_override(settings: Settings) -> Settings:
    """Override settings with environment variables."""
    env_map = {
        "AITF_GITHUB_TOKEN": "github_token",
        "GITHUB_TOKEN": "github_token",
        "AITF_HF_TOKEN": "huggingface_token",
        "HF_TOKEN": "huggingface_token",
        "AITF_NOTION_KEY": "notion_api_key",
        "AITF_LLM_KEY": "llm_api_key",
        "AITF_LLM_PROVIDER": "llm_provider",
    }
    updates: dict[str, Any] = {}
    for env_var, field_name in env_map.items():
        val = os.environ.get(env_var)
        if val:
            updates[field_name] = val
    if updates:
        settings = settings.model_copy(update=updates)
    return settings


def load_settings(config_path: Path | None = None) -> Settings:
    """Load settings from config file, then override with env vars."""
    path = config_path or CONFIG_FILE
    settings = Settings()

    if path.exists():
        with open(path, "rb") as f:
            data = tomllib.load(f)

        general = data.get("general", {})
        api_keys = data.get("api_keys", {})
        sources_raw = data.get("sources", {})

        # Collect top-level scalar keys as fallback (flat config format)
        top_level = {k: v for k, v in data.items() if not isinstance(v, dict)}

        merged = {**top_level, **general, **api_keys}
        sources_parsed = {
            k: SourceConfig(**v) for k, v in sources_raw.items() if isinstance(v, dict)
        }
        if sources_parsed:
            merged["sources"] = sources_parsed
        settings = Settings(**merged)

    return _env_override(settings)


def ensure_config_dir() -> Path:
    """Create config directory if it doesn't exist."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    return CONFIG_DIR
