"""Tests for settings loading. Only fake keys are used, in temp files."""

from __future__ import annotations

from aitoolsfetcher.config import Settings, SourceConfig, load_settings


def test_defaults_without_config_file(isolated_user_dir):
    settings = load_settings()
    assert settings.github_token == ""
    assert not settings.has_llm_key
    assert settings.is_source_enabled("github")


def test_reads_sections_from_toml(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(
        """
[general]
cache_ttl_days = 30
max_results_per_source = 5

[api_keys]
github_token = "fake-github-token"
llm_provider = "openai"

[sources.github]
enabled = false
""",
        encoding="utf-8",
    )
    settings = load_settings(path)
    assert settings.cache_ttl_days == 30
    assert settings.max_results_per_source == 5
    assert settings.github_token == "fake-github-token"
    assert settings.llm_provider == "openai"
    assert not settings.is_source_enabled("github")


def test_reads_flat_toml(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('github_token = "flat-token"\n', encoding="utf-8")
    assert load_settings(path).github_token == "flat-token"


def test_environment_overrides_file(tmp_path, monkeypatch):
    path = tmp_path / "config.toml"
    path.write_text('[api_keys]\ngithub_token = "from-file"\n', encoding="utf-8")
    monkeypatch.setenv("GITHUB_TOKEN", "from-env")
    monkeypatch.setenv("AITF_LLM_KEY", "fake-llm-key")
    settings = load_settings(path)
    assert settings.github_token == "from-env"
    assert settings.has_llm_key


def test_twitter_and_linkedin_never_enabled_by_default():
    settings = Settings(default_sources=["github", "twitter", "linkedin"])
    assert not settings.is_source_enabled("twitter")
    assert not settings.is_source_enabled("linkedin")
    assert Settings(sources={"reddit": SourceConfig(enabled=True)}).is_source_enabled("reddit")


def test_example_config_parses():
    """The committed example must load and must not contain real-looking secrets."""
    from pathlib import Path

    example = Path(__file__).resolve().parents[1] / "config.example.toml"
    settings = load_settings(example)
    assert settings.github_token == ""
    assert settings.llm_api_key == ""
