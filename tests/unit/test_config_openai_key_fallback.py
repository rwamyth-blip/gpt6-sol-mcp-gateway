"""Unit tests for the OPENAI_API_KEY fallback in Settings.

Kept in its own module because tests/unit/test_config.py can be held open by
another process on this host (memory-mapped section), which makes in-place
edits fail; a separate file keeps the coverage without a fight.

The gateway reads LLM_API_KEY. Service wrappers, Startup-folder shortcuts and
fresh shells usually export only the provider's conventional name, so without
this fallback every request failed with "LLM_API_KEY is not configured" even
though a valid key was present in the environment.
"""

from __future__ import annotations

import pytest

from mcp_for_copilot.config import Settings


class TestOpenAIApiKeyFallback:
    def test_openai_api_key_env_is_used_when_llm_api_key_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        settings = Settings(_env_file=None)
        assert settings.llm_api_key == "sk-from-env"
        assert settings.llm_configured is True

    def test_explicit_llm_api_key_wins_over_openai_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
        settings = Settings(llm_api_key="sk-explicit", _env_file=None)
        assert settings.llm_api_key == "sk-explicit"

    def test_blank_llm_api_key_still_falls_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A .env line left as LLM_API_KEY= is the common real-world shape.
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
        settings = Settings(llm_api_key="   ", _env_file=None)
        assert settings.llm_api_key == "sk-from-env"

    def test_no_key_anywhere_stays_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        settings = Settings(_env_file=None)
        assert settings.llm_api_key == ""
        assert settings.llm_configured is False

    def test_fallback_value_is_never_logged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """describe() is served over HTTP by GET /v1/status, so the key must
        never appear in it regardless of which env var supplied it."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-super-secret-from-env")
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        described = Settings(_env_file=None).describe()
        assert described["llm_api_key_present"] is True
        assert "sk-super-secret-from-env" not in str(described)
