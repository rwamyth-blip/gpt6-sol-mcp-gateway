"""Unit tests for Settings and the secret-redaction helpers."""

from __future__ import annotations

import logging

import pytest
from pydantic import ValidationError

from mcp_for_copilot.config import Settings
from mcp_for_copilot.logging_utils import (
    contains_secret,
    redact,
    redact_mapping,
    safe_log,
)


class TestSettings:
    def test_defaults_are_safe(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        # Settings() reads the developer's real .env, so isolate it to assert true defaults.
        monkeypatch.chdir(tmp_path)
        for name in (
            "LLM_PROVIDER",
            "LLM_MODEL_ID",
            "LLM_API_KEY",
            "MCP_TRANSPORT",
            "MCP_REQUIRE_APPROVAL",
            "MCP_MAX_TOOL_ROUNDS",
        ):
            monkeypatch.delenv(name, raising=False)

        settings = Settings(_env_file=None)
        assert settings.llm_provider == "openai"
        assert settings.llm_model_id == "gpt-6-sol"
        assert settings.llm_api_key == ""
        assert settings.mcp_transport == "stdio"
        assert settings.mcp_require_approval is True
        assert settings.mcp_max_tool_rounds == 5

    def test_transport_is_validated(self) -> None:
        with pytest.raises(ValueError, match="mcp_transport must be one of"):
            Settings(mcp_transport="smoke-signals")

    def test_transport_is_normalised(self) -> None:
        assert Settings(mcp_transport="HTTP").mcp_transport == "http"

    def test_allowed_tools_parsing(self) -> None:
        settings = Settings(mcp_allowed_tools_raw=" read_file , mongo_find ,, ")
        assert settings.mcp_allowed_tools == ("read_file", "mongo_find")

    def test_allowed_tools_empty_means_default(self) -> None:
        assert Settings(mcp_allowed_tools_raw="").mcp_allowed_tools == ()

    def test_cors_origins_parsing(self) -> None:
        settings = Settings(gateway_cors_origins_raw="https://a.test, https://b.test")
        assert settings.gateway_cors_origins == ["https://a.test", "https://b.test"]

    def test_llm_configured_requires_key_and_model(self) -> None:
        assert Settings(llm_api_key="k", llm_model_id="gpt-6-sol").llm_configured is True
        assert Settings(llm_api_key="", llm_model_id="gpt-6-sol").llm_configured is False

    def test_mcp_configured_requires_url(self) -> None:
        assert Settings(mcp_server_url="python -m x").mcp_configured is True
        assert Settings(mcp_server_url="").mcp_configured is False

    def test_describe_never_leaks_secrets(self) -> None:
        settings = Settings(
            llm_api_key="sk-super-secret-value",
            mcp_auth_token="tok-secret",
            gateway_api_key="gw-secret",
        )
        described = settings.describe()
        blob = str(described)
        assert "sk-super-secret-value" not in blob
        assert "tok-secret" not in blob
        assert "gw-secret" not in blob
        assert described["llm_api_key_present"] is True
        assert described["gateway_api_key_required"] is True

    def test_port_bounds(self) -> None:
        with pytest.raises(ValueError):
            Settings(gateway_port=0)
        with pytest.raises(ValueError):
            Settings(gateway_port=70000)

    def test_mcp_servers_parses_json_array(self) -> None:
        settings = Settings(mcp_servers='[{"name":"ollama","transport":"stdio","command":"x"}]')
        assert settings.mcp_configured is True
        assert settings.mcp_servers_parsed == [
            {"name": "ollama", "transport": "stdio", "command": "x"}
        ]

    def test_mcp_servers_malformed_json_is_tolerated(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Regression: this warning path used log_warning without importing it,
        # so a typo in MCP_SERVERS raised NameError instead of being skipped.
        settings = Settings(mcp_servers="[not json")
        with caplog.at_level(logging.WARNING, logger="mcp_for_copilot"):
            parsed = settings.mcp_servers_parsed
        assert parsed == []
        assert "Failed to parse MCP_SERVERS" in caplog.text

    def test_mcp_servers_non_array_is_tolerated(self, caplog: pytest.LogCaptureFixture) -> None:
        settings = Settings(mcp_servers='{"name":"ollama"}')
        with caplog.at_level(logging.WARNING, logger="mcp_for_copilot"):
            parsed = settings.mcp_servers_parsed
        assert parsed == []
        assert "not a JSON array" in caplog.text

    def test_describe_hides_mcp_server_tokens(self) -> None:
        # describe() is served by GET /v1/status, so MCP_SERVERS must never be
        # echoed verbatim: an entry can carry auth_token.
        settings = Settings(
            mcp_servers=(
                '[{"name":"remote","transport":"http",'
                '"url":"https://mcp.example.test/mcp","auth_token":"tok-mcp-secret"}]'
            )
        )
        described = settings.describe()
        assert "tok-mcp-secret" not in str(described)
        assert described["mcp_servers"] == [{"name": "remote", "transport": "http"}]

    def test_plan_mode_defaults_to_off(self) -> None:
        assert Settings().mcp_plan_mode is False

    def test_describe_reports_plan_mode(self) -> None:
        assert Settings().describe()["mcp_plan_mode"] is False
        assert Settings(mcp_plan_mode=True).describe()["mcp_plan_mode"] is True

    def test_loop_detection_defaults_match_cline(self) -> None:
        settings = Settings()
        assert settings.mcp_loop_detection is True
        assert settings.mcp_loop_soft_threshold == 3
        assert settings.mcp_loop_hard_threshold == 5

    def test_describe_reports_loop_detection(self) -> None:
        described = Settings().describe()
        assert described["mcp_loop_detection"] is True
        assert described["mcp_loop_soft_threshold"] == 3
        assert described["mcp_loop_hard_threshold"] == 5

    def test_hard_threshold_must_exceed_soft(self) -> None:
        with pytest.raises(ValidationError):
            Settings(mcp_loop_soft_threshold=5, mcp_loop_hard_threshold=5)
        with pytest.raises(ValidationError):
            Settings(mcp_loop_soft_threshold=5, mcp_loop_hard_threshold=4)

    def test_loop_thresholds_accept_a_valid_pair(self) -> None:
        settings = Settings(mcp_loop_soft_threshold=2, mcp_loop_hard_threshold=4)
        assert settings.mcp_loop_soft_threshold == 2
        assert settings.mcp_loop_hard_threshold == 4


class TestRedact:
    @pytest.mark.parametrize(
        "secret",
        [
            "sk-abcdefghijklmnopqrstuvwxyz012345",
            "sk-proj-abcdefghijklmnopqrstuvwxyz",
            "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.payload.signature",
            "ghp_abcdefghijklmnopqrstuvwxyz0123456789",
            "AKIAIOSFODNN7EXAMPLE",
        ],
    )
    def test_known_secret_shapes_are_redacted(self, secret: str) -> None:
        assert secret not in redact(f"the value is {secret} ok")

    def test_plain_text_is_unchanged(self) -> None:
        assert redact("nothing to hide here") == "nothing to hide here"

    def test_empty_string(self) -> None:
        assert redact("") == ""

    def test_redact_mapping_recurses(self) -> None:
        payload = {"outer": {"key": "sk-abcdefghijklmnopqrstuvwxyz012345"}}
        assert "sk-abcdefghijklmnopqrstuvwxyz012345" not in str(redact_mapping(payload))

    def test_redact_mapping_handles_lists(self) -> None:
        payload = {"items": ["sk-abcdefghijklmnopqrstuvwxyz012345"]}
        assert "sk-abcdefghijklmnopqrstuvwxyz012345" not in str(redact_mapping(payload))

    def test_contains_secret_detects(self) -> None:
        assert contains_secret("sk-abcdefghijklmnopqrstuvwxyz012345") is True
        assert contains_secret("just words") is False

    def test_safe_log_redacts(self, caplog: pytest.LogCaptureFixture) -> None:
        secret = "sk-abcdefghijklmnopqrstuvwxyz012345"
        with caplog.at_level(logging.WARNING, logger="mcp_for_copilot"):
            safe_log(logging.WARNING, "key=%s", secret)
        assert secret not in caplog.text
        assert "REDACTED" in caplog.text

    def test_safe_log_keeps_scalar_types(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO, logger="mcp_for_copilot"):
            safe_log(logging.INFO, "rounds=%d ratio=%.2f", 3, 0.5)
        assert "rounds=3 ratio=0.50" in caplog.text
