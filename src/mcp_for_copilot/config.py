"""Configuration — env-driven, no hardcoded secrets.

Every value has a safe default so the package imports cleanly with an empty
environment. A credential is never given a default: an unset key stays empty
and the layer that needs it fails loudly instead of silently using a literal.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .logging_utils import log_warning, redact


class Settings(BaseSettings):
    """Runtime configuration, read from the environment or a ``.env`` file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # -- LLM provider -----------------------------------------------------
    llm_provider: str = Field(
        default="openai",
        description="Provider adapter to use. Only 'openai' is implemented.",
    )
    llm_model_id: str = Field(
        default="gpt-6-sol",
        description="Default model id or alias (e.g. 'gpt-6-sol', 'sol').",
    )
    llm_api_key: str = Field(
        default="",
        description="Provider API key. Never commit this.",
    )
    llm_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="OpenAI-compatible base URL (no trailing /chat/completions).",
    )
    llm_timeout_seconds: int = Field(default=120, ge=1)
    llm_reasoning_effort: str = Field(
        default="",
        description=(
            "Reasoning effort forwarded to the model. Empty means 'do not send "
            "the field'. Ignored (forced to 'none') whenever tools are present."
        ),
    )

    # -- MCP servers ------------------------------------------------------
    # Support for multiple MCP servers. Either use the new MCP_SERVERS field
    # (recommended) or the deprecated single server fields below.
    mcp_servers: str = Field(
        default="",
        description=(
            "JSON array of MCP server configurations. Each object should have: "
            "{name: str, transport: 'stdio'|'http'|'sse', url: str (for http/sse) "
            "or command: str (for stdio), auth_token?: str}. "
            'Example: \'[{"name":"ollama","transport":"stdio","command":"ollama"},{"name":"codex","transport":"http","url":"http://127.0.0.1:7421/mcp"}]\''
        ),
    )
    # Deprecated single server fields (kept for backward compatibility)
    mcp_server_url: str = Field(
        default="",
        description="DEPRECATED: MCP server URL for http/sse, or a command for stdio.",
    )
    mcp_transport: str = Field(
        default="stdio",
        description="DEPRECATED: One of: stdio, http, sse.",
    )
    mcp_auth_token: str = Field(
        default="", description="DEPRECATED: Bearer token for the MCP server."
    )
    mcp_require_approval: bool = Field(
        default=True,
        description="When True, risky tools need an explicit approval decision.",
    )
    mcp_allowed_tools_raw: str = Field(
        default="",
        description=(
            "Comma-separated tool allowlist. Empty means the read-only default "
            "set in tool_router.DEFAULT_ALLOWED_TOOLS is used."
        ),
    )
    mcp_max_tool_rounds: int = Field(
        default=5,
        ge=0,
        description="Cap on model -> tool -> model rounds per turn.",
    )
    mcp_plan_mode: bool = Field(
        default=False,
        description=(
            "When True, the orchestrator offers a synthetic 'update_plan' tool "
            "and asks the model to keep a short working plan. The plan is "
            "advisory only and never bypasses the allowlist or approval gates."
        ),
    )
    mcp_loop_detection: bool = Field(
        default=True,
        description=(
            "When True, the orchestrator watches for repeated identical tool "
            "calls. A soft threshold injects a warning telling the model to "
            "change approach; a hard threshold stops the turn. This is what "
            "keeps a stuck model from burning every remaining round."
        ),
    )
    mcp_loop_soft_threshold: int = Field(
        default=3,
        ge=1,
        description=(
            "Number of identical tool calls (same tool and arguments) that "
            "triggers a soft warning. Must be lower than the hard threshold."
        ),
    )
    mcp_loop_hard_threshold: int = Field(
        default=5,
        ge=2,
        description=(
            "Number of identical tool calls (same tool and arguments) that "
            "stops the turn. Must be greater than the soft threshold."
        ),
    )
    mcp_timeout_seconds: int = Field(default=60, ge=1)

    # -- Gateway server ---------------------------------------------------
    gateway_host: str = Field(default="0.0.0.0")
    gateway_port: int = Field(default=8000, ge=1, le=65535)
    gateway_api_key: str = Field(
        default="",
        description=(
            "When set, callers must send 'Authorization: Bearer <key>'. "
            "When empty, any key is accepted (development only)."
        ),
    )
    gateway_cors_origins_raw: str = Field(
        default="*",
        description="Comma-separated allowed CORS origins.",
    )

    # -- validators -------------------------------------------------------
    @field_validator("llm_api_key", mode="before")
    @classmethod
    def _fallback_to_openai_key(cls, value: object) -> object:
        """Accept ``OPENAI_API_KEY`` when ``LLM_API_KEY`` is unset.

        The gateway is often started from a service wrapper or a new shell that
        has the provider key exported under the provider's own conventional
        name but never under the generic one. Without this fallback the key
        silently resolves to empty and every request fails with
        "LLM_API_KEY is not configured", which reads like a bug rather than a
        missing environment variable.

        An explicitly configured LLM_API_KEY always wins -- ``_load_dotenv``
        uses setdefault semantics, so a real environment value is never
        shadowed by a stale .env entry.
        """
        if isinstance(value, str) and value.strip():
            return value
        for name in ("OPENAI_API_KEY", "LLM_API_KEY"):
            fallback = os.environ.get(name, "").strip()
            if fallback:
                return fallback
        return value

    @field_validator("mcp_transport")
    @classmethod
    def _check_transport(cls, value: str) -> str:
        allowed = ("stdio", "http", "sse")
        normalised = (value or "").strip().lower()
        if normalised not in allowed:
            raise ValueError(f"mcp_transport must be one of {', '.join(allowed)}; got {value!r}")
        return normalised

    @field_validator("mcp_loop_hard_threshold")
    @classmethod
    def _check_loop_thresholds(cls, value: int, info: Any) -> int:
        """Keep the hard threshold strictly above the soft one.

        A hard threshold at or below the soft threshold would make the warning
        unreachable: the turn would stop before the model ever saw it.
        """
        soft = info.data.get("mcp_loop_soft_threshold")
        if isinstance(soft, int) and value <= soft:
            raise ValueError(
                "mcp_loop_hard_threshold must be greater than "
                f"mcp_loop_soft_threshold ({soft}); got {value}"
            )
        return value

    # -- derived ----------------------------------------------------------
    @property
    def mcp_allowed_tools(self) -> tuple[str, ...]:
        """Parsed allowlist. Empty tuple means 'use the default set'."""
        return tuple(t.strip() for t in self.mcp_allowed_tools_raw.split(",") if t.strip())

    @property
    def gateway_cors_origins(self) -> list[str]:
        return [o.strip() for o in self.gateway_cors_origins_raw.split(",") if o.strip()]

    @property
    def llm_configured(self) -> bool:
        return bool(self.llm_api_key.strip() and self.llm_model_id.strip())

    @property
    def mcp_configured(self) -> bool:
        # Backward compatibility: check if deprecated single server fields are set
        if bool(self.mcp_server_url.strip()):
            return True
        # New way: check if MCP_SERVERS is configured
        return bool(self.mcp_servers.strip())

    @property
    def mcp_servers_parsed(self) -> list[dict[str, Any]]:
        """Parsed MCP servers configuration. Returns empty list if not configured or invalid."""
        if not self.mcp_servers.strip():
            return []
        try:
            import json

            servers = json.loads(self.mcp_servers)
            if isinstance(servers, list):
                return servers
            # Never log the raw payload: an entry may carry auth_token, and the
            # ``key: value`` redaction rule does not match JSON quoting.
            log_warning("MCP_SERVERS is not a JSON array (got %s)", type(servers).__name__)
            return []
        except (json.JSONDecodeError, Exception) as exc:
            log_warning("Failed to parse MCP_SERVERS: %s: %s", type(exc).__name__, redact(str(exc)))
            return []

    def describe(self) -> dict[str, object]:
        """Non-secret summary, safe to log or return over HTTP."""
        return {
            "llm_provider": self.llm_provider,
            "llm_model_id": self.llm_model_id,
            "llm_base_url": self.llm_base_url,
            "llm_api_key_present": bool(self.llm_api_key),
            "llm_reasoning_effort": self.llm_reasoning_effort or None,
            "mcp_transport": self.mcp_transport,
            "mcp_server_url": redact(self.mcp_server_url),
            "mcp_servers": self._describe_mcp_servers(),
            "mcp_configured": self.mcp_configured,
            "mcp_require_approval": self.mcp_require_approval,
            "mcp_max_tool_rounds": self.mcp_max_tool_rounds,
            "mcp_plan_mode": self.mcp_plan_mode,
            "mcp_loop_detection": self.mcp_loop_detection,
            "mcp_loop_soft_threshold": self.mcp_loop_soft_threshold,
            "mcp_loop_hard_threshold": self.mcp_loop_hard_threshold,
            "gateway_api_key_required": bool(self.gateway_api_key),
        }

    def _describe_mcp_servers(self) -> list[dict[str, str]]:
        """Names and transports only — never tokens or raw command lines.

        The raw ``MCP_SERVERS`` JSON must never be echoed back: an entry may
        carry ``auth_token``, and ``describe()`` is served over HTTP by
        ``GET /v1/status``.
        """
        described: list[dict[str, str]] = []
        for server in self.mcp_servers_parsed:
            if not isinstance(server, dict):
                continue
            name = str(server.get("name") or "")
            if not name:
                continue
            described.append({"name": name, "transport": str(server.get("transport") or "stdio")})
        return described


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader used when pydantic-settings is not enough.

    ``setdefault`` semantics: a real environment variable always wins over the
    file, so CI secrets are never shadowed by a stale local ``.env``.
    """
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value:
            os.environ.setdefault(key, value)


def _env_file_candidates() -> list[Path]:
    """Every ``.env`` location that should be honoured, most specific first.

    ``Path.cwd()`` alone is not enough: the gateway is routinely started from a
    different working directory (a service wrapper, an IDE task, a shell that
    never ``cd``-ed), and in that case every setting silently fell back to its
    default -- including ``MCP_SERVER_URL``, which disabled tool calling with no
    error. The package root is derived from this file's location
    (``<root>/src/mcp_for_copilot/config.py``), so it is correct regardless of cwd.
    """
    package_root = Path(__file__).resolve().parents[2]
    candidates = [Path.cwd() / ".env", package_root / ".env"]
    unique: list[Path] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return unique


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    for candidate in _env_file_candidates():
        _load_dotenv(candidate)
    return Settings()


def reset_settings_cache() -> None:
    """Clear the settings cache. Intended for tests."""
    get_settings.cache_clear()
