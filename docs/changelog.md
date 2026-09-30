# All notable changes to this project are documented in this file.
#
# The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
# and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Nothing yet.

## [0.1.0] - 2025-01-01

First public release.

### Added

- **Library mode** — `Gateway` facade that owns the provider, MCP client,
  router, approval layer, and orchestrator for one process.
- **Server mode** — FastAPI app exposing an OpenAI-compatible
  `/v1/chat/completions` (with SSE streaming), `/v1/models`, and `/v1/status`.
- **stdio MCP server** — exposes the gateway itself as MCP tools
  (`gpt6_chat`, `gpt6_models`, `gpt6_status`, `gpt6_tools`).
- **Three MCP transports** — `stdio`, `http` (streamable), and `sse`.
- **Three-gate tool safety** — allowlist, approval, and dispatch, each failing
  closed.
- **Risky-tool detection** — 40+ mutating verbs plus explicit risky/safe sets.
- **Argument validation** — JSON-schema subset validation with an explicit
  boolean check, since `bool` is a subclass of `int` in Python.
- **Approval layer** — `allow_all_approver`, `deny_all_approver`, and
  `static_approver`; denies by default when no approver is configured.
- **Prompt-injection guard** — tool output is wrapped and labelled as untrusted
  data before it re-enters the model context.
- **Secret redaction** — 11 regex rules applied to every log line and error
  message.
- **Model catalog** — GPT-6 Astra/Sol/Luna and GPT-5.6 Sol/Luna with context
  windows, output caps, reasoning-effort support, and pricing metadata.
- **Model aliases** — `gpt-6`, `gpt6`, `sol`, `luna`, `astra`, `gpt-5.6`.
- **CLI** — `mcp-for-copilot` with `serve`, `mcp`, `chat`, `models`, and
  `status` subcommands.
- **Docker** — multi-stage image, non-root user, healthcheck, and a
  `docker-compose.yml` with `gateway` and `mcp` services.
- **CI/CD** — `ci.yml` (ruff + mypy + pytest + coverage), `release.yml` (PyPI
  Trusted Publishing), `codeql.yml`, and `docs.yml`.
- **Documentation** — MkDocs Material site with guides and an API reference.

### Fixed

- `reasoning_effort` is forced to `"none"` whenever tools are present. The
  GPT-6 family rejects function tools combined with `reasoning_effort` on
  `/v1/chat/completions`, and the field must be present — omitting it is also
  rejected.

[Unreleased]: https://github.com/rwamyth-blip/mcp-for-copilot/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/rwamyth-blip/mcp-for-copilot/releases/tag/v0.1.0
