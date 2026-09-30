# All notable changes to this project are documented in this file.
#
# The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
# and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Multi-server MCP** — `MCP_SERVERS` accepts a JSON array of servers. The
  gateway fans out `list_tools()` and `call_tool()` across every reachable
  server, drops servers that fail to connect, and reports the connected set
  under `mcp.servers` with `mcp.transport = "multi"`.
- **Plan mode** — `MCP_PLAN_MODE` (default `false`) offers the model a synthetic
  `update_plan` tool and asks it to keep a short working plan. The plan is
  advisory only: `update_plan` is intercepted in the orchestrator and never
  reaches the allowlist, the approval layer, or a back-end MCP server.
- **Plan reporting** — `GatewayResult.plan` / `plan_progress`, the HTTP
  `x_gateway.plan` / `x_gateway.plan_progress` fields, the `gpt6_plan` MCP tool,
  and `chat --plan` / `chat --json` in the CLI.
- **Honest verification state** — debug marathon tasks and jobs now report
  `verification` (`not_executed` unless a stage was actually given execution
  output) and a per-task `plan` parsed from the model's `MARATHON_PLAN` line.
- **`marathon` CLI subcommand** — queue debug tasks and poll the job to
  completion.
- **`MCP_PLAN_MODE` in `describe()`** — `GET /v1/status` and `gpt6_status` now
  report whether plan mode is enabled.

### Changed

- `gpt6_chat` now returns a JSON object (with `plan` and `plan_progress`) instead
  of a bare text answer.
- `docs/configuration.md` documents `MCP_SERVERS` and `MCP_PLAN_MODE`.

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
