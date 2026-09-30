# FAQ

## General

### What is MCP?

The [Model Context Protocol](https://modelcontextprotocol.io/) is an open standard for connecting LLM
applications to external tools and data. An MCP *server* advertises tools; an MCP *client* discovers
and calls them. This gateway is both: it is an MCP client to your tools, and optionally an MCP server
to other agents.

### Which models are supported?

`gpt-6-astra`, `gpt-6-sol`, `gpt-6-luna`, `gpt-5.6-sol`, and `gpt-5.6-luna`. Aliases such as `sol`,
`luna`, `astra`, `gpt6`, and `gpt-5.6` resolve to the canonical ids.

### Does it work with non-OpenAI providers?

Not yet. `LLM_PROVIDER` accepts only `openai`. The provider layer is small and isolated, so adding
another OpenAI-compatible endpoint is a contained change — PRs welcome.

## Errors

### `400 'messages' must be a non-empty list`

The request body had no `messages`, an empty list, or a non-list value. Send at least one message.

### `400` with an unknown model

The `model` field did not match a known id or alias. Call `GET /v1/models` for the list.

### `502` from `/v1/chat/completions`

The upstream provider failed. The gateway itself is healthy, which is why the status is `502` rather
than `500`. Check `GatewayResult.error` (library) or the server logs for the upstream message.

### `reasoning_effort` is rejected when I send tools

This is an upstream constraint, not a gateway bug. The GPT-6 family rejects `reasoning_effort`
together with function tools on `/v1/chat/completions`. The gateway forces `reasoning_effort = "none"`
whenever `tools` are present. If you are calling the provider directly, you must do the same — the
field has to be **present and equal to `"none"`**; omitting it is also rejected.

### My tool call is denied and I did not configure an approver

That is the fail-closed default. Pass an approver such as `allow_all_approver()` in trusted
environments, or remove the risky tool from the allowlist.

### A tool is allowlisted but still denied

The tool must be **both** allowlisted **and** advertised by the connected MCP server. Without a schema
the gateway cannot validate arguments, so it refuses to call it.

## Configuration

### `.env` is not being read

`.env` is loaded from the **current working directory**. Real environment variables always take
precedence over `.env` values. Check both.

### I changed an environment variable but nothing happened

`get_settings()` is cached. Call `reset_settings_cache()` (library) or restart the process (server).

### How do I allow *no* tools?

Set `MCP_ALLOWED_TOOLS` to a value that parses to an empty tuple, or pass `allowed_tools=()` in code.
An unset variable means "use the read-only defaults", which is different.

## Deployment

### Streaming arrives all at once

Your reverse proxy is buffering. In nginx, set `proxy_buffering off;` for the streaming route.

### The container exits immediately

Check that `LLM_API_KEY` is set. The server starts without it, but the first request will fail with
`502`.

### Can I run the gateway and an MCP server in one container?

Yes — `docker-compose.yml` shows the two-service pattern. Keep them as separate services so each can
be scaled and restarted independently.

## Development

### How do I run the tests?

```bash
make test
```

The suite needs no network access. Two end-to-end tests spawn a real Python subprocess MCP server over
stdio to prove the transport works.

### How do I add a new model?

Add an entry to `KNOWN_MODELS` in `src/mcp_for_copilot/provider.py`, plus any aliases in `MODEL_ALIASES`.
Add a test in `tests/unit/test_provider.py`.
