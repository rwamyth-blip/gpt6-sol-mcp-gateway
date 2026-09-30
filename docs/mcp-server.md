# MCP Server

The gateway can also run *as* an MCP server, so other MCP clients (Claude Desktop, Cursor, VS Code,
another agent) can use GPT6-SOL as a tool.

## Start it

```bash
mcp-for-copilot mcp
```

This speaks MCP over **stdio**. The process reads JSON-RPC from stdin and writes to stdout, so it must
be launched by an MCP client rather than run interactively.

## Exposed tools

| Tool | Arguments | Returns |
| --- | --- | --- |
| `gpt6_chat` | `prompt` (required), `model`, `system` | JSON object with `content`, `model`, `rounds`, `used_tools`, `plan`, `plan_progress` |
| `gpt6_models` | — | JSON object of model id → metadata |
| `gpt6_status` | — | JSON object of configuration booleans |
| `gpt6_tools` | — | JSON array of tools advertised by the connected MCP server |
| `gpt6_plan` | — | The working plan from the most recent `gpt6_chat` call |
| `gpt6_debug_marathon` | `tasks` (1-10 items) | Queue ordered debug work; returns `job_id` |
| `gpt6_debug_marathon_status` | `job_id` | Queue progress and model responses |

### Plan and verification fields

`gpt6_chat` always returns `plan` and `plan_progress`. Both are empty unless `MCP_PLAN_MODE` is
enabled, in which case `plan` is a list of `{"description", "status"}` objects and `plan_progress`
counts steps per status. `gpt6_plan` re-reads the plan from the last `gpt6_chat` call without
re-running the model; it is read-only and never influences routing or approval.

`gpt6_debug_marathon_status` reports a `verification` field per task and per job. It is
`not_executed` unless a stage was actually given execution output, in which case it is `executed`.
The default is deliberately pessimistic: a model claiming success does **not** set `executed`. Each
task also carries a `plan` list parsed from the model's `MARATHON_PLAN` line.

Each task accepts `question` plus optional `title`, `priority`, `difficulty`, and `complexity` (scores
1-5, default 3). Queue order is priority, difficulty, and complexity, descending. Per task, the
default target paths are Luna-only for scores 1-2, Luna -> Sol for 3-4, and Luna -> Sol -> Astra for 5.
An uncertain model result escalates regardless of the target tier. These defaults aim for an
80/15/5 task mix; they do not enforce quotas or guarantee a measured success rate. A `completed`
result is model-self-reported and does not mean tests were run. Job state lives in the MCP process and
is not durable across restarts. Status results include processed/completed counts, model-reported
success rate, token usage, and estimated cost from the local model catalog.

## Client configuration

=== "Claude Desktop"

    Add to `claude_desktop_config.json`:

    ```json
    {
      "mcpServers": {
        "gpt6-sol": {
          "command": "mcp-for-copilot",
          "args": ["mcp"],
          "env": {"OPENAI_API_KEY": "sk-..."}
        }
      }
    }
    ```

=== "VS Code"

    Add to `.vscode/mcp.json`:

    ```json
    {
      "servers": {
        "gpt6-sol": {
          "type": "stdio",
          "command": "mcp-for-copilot",
          "args": ["mcp"],
          "env": {"OPENAI_API_KEY": "sk-..."}
        }
      }
    }
    ```

=== "Cursor"

    Add to `~/.cursor/mcp.json`:

    ```json
    {
      "mcpServers": {
        "gpt6-sol": {
          "command": "mcp-for-copilot",
          "args": ["mcp"],
          "env": {"OPENAI_API_KEY": "sk-..."}
        }
      }
    }
    ```

## Programmatic use

```python
import asyncio
from mcp_for_copilot.gateway.server import build_server

server = build_server()
asyncio.run(server.run_stdio_async())
```

## Verifying the server

The test suite includes an end-to-end test that spawns this server as a real subprocess and performs a
full MCP handshake over stdio:

```bash
pytest tests/e2e -v
```

## Notes

- The server uses the `mcp` 2.x callback API (`on_list_tools` / `on_call_tool` passed to the `Server`
  constructor). The older decorator API (`@server.list_tools()`) was removed in `mcp` 2.0.
- `gpt6_chat` requires a non-empty `prompt`; an empty value returns an error result rather than
  raising, so the client sees a normal tool error.
- Tool results are returned as MCP `TextContent` blocks. `gpt6_models`, `gpt6_status`, and
  `gpt6_tools` return JSON-encoded strings.
