# Examples

Three runnable examples, in increasing order of complexity. Each one needs only `LLM_API_KEY`.

| File | What it shows |
| --- | --- |
| [`01_basic_chat.py`](01_basic_chat.py) | Plain chat with no MCP server |
| [`02_tool_calling.py`](02_tool_calling.py) | Tool calling against a real stdio MCP server |
| [`03_custom_approval.py`](03_custom_approval.py) | A custom approval policy and the audit trail |

## Setup

```bash
pip install "mcp-for-copilot[all]"
export LLM_API_KEY="sk-..."
```

## Run

```bash
python examples/01_basic_chat.py
python examples/02_tool_calling.py
python examples/03_custom_approval.py
```

Examples 2 and 3 spawn the gateway's own bundled MCP server as a subprocess, so they need no external
MCP server. They set `MCP_SERVER_URL` to the current interpreter plus
`-m mcp_for_copilot.gateway.server`, which is the same command the CLI uses for `mcp-for-copilot mcp`.

## Notes

- Examples 2 and 3 make real API calls and therefore cost money.
- `reset_settings_cache()` is called after mutating `os.environ`, because `get_settings()` is cached.
- The approval example deliberately uses a narrow allowlist so the denial path is easy to observe.
