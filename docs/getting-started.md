# Getting Started

This page takes you from zero to a working GPT6-SOL tool-calling request in about five minutes.

## 1. Requirements

| Requirement | Version |
| --- | --- |
| Python | 3.10 or newer |
| OpenAI API key | with access to the GPT-6 family |
| An MCP server | optional — only needed for tool calling |

## 2. Install

=== "Library only"

    ```bash
    pip install mcp-for-copilot
    ```

=== "With the HTTP server"

    ```bash
    pip install "mcp-for-copilot[server]"
    ```

=== "Everything"

    ```bash
    pip install "mcp-for-copilot[all]"
    ```

## 3. Configure

The gateway reads configuration from environment variables, and falls back to a `.env` file in the
current working directory. Real environment variables always win over `.env` values.

```bash
cp .env.example .env
```

Then set at minimum:

```dotenv
LLM_API_KEY=sk-your-key-here
```

See [Configuration](configuration.md) for the full list.

## 4. First request

### As a library

```python
import asyncio
from mcp_for_copilot import Gateway

async def main() -> None:
    async with Gateway(connect_mcp=False) as gateway:
        result = await gateway.chat([{"role": "user", "content": "What is the Model Context Protocol?"}])
        print(result.content)

asyncio.run(main())
```

### As a server

```bash
mcp-for-copilot serve --host 0.0.0.0 --port 8080
```

```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "gpt-6-sol",
        "messages": [{"role": "user", "content": "What is MCP?"}]
      }' | jq -r '.choices[0].message.content'
```

## 5. Add tools

Point the gateway at any MCP server. The command below assumes a stdio server that speaks MCP:

```python
import asyncio
from mcp_for_copilot import Gateway, allow_all_approver

async def main() -> None:
    async with Gateway(approver=allow_all_approver()) as gateway:
        result = await gateway.chat([{"role": "user", "content": "What files are in this directory?"}])
        print(result.content)

asyncio.run(main())
```

The MCP endpoint and allowlist come from `MCP_SERVER_URL`, `MCP_TRANSPORT`, and `MCP_ALLOWED_TOOLS`.

!!! warning "Approval is fail-closed"
    If you do not pass an `approver`, every risky tool call is **denied**. That is intentional.
    Use `allow_all_approver()` only in trusted, non-interactive environments.

## 6. Verify your setup

```bash
mcp-for-copilot status
```

```json
{
  "llm": {"provider": "openai", "api_key_configured": true, "default_model": "gpt-6-sol"},
  "mcp": {"transport": "stdio", "url_configured": false, "allowed_tools": ["read_file", "..."]}
}
```
`status` never prints credential values — only booleans.

## Next steps

- [Architecture](architecture.md) — how the pieces fit together
- [Library Usage](library.md) — custom approvers, tool routing, streaming
- [Server Usage](server.md) — Docker, compose, reverse proxies
- [Security](security.md) — threat model and hardening checklist
