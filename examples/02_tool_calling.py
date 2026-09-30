"""Example 2 — tool calling against a real MCP server over stdio.

This example spawns the gateway's own bundled MCP server as a subprocess, so it
needs no external MCP server to run.

Run:
    export LLM_API_KEY=sk-...
    python examples/02_tool_calling.py
"""

from __future__ import annotations

import asyncio
import os
import sys

from mcp_for_copilot import Gateway, allow_all_approver


async def main() -> int:
    # Point the gateway at its own MCP server, spawned over stdio.
    os.environ["MCP_TRANSPORT"] = "stdio"
    os.environ["MCP_SERVER_URL"] = f'"{sys.executable}" -m mcp_for_copilot.gateway.server'
    os.environ["MCP_ALLOWED_TOOLS"] = "gpt6_models,gpt6_status,gpt6_tools"

    from mcp_for_copilot import reset_settings_cache

    reset_settings_cache()

    async with Gateway(approver=allow_all_approver()) as gateway:
        print("advertised tools:", [t["function"]["name"] for t in await gateway.list_tools()])

        result = await gateway.chat(
            [{"role": "user", "content": "Use your tools to tell me which models are available."}]
        )

        if result.error:
            print(f"provider error: {result.error}", file=sys.stderr)
            return 1

        print("\nanswer:", result.content)
        print(f"rounds={result.rounds} used_tools={result.used_tools}")

        for call in result.invocations:
            state = "executed" if call.executed else "blocked"
            print(f"  [{state}] {call.tool}({call.arguments}) -> {call.result[:120]}")
            if call.reason:
                print(f"           reason: {call.reason}")

        return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
