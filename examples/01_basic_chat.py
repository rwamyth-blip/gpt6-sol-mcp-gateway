"""Example 1 — plain chat, no MCP server required.

Run:
    export LLM_API_KEY=sk-...
    python examples/01_basic_chat.py
"""

from __future__ import annotations

import asyncio
import sys

from mcp_for_copilot import Gateway


async def main() -> int:
    async with Gateway(connect_mcp=False) as gateway:
        result = await gateway.chat(
            [{"role": "user", "content": "Explain the Model Context Protocol in two sentences."}],
            system="You are a concise technical writer.",
        )

        if result.error:
            print(f"provider error: {result.error}", file=sys.stderr)
            return 1

        print(result.content)
        print(f"\nmodel={result.model} rounds={result.rounds} used_tools={result.used_tools}")
        return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
