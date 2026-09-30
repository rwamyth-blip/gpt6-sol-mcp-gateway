"""Smoke-test the ollama_mcp stdio server through the gateway's MCPClient.

Run from the gateway package root so .env is picked up:
    venv\\Scripts\\python.exe scripts\\_smoke_mcp_ollama.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MCP_PY = ROOT.parents[1] / "mcp" / "ollama_mcp.py"


async def main() -> int:
    from mcp_for_copilot.config import get_settings
    from mcp_for_copilot.mcp_client import MCPClient

    s = get_settings()
    print("=== settings.describe() ===")
    print(json.dumps(s.describe(), indent=2, default=str))
    print("MCP_SERVER_URL:", s.mcp_server_url)
    print("MCP_ALLOWED_TOOLS_RAW:", s.mcp_allowed_tools_raw)
    print("mcp script exists:", MCP_PY.exists(), MCP_PY)

    cmd = s.mcp_server_url or f"{sys.executable} {MCP_PY}"
    # forward the env the child needs
    env = dict(os.environ)
    env.setdefault("OLLAMA_HOST", os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))
    env.setdefault("OLLAMA_ALLOWED_MODELS", os.environ.get("OLLAMA_ALLOWED_MODELS", ""))
    env.setdefault("OLLAMA_ALLOW_PULL", os.environ.get("OLLAMA_ALLOW_PULL", "false"))

    client = MCPClient(url=cmd, transport="stdio", timeout=300.0, env=env)
    failures: list[str] = []
    try:
        await client.connect()
        print("\n=== connected ===")

        tools = await client.list_tools()
        print(f"=== tools/list -> {len(tools)} tools ===")
        for t in tools:
            print(f"  - {t.name}: {t.description[:70]}")

        calls: list[tuple[str, dict]] = [
            ("ollama_status", {}),
            ("ollama_models", {}),
            ("ollama_ps", {}),
            ("ollama_chat", {"model": "qwen2.5vl:7b", "prompt": "Reply with exactly: OK", "max_tokens": 16}),
            ("ollama_chat", {"model": "deepseek-r1:7b", "prompt": "Reply with exactly: OK", "max_tokens": 16}),
        ]
        for name, args in calls:
            print(f"\n=== tools/call {name} {args} ===")
            res = await client.call_tool(name, args)
            body = res.content
            print(f"is_error={res.is_error}")
            print(body[:700] if isinstance(body, str) else body)
            if res.is_error:
                failures.append(f"{name} -> is_error")
            if isinstance(body, str) and '"ok": false' in body.replace("'", '"'):
                failures.append(f"{name} -> ok:false")

        # Every allowed model must resolve through the allowlist AND be
        # classified with the correct local/cloud kind.
        show_targets = [
            ("deepseek-r1:14b", "local"),
            ("qwen2.5vl:7b", "local"),
            ("qwen2.5-coder:7b", "local"),
            ("gemma4:31b-cloud", "cloud"),
            ("gpt-oss:20b-cloud", "cloud"),
            ("gpt-oss:120b-cloud", "cloud"),
            ("nemotron-3-super:cloud", "cloud"),
            ("nemotron-3-ultra:cloud", "cloud"),
            ("nemotron-3-nano:30b-cloud", "cloud"),
        ]
        for model, expected_kind in show_targets:
            print(f"\n=== tools/call ollama_show {{'model': '{model}'}} (expect kind={expected_kind}) ===")
            res = await client.call_tool("ollama_show", {"model": model})
            body = res.content or ""
            print(f"is_error={res.is_error}")
            print(body[:400])
            if res.is_error or '"ok": false' in body:
                failures.append(f"ollama_show {model} -> error")
                continue
            try:
                payload = json.loads(body)
            except json.JSONDecodeError:
                failures.append(f"ollama_show {model} -> non-JSON body")
                continue
            if payload.get("kind") != expected_kind:
                failures.append(
                    f"ollama_show {model} -> kind={payload.get('kind')!r} expected {expected_kind!r}"
                )
            if expected_kind == "cloud" and not payload.get("remote_host"):
                failures.append(f"ollama_show {model} -> cloud but remote_host missing")

        # Negative control: a model outside the allowlist must be refused.
        # This asserts the security gate works, so it is expected to fail.
        print("\n=== negative control: ollama_chat llama3.2:1b (expect refusal) ===")
        denied = await client.call_tool("ollama_chat", {"model": "llama3.2:1b", "prompt": "hi"})
        print((denied.content or "")[:300])
        if "model not allowed" not in (denied.content or ""):
            failures.append("allowlist gate did NOT block a non-allowed model")
    finally:
        await client.close()

    print("\n=== RESULT ===")
    if failures:
        print("FAILURES:")
        for f in failures:
            print("  -", f)
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
