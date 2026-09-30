"""VihokAI mcp5-Zai stdio MCP server — standalone bridge to the Z.ai (GLM) gateway.

    MCP client  <--stdio-->  THIS SCRIPT  -->  http://127.0.0.1:7424/v1/*

Same contract as the root mcp_bridge.py but pointed at the mcp5-Zai gateway so
this MCP server works on its own: it never references ports 7420/7423 and can
run even when every other MCP stack is down.

Tools
-----
    zai_status    - gateway + LLM + MCP status
    zai_models    - list GLM models this gateway can serve
    zai_chat      - chat completion (non-streaming); pass "model" to pick tier
    zai_compare   - ask the same question to several GLM tiers

Environment
-----------
    ZAI_GATEWAY_URL   base URL, default http://127.0.0.1:7424
    ZAI_API_KEY       bearer token for the gateway (GATEWAY_API_KEY in .env)
    ZAI_TIMEOUT       per-request timeout in seconds, default 300
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx

DEFAULT_BASE_URL = "http://127.0.0.1:7424"
DEFAULT_TIMEOUT = 300.0


def _base_url() -> str:
    return os.environ.get("ZAI_GATEWAY_URL", DEFAULT_BASE_URL).rstrip("/")


def _api_key() -> str:
    key = os.environ.get("ZAI_API_KEY", "").strip()
    if key and key != "${env:GATEWAY_API_KEY}":
        return key
    # Some MCP hosts do not expand ${env:VAR}; fall back to the repo .env.
    env_file = Path(__file__).resolve().parent / ".env"
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if line.startswith("GATEWAY_API_KEY="):
                value = line.split("=", 1)[1].strip().strip("'\"")
                if value:
                    return value
    return key


def _timeout() -> float:
    raw = os.environ.get("ZAI_TIMEOUT", "").strip()
    try:
        return float(raw) if raw else DEFAULT_TIMEOUT
    except ValueError:
        return DEFAULT_TIMEOUT


def _headers() -> dict[str, str]:
    key = _api_key()
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


async def _post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    url = f"{_base_url()}{path}"
    async with httpx.AsyncClient(timeout=_timeout()) as client:
        response = await client.post(url, json=payload, headers=_headers())
    if response.status_code != 200:
        return {"error": f"HTTP {response.status_code}", "detail": response.text[:300]}
    return response.json()


async def _get(path: str) -> dict[str, Any]:
    url = f"{_base_url()}{path}"
    async with httpx.AsyncClient(timeout=_timeout()) as client:
        response = await client.get(url, headers=_headers())
    if response.status_code != 200:
        return {"error": f"HTTP {response.status_code}", "detail": response.text[:300]}
    return response.json()


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

async def tool_zai_status(_args: dict[str, Any]) -> str:
    data = await _get("/v1/status")
    return json.dumps(data, ensure_ascii=False)


async def tool_zai_models(_args: dict[str, Any]) -> str:
    data = await _get("/v1/models")
    return json.dumps(data, ensure_ascii=False)


async def tool_zai_chat(args: dict[str, Any]) -> str:
    messages = args.get("messages")
    if not isinstance(messages, list) or not messages:
        question = str(args.get("question") or "").strip()
        if not question:
            return "error: provide 'messages' (array) or 'question' (string)"
        messages = [{"role": "user", "content": question}]
    payload: dict[str, Any] = {
        "messages": messages,
        "model": str(args.get("model") or "glm-4.5-flash"),
    }
    if args.get("system"):
        payload["system"] = str(args["system"])
    data = await _post("/v1/chat/completions", payload)
    if "error" in data:
        return json.dumps(data, ensure_ascii=False)
    try:
        content = data["choices"][0]["message"]["content"]
        used = data.get("x_gateway", {}).get("used_tools", [])
        return json.dumps({"answer": content, "model": data.get("model"), "used_tools": used}, ensure_ascii=False)
    except (KeyError, IndexError, TypeError):
        return json.dumps(data, ensure_ascii=False)


async def tool_zai_compare(args: dict[str, Any]) -> str:
    question = str(args.get("question") or "").strip()
    if not question:
        return "error: 'question' is required"
    models = args.get("models") or ["glm-4.5-flash", "glm-5", "glm-5.1"]
    results: dict[str, Any] = {}
    for model in models:
        data = await _post(
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": question}], "model": str(model)},
        )
        if "error" in data:
            results[str(model)] = data
        else:
            try:
                results[str(model)] = data["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError):
                results[str(model)] = data
    return json.dumps(results, ensure_ascii=False)


TOOLS: list[dict[str, Any]] = [
    {
        "name": "zai_status",
        "description": "Show mcp5-Zai standalone gateway status (LLM, model, port).",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "zai_models",
        "description": "List GLM models served by the mcp5-Zai gateway (glm-4.5-flash, glm-5, glm-5.1).",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "zai_chat",
        "description": (
            "Chat with the Z.ai GLM team. 'model' picks the tier: glm-4.5-flash "
            "(easy/cheap), glm-5 (mid), glm-5.1 (hard). Provide 'messages' or 'question'."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "messages": {"type": "array", "items": {"type": "object"}},
                "model": {
                    "type": "string",
                    "enum": ["glm-4.5-flash", "glm-5.3-flash", "glm-5", "glm-5.1"],
                },
                "system": {"type": "string"},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "zai_compare",
        "description": "Ask one question to several GLM tiers and compare answers.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "models": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["question"],
            "additionalProperties": False,
        },
    },
]


async def _dispatch(name: str, args: dict[str, Any]) -> str:
    handlers = {
        "zai_status": tool_zai_status,
        "zai_models": tool_zai_models,
        "zai_chat": tool_zai_chat,
        "zai_compare": tool_zai_compare,
    }
    handler = handlers.get(name)
    if handler is None:
        return f"error: unknown tool {name!r}"
    try:
        return await handler(args or {})
    except Exception as exc:  # noqa: BLE001 - stdio server must not crash
        return f"error: {type(exc).__name__}: {exc}"


async def _handle(request: dict[str, Any]) -> dict[str, Any] | None:
    method = request.get("method")
    request_id = request.get("id")
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "mcp5-zai", "version": "1.0.0"},
            },
        }
    if method == "notifications/initialized":
        return None
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": request_id, "result": {"tools": TOOLS}}
    if method == "tools/call":
        params = request.get("params") or {}
        text = await _dispatch(str(params.get("name")), dict(params.get("arguments") or {}))
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {"content": [{"type": "text", "text": text}]},
        }
    if request_id is None:
        return None
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": -32601, "message": f"method not found: {method}"},
    }


async def _serve() -> None:
    loop = asyncio.get_running_loop()
    while True:
        line = await loop.run_in_executor(None, sys.stdin.readline)
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            continue
        response = await _handle(request)
        if response is not None:
            sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    asyncio.run(_serve())
