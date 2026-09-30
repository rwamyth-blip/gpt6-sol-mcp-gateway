"""Shared pytest fixtures.

No test in this suite touches the network. Provider calls are served by an
``httpx.MockTransport``; MCP calls are served by :class:`FakeMCPClient`.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from mcp_for_copilot.config import Settings, reset_settings_cache
from mcp_for_copilot.mcp_client import MCPCallResult, MCPTool


@pytest.fixture(autouse=True)
def _clean_settings_cache() -> Any:
    """Keep the settings singleton from leaking between tests."""
    reset_settings_cache()
    yield
    reset_settings_cache()


@pytest.fixture(autouse=True)
def _isolate_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Keep the developer's ``.env`` and shell exports out of every test.

    ``Settings`` reads ``./.env`` through pydantic-settings, and
    ``get_settings()`` additionally loads ``Path.cwd()/.env``. Running pytest
    from the project root therefore used to pick up the local Ollama allowlist
    and fail tests that expect the read-only defaults. Running from an empty
    temporary directory makes both lookups miss, and deleting any exported
    ``LLM_*`` / ``MCP_*`` / ``GATEWAY_*`` variables stops a stray shell export
    from winning instead.
    """
    for name in [k for k in os.environ if k.upper().startswith(("LLM_", "MCP_", "GATEWAY_"))]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    yield


@pytest.fixture
def settings() -> Settings:
    """Settings with a fake key and no MCP server configured."""
    return Settings(
        llm_provider="openai",
        llm_model_id="gpt-6-sol",
        llm_api_key="sk-test-not-a-real-key",
        llm_base_url="https://api.example.test/v1",
        llm_timeout_seconds=5,
        mcp_server_url="",
        mcp_transport="stdio",
        mcp_require_approval=True,
        mcp_max_tool_rounds=3,
        gateway_api_key="",
    )


def make_completion(
    *,
    content: str = "Hello!",
    tool_calls: list[dict[str, Any]] | None = None,
    model: str = "gpt-6-sol",
    finish_reason: str = "stop",
) -> dict[str, Any]:
    """Build an OpenAI-shaped chat completion payload."""
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
        message["content"] = content or None
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1_700_000_000,
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def make_tool_call(
    name: str, arguments: dict[str, Any] | str, call_id: str = "call_1"
) -> dict[str, Any]:
    """Build one entry of ``message.tool_calls``."""
    payload = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": payload},
    }


def mock_transport(
    responses: list[dict[str, Any]] | dict[str, Any],
    *,
    status_code: int = 200,
    capture: list[dict[str, Any]] | None = None,
) -> httpx.MockTransport:
    """Return an ``httpx.MockTransport`` that replays *responses* in order.

    The last response repeats once the queue is exhausted, so a multi-round
    test does not need to enumerate every call.
    """
    queue = responses if isinstance(responses, list) else [responses]
    state = {"index": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if capture is not None:
            capture.append(json.loads(request.content.decode("utf-8")))
        index = min(state["index"], len(queue) - 1)
        state["index"] += 1
        return httpx.Response(status_code, json=queue[index])

    return httpx.MockTransport(handler)


class FakeMCPClient:
    """In-memory stand-in for :class:`mcp_for_copilot.mcp_client.MCPClient`."""

    def __init__(
        self,
        tools: list[MCPTool] | None = None,
        results: dict[str, MCPCallResult] | None = None,
    ) -> None:
        self._tools = tools or []
        self._results = results or {}
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.connected = False

    async def connect(self) -> None:
        self.connected = True

    async def close(self) -> None:
        self.connected = False

    async def list_tools(self) -> list[MCPTool]:
        return list(self._tools)

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> MCPCallResult:
        self.calls.append((name, arguments or {}))
        if name in self._results:
            return self._results[name]
        return MCPCallResult(tool=name, content=f"result of {name}")

    @property
    def server_info(self) -> dict[str, Any]:
        return {"serverInfo": {"name": "fake", "version": "0"}}

    @property
    def initialised(self) -> bool:
        return self.connected


@pytest.fixture
def fake_tools() -> list[MCPTool]:
    """A read-only tool and a risky tool, both with schemas."""
    return [
        MCPTool(
            name="read_file",
            description="Read a file",
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        ),
        MCPTool(
            name="write_file",
            description="Write a file",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        ),
    ]


@pytest.fixture
def fake_client(fake_tools: list[MCPTool]) -> FakeMCPClient:
    return FakeMCPClient(tools=fake_tools)
