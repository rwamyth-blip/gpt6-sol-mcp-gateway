"""Unit tests for :class:`MCPClientManager` — multi-server wiring.

These are the regression cover for the defects found while the manager was
being wired in:

* ``list_all_tools`` called ``log_warning`` without importing it, so *any*
  server that failed to list tools raised ``NameError`` instead of being skipped.
* ``_tool_to_client`` is only populated by :meth:`list_all_tools`, so the
  orchestrator must call it before a tool call can be routed.
* ``from_settings`` has to keep honouring the deprecated single-server fields.
"""

from __future__ import annotations

import logging

import pytest

from mcp_for_copilot.config import Settings
from mcp_for_copilot.mcp_client import (
    MCPClient,
    MCPClientError,
    MCPClientManager,
    MCPTool,
)

from ..conftest import FakeMCPClient

MULTI_SERVERS = (
    '[{"name":"ollama","transport":"stdio","command":"python -m ollama_mcp"},'
    '{"name":"codex","transport":"stdio","command":"python -m codex_mcp"}]'
)


class _BrokenClient:
    """A client whose server refuses to list tools."""

    async def list_tools(self) -> list[MCPTool]:
        raise MCPClientError("server is down")

    async def call_tool(self, name: str, arguments: dict | None = None) -> object:
        raise MCPClientError("server is down")

    async def close(self) -> None:
        return None


class TestFromSettings:
    def test_builds_one_client_per_server(self) -> None:
        manager = MCPClientManager.from_settings(
            Settings(mcp_servers=MULTI_SERVERS, mcp_server_url="")
        )

        ollama = manager.get_client("ollama")
        codex = manager.get_client("codex")
        assert isinstance(ollama, MCPClient)
        assert isinstance(codex, MCPClient)
        assert ollama.transport == "stdio"
        assert ollama.url == "python -m ollama_mcp"
        assert codex.url == "python -m codex_mcp"

    def test_deprecated_single_server_still_works(self) -> None:
        settings = Settings(
            mcp_servers="", mcp_server_url="python -m legacy", mcp_transport="stdio"
        )
        manager = MCPClientManager.from_settings(settings)

        client = manager.get_client()
        assert isinstance(client, MCPClient)
        assert client.url == "python -m legacy"

    def test_mcp_servers_wins_over_deprecated_fields(self) -> None:
        settings = Settings(mcp_servers=MULTI_SERVERS, mcp_server_url="python -m legacy")
        manager = MCPClientManager.from_settings(settings)

        assert manager.get_client("ollama") is not None
        assert manager.get_client("default") is None

    def test_no_server_configured_yields_no_clients(self) -> None:
        manager = MCPClientManager.from_settings(Settings(mcp_servers="", mcp_server_url=""))
        assert manager.get_client() is None


class TestRouting:
    async def test_tool_to_client_map_is_built_from_list_all_tools(
        self, fake_tools: list[MCPTool]
    ) -> None:
        ollama = FakeMCPClient(tools=fake_tools)
        codex = FakeMCPClient(tools=[MCPTool(name="codex_run", input_schema={"type": "object"})])
        manager = MCPClientManager({"ollama": ollama, "codex": codex})

        listed = await manager.list_all_tools()
        assert [t.name for t in listed["ollama"]] == ["read_file", "write_file"]

        read_file = manager.get_client_for_tool("read_file")
        assert read_file is not None
        assert read_file[0] == "ollama"
        assert read_file[1] is ollama

        codex_run = manager.get_client_for_tool("codex_run")
        assert codex_run is not None
        assert codex_run[0] == "codex"

    async def test_unknown_tool_has_no_client(self) -> None:
        manager = MCPClientManager({"ollama": FakeMCPClient()})
        await manager.list_all_tools()
        assert manager.get_client_for_tool("nope") is None

    async def test_broken_client_is_skipped_and_logged(
        self, fake_tools: list[MCPTool], caplog: pytest.LogCaptureFixture
    ) -> None:
        # Regression: this used to raise NameError (log_warning not imported).
        manager = MCPClientManager(
            {"good": FakeMCPClient(tools=fake_tools), "broken": _BrokenClient()}
        )
        with caplog.at_level(logging.WARNING, logger="mcp_for_copilot"):
            listed = await manager.list_all_tools()

        assert [t.name for t in listed["good"]] == ["read_file", "write_file"]
        assert listed["broken"] == []
        assert "broken" in caplog.text
        assert manager.get_client_for_tool("read_file")[0] == "good"

    async def test_call_tool_routes_to_the_named_client(self) -> None:
        ollama = FakeMCPClient(tools=[MCPTool(name="ollama_status")])
        codex = FakeMCPClient(tools=[MCPTool(name="codex_run")])
        manager = MCPClientManager({"ollama": ollama, "codex": codex})
        await manager.list_all_tools()

        result = await manager.call_tool("codex", "codex_run", {"project_id": "p"})
        assert result.content == "result of codex_run"
        assert codex.calls == [("codex_run", {"project_id": "p"})]
        assert ollama.calls == []

    async def test_call_tool_rejects_unknown_server(self) -> None:
        manager = MCPClientManager()
        with pytest.raises(MCPClientError, match="No MCP client named"):
            await manager.call_tool("ghost", "read_file")

    async def test_close_closes_every_client(self, fake_tools: list[MCPTool]) -> None:
        first = FakeMCPClient(tools=fake_tools)
        second = FakeMCPClient(tools=fake_tools)
        manager = MCPClientManager({"a": first, "b": second})
        await first.connect()
        await second.connect()

        await manager.close()
        assert first.connected is False
        assert second.connected is False
