"""Integration tests for ``Gateway`` wired to an :class:`MCPClientManager`.

``MCP_SERVERS`` was parsed by :class:`Settings` and the orchestrator already
branched on :class:`MCPClientManager`, but the facade never built one — the
multi-server configuration was silently ignored and the gateway fell back to
the bundled stdio server. These tests pin the wiring down:

* ``MCP_SERVERS`` set  -> a manager is built and every reachable server connects.
* ``MCP_SERVERS`` unset -> the deprecated single-server path is unchanged.
* a server that refuses to connect is dropped, not fatal.
* tools from every server are merged into one allowlist.
* ``status()`` reports the server names without leaking credentials.
"""

from __future__ import annotations

from typing import Any

import pytest

from mcp_for_copilot.config import Settings
from mcp_for_copilot.gateway.facade import Gateway
from mcp_for_copilot.mcp_client import MCPClientError, MCPClientManager, MCPTool
from mcp_for_copilot.provider import LLMProvider

from ..conftest import FakeMCPClient, make_completion, mock_transport

TWO_SERVERS = (
    '[{"name":"ollama","transport":"stdio","command":"python -m ollama_mcp"},'
    '{"name":"codex","transport":"stdio","command":"python -m codex_mcp"}]'
)


def _settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "llm_api_key": "sk-test",
        "llm_model_id": "gpt-6-sol",
        "llm_base_url": "https://api.example.test/v1",
        "mcp_server_url": "",
        "mcp_transport": "stdio",
    }
    base.update(overrides)
    return Settings(**base)


def _gateway(settings: Settings) -> Gateway:
    provider = LLMProvider(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        model_id=settings.llm_model_id,
        transport=mock_transport(make_completion()),
    )
    return Gateway(settings=settings, provider=provider, connect_mcp=False)


async def _connect(gateway: Gateway) -> None:
    """Run the connect step without the ``connect_mcp`` short-circuit.

    ``Gateway(connect_mcp=False)`` is how every other test builds a gateway,
    but it makes ``start()`` a no-op. These tests exercise the connection
    wiring itself, so they call the same helpers ``start()`` would.
    """
    if gateway.settings.mcp_servers_parsed:
        gateway.client = await gateway._connect_manager()
    else:
        gateway.client = await gateway._connect_single()
    gateway.orchestrator.client = gateway.client
    gateway._connected = True


class _RefusingClient:
    """A client whose server is unreachable."""

    def __init__(self) -> None:
        self.connected = False

    async def connect(self) -> None:
        raise MCPClientError("connection refused")

    async def close(self) -> None:
        return None

    async def list_tools(self) -> list[MCPTool]:
        raise MCPClientError("connection refused")

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        raise MCPClientError("connection refused")


class TestManagerSelection:
    async def test_mcp_servers_builds_a_manager(self, monkeypatch: pytest.MonkeyPatch) -> None:
        gateway = _gateway(_settings(mcp_servers=TWO_SERVERS))
        monkeypatch.setattr(
            MCPClientManager,
            "from_settings",
            classmethod(
                lambda cls, settings: cls({"ollama": FakeMCPClient(), "codex": FakeMCPClient()})
            ),
        )

        await _connect(gateway)

        assert isinstance(gateway.client, MCPClientManager)
        assert gateway.orchestrator.client is gateway.client
        assert sorted(gateway.client._clients) == ["codex", "ollama"]

    async def test_without_mcp_servers_the_single_path_is_used(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        gateway = _gateway(_settings(mcp_server_url="python -m legacy"))
        fake = FakeMCPClient()
        monkeypatch.setattr("mcp_for_copilot.gateway.facade.MCPClient", lambda **kwargs: fake)

        await _connect(gateway)

        assert gateway.client is fake
        assert not isinstance(gateway.client, MCPClientManager)
        assert fake.connected is True

    async def test_mcp_servers_wins_over_the_deprecated_url(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        gateway = _gateway(_settings(mcp_servers=TWO_SERVERS, mcp_server_url="python -m legacy"))
        monkeypatch.setattr(
            MCPClientManager,
            "from_settings",
            classmethod(lambda cls, settings: cls({"ollama": FakeMCPClient()})),
        )

        await _connect(gateway)

        assert isinstance(gateway.client, MCPClientManager)


class TestPartialFailure:
    async def test_unreachable_server_is_dropped_not_fatal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        gateway = _gateway(_settings(mcp_servers=TWO_SERVERS))
        monkeypatch.setattr(
            MCPClientManager,
            "from_settings",
            classmethod(
                lambda cls, settings: cls({"good": FakeMCPClient(), "broken": _RefusingClient()})
            ),
        )

        await _connect(gateway)

        assert isinstance(gateway.client, MCPClientManager)
        assert sorted(gateway.client._clients) == ["good"]

    async def test_all_servers_unreachable_yields_no_client(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        gateway = _gateway(_settings(mcp_servers=TWO_SERVERS))
        monkeypatch.setattr(
            MCPClientManager,
            "from_settings",
            classmethod(lambda cls, settings: cls({"broken": _RefusingClient()})),
        )

        await _connect(gateway)

        assert gateway.client is None
        assert gateway.orchestrator.client is None


class TestToolAggregation:
    async def test_tools_from_every_server_are_merged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        gateway = _gateway(_settings(mcp_servers=TWO_SERVERS))
        monkeypatch.setattr(
            MCPClientManager,
            "from_settings",
            classmethod(
                lambda cls, settings: cls(
                    {
                        "ollama": FakeMCPClient(tools=[MCPTool(name="read_file")]),
                        "codex": FakeMCPClient(tools=[MCPTool(name="list_directory")]),
                    }
                )
            ),
        )

        await _connect(gateway)
        tools = await gateway.list_tools()

        names = {t["function"]["name"] for t in tools}
        # The router merges the discovered tools with its read-only defaults,
        # so the result is a superset of what the two servers advertised.
        assert {"read_file", "list_directory"} <= names

    async def test_tool_routing_reaches_the_owning_server(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        gateway = _gateway(_settings(mcp_servers=TWO_SERVERS))
        ollama = FakeMCPClient(tools=[MCPTool(name="read_file")])
        codex = FakeMCPClient(tools=[MCPTool(name="list_directory")])
        monkeypatch.setattr(
            MCPClientManager,
            "from_settings",
            classmethod(lambda cls, settings: cls({"ollama": ollama, "codex": codex})),
        )

        await _connect(gateway)
        await gateway.list_tools()
        assert isinstance(gateway.client, MCPClientManager)
        await gateway.client.call_tool("codex", "list_directory", {"path": "."})

        assert codex.calls == [("list_directory", {"path": "."})]
        assert ollama.calls == []


class TestStatus:
    async def test_status_lists_server_names(self, monkeypatch: pytest.MonkeyPatch) -> None:
        gateway = _gateway(_settings(mcp_servers=TWO_SERVERS))
        monkeypatch.setattr(
            MCPClientManager,
            "from_settings",
            classmethod(
                lambda cls, settings: cls({"ollama": FakeMCPClient(), "codex": FakeMCPClient()})
            ),
        )

        await _connect(gateway)
        status = gateway.status()

        assert status["mcp"]["transport"] == "multi"
        assert status["mcp"]["servers"] == ["codex", "ollama"]
        assert status["mcp"]["connected"] is True

    async def test_status_never_leaks_auth_tokens(self, monkeypatch: pytest.MonkeyPatch) -> None:
        settings = _settings(
            mcp_servers=(
                '[{"name":"remote","transport":"http","url":"https://mcp.example.test",'
                '"auth_token":"super-secret-token"}]'
            )
        )
        gateway = _gateway(settings)
        monkeypatch.setattr(
            MCPClientManager,
            "from_settings",
            classmethod(lambda cls, settings: cls({"remote": FakeMCPClient()})),
        )

        await _connect(gateway)
        rendered = repr(gateway.status())

        assert "super-secret-token" not in rendered
        assert "remote" in rendered


class TestStop:
    async def test_stop_closes_every_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        gateway = _gateway(_settings(mcp_servers=TWO_SERVERS))
        first = FakeMCPClient()
        second = FakeMCPClient()
        monkeypatch.setattr(
            MCPClientManager,
            "from_settings",
            classmethod(lambda cls, settings: cls({"a": first, "b": second})),
        )

        await _connect(gateway)
        assert first.connected and second.connected

        await gateway.stop()

        assert first.connected is False
        assert second.connected is False
        assert gateway.client is None
