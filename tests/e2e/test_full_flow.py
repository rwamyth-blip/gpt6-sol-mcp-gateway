"""End-to-end tests — the full model -> tool -> model loop.

These tests exercise the real :class:`Gateway` facade, the real orchestrator,
the real router, and the real approval layer. Only two things are faked:

* the LLM provider's HTTP transport (so no network call is made), and
* the MCP client (so no subprocess is spawned).

One test goes further and spawns a real MCP server subprocess over stdio, to
prove the transport works end to end.
"""

from __future__ import annotations

import json
import sys

import pytest

from mcp_for_copilot.approval import allow_all_approver, static_approver
from mcp_for_copilot.config import Settings
from mcp_for_copilot.gateway.facade import Gateway
from mcp_for_copilot.mcp_client import MCPClient, MCPTool
from mcp_for_copilot.provider import LLMProvider

from ..conftest import FakeMCPClient, make_completion, make_tool_call, mock_transport


def _settings(**overrides) -> Settings:
    base = {
        "llm_api_key": "sk-test",
        "llm_model_id": "gpt-6-sol",
        "llm_base_url": "https://api.example.test/v1",
        "mcp_server_url": "",
        "mcp_require_approval": True,
        "mcp_max_tool_rounds": 3,
        # The default allowlist is read-only; these tests also exercise a
        # mutating tool so the approval gate has something to decide on.
        "mcp_allowed_tools_raw": "read_file,write_file",
    }
    base.update(overrides)
    return Settings(**base)


def _gateway(responses, *, fake_client=None, approver=None, settings=None, capture=None):
    resolved = settings or _settings()
    provider = LLMProvider(
        api_key=resolved.llm_api_key,
        base_url=resolved.llm_base_url,
        model_id=resolved.llm_model_id,
        transport=mock_transport(responses, capture=capture),
    )
    gateway = Gateway(settings=resolved, provider=provider, approver=approver, connect_mcp=False)
    gateway.client = fake_client
    gateway.orchestrator.client = fake_client
    gateway._connected = True
    return gateway


class TestHappyPath:
    async def test_read_then_answer(self, fake_client: FakeMCPClient) -> None:
        gateway = _gateway(
            [
                make_completion(
                    content="",
                    tool_calls=[make_tool_call("read_file", {"path": "notes.txt"})],
                    finish_reason="tool_calls",
                ),
                make_completion(content="The notes say: buy milk."),
            ],
            fake_client=fake_client,
        )
        result = await gateway.chat([{"role": "user", "content": "What do my notes say?"}])

        assert result.content == "The notes say: buy milk."
        assert result.used_tools == ["read_file"]
        assert result.rounds == 2
        assert fake_client.calls == [("read_file", {"path": "notes.txt"})]

    async def test_multiple_tools_in_one_round(self, fake_client: FakeMCPClient) -> None:
        gateway = _gateway(
            [
                make_completion(
                    content="",
                    tool_calls=[
                        make_tool_call("read_file", {"path": "a.txt"}, "call_1"),
                        make_tool_call("read_file", {"path": "b.txt"}, "call_2"),
                    ],
                    finish_reason="tool_calls",
                ),
                make_completion(content="Both files read."),
            ],
            fake_client=fake_client,
        )
        result = await gateway.chat([{"role": "user", "content": "read both"}])
        assert len(result.used_tools) == 2
        assert len(fake_client.calls) == 2

    async def test_no_tools_needed(self, fake_client: FakeMCPClient) -> None:
        gateway = _gateway(make_completion(content="2 + 2 is 4."), fake_client=fake_client)
        result = await gateway.chat([{"role": "user", "content": "what is 2+2"}])
        assert result.content == "2 + 2 is 4."
        assert result.used_tools == []
        assert fake_client.calls == []


class TestSafetyEndToEnd:
    async def test_risky_tool_denied_without_approver(self, fake_client) -> None:
        gateway = _gateway(
            [
                make_completion(
                    content="",
                    tool_calls=[make_tool_call("write_file", {"path": "x", "content": "y"})],
                    finish_reason="tool_calls",
                ),
                make_completion(content="I was not allowed to write."),
            ],
            fake_client=fake_client,
        )
        result = await gateway.chat([{"role": "user", "content": "write x"}])

        assert fake_client.calls == []
        assert result.used_tools == []
        assert result.invocations[0]["approved"] is False
        assert result.content == "I was not allowed to write."

    async def test_risky_tool_allowed_with_approver(self, fake_client) -> None:
        gateway = _gateway(
            [
                make_completion(
                    content="",
                    tool_calls=[make_tool_call("write_file", {"path": "x", "content": "y"})],
                    finish_reason="tool_calls",
                ),
                make_completion(content="Written."),
            ],
            fake_client=fake_client,
            approver=allow_all_approver(),
        )
        result = await gateway.chat([{"role": "user", "content": "write x"}])
        assert result.used_tools == ["write_file"]
        assert result.invocations[0]["approved"] is True

    async def test_selective_approver_blocks_the_rest(self, fake_client) -> None:
        gateway = _gateway(
            [
                make_completion(
                    content="",
                    tool_calls=[make_tool_call("write_file", {"path": "x", "content": "y"})],
                    finish_reason="tool_calls",
                ),
                make_completion(content="Blocked."),
            ],
            fake_client=fake_client,
            approver=static_approver(allow=["read_file"]),
        )
        result = await gateway.chat([{"role": "user", "content": "write x"}])
        assert fake_client.calls == []
        assert result.invocations[0]["approved"] is False

    async def test_allowlist_blocks_unlisted_tool(self, fake_client: FakeMCPClient) -> None:
        gateway = _gateway(
            [
                make_completion(
                    content="",
                    tool_calls=[make_tool_call("delete_file", {"path": "x"})],
                    finish_reason="tool_calls",
                ),
                make_completion(content="Not allowed."),
            ],
            fake_client=fake_client,
        )
        result = await gateway.chat([{"role": "user", "content": "delete x"}])
        assert fake_client.calls == []
        assert result.invocations[0]["allowed"] is False

    async def test_approval_disabled_lets_risky_tools_through(
        self, fake_client: FakeMCPClient
    ) -> None:
        gateway = _gateway(
            [
                make_completion(
                    content="",
                    tool_calls=[make_tool_call("write_file", {"path": "x", "content": "y"})],
                    finish_reason="tool_calls",
                ),
                make_completion(content="Written."),
            ],
            fake_client=fake_client,
            settings=_settings(mcp_require_approval=False),
        )
        result = await gateway.chat([{"role": "user", "content": "write x"}])
        assert result.used_tools == ["write_file"]
        assert result.invocations[0]["approved"] is None


class TestRoundCap:
    async def test_loop_stops_at_the_cap(self, fake_client: FakeMCPClient) -> None:
        gateway = _gateway(
            make_completion(
                content="",
                tool_calls=[make_tool_call("read_file", {"path": "a"})],
                finish_reason="tool_calls",
            ),
            fake_client=fake_client,
            settings=_settings(mcp_max_tool_rounds=2),
        )
        result = await gateway.chat([{"role": "user", "content": "loop forever"}])
        assert len(fake_client.calls) == 2
        assert "Stopped after 2 tool rounds" in result.content


class TestGatewayStatus:
    async def test_status_reports_connection_state(self, fake_client) -> None:
        gateway = _gateway(make_completion(), fake_client=fake_client)
        status = gateway.status()
        assert status["mcp"]["connected"] is True
        assert status["llm"]["model_id"] == "gpt-6-sol"
        assert "sk-test" not in json.dumps(status)

    async def test_list_tools_returns_openai_schemas(self, fake_client) -> None:
        gateway = _gateway(make_completion(), fake_client=fake_client)
        tools = await gateway.list_tools()
        assert [t["function"]["name"] for t in tools] == ["read_file", "write_file"]


class TestRealStdioTransport:
    """Spawn a real MCP server subprocess and drive the full loop through it."""

    async def test_full_loop_over_stdio(self) -> None:
        server_script = (
            "import json,sys\n"
            "TOOLS=[{'name':'read_file','description':'Read a file',"
            "'inputSchema':{'type':'object','properties':{'path':{'type':'string'}},"
            "'required':['path']}}]\n"
            "for line in sys.stdin:\n"
            "    line=line.strip()\n"
            "    if not line: continue\n"
            "    msg=json.loads(line)\n"
            "    if 'id' not in msg: continue\n"
            "    m=msg['method']\n"
            "    if m=='initialize':\n"
            "        r={'serverInfo':{'name':'e2e','version':'1'}}\n"
            "    elif m=='tools/list':\n"
            "        r={'tools':TOOLS}\n"
            "    elif m=='tools/call':\n"
            "        r={'content':[{'type':'text','text':'contents of '+"
            "msg['params']['arguments']['path']}]}\n"
            "    else:\n"
            "        r={}\n"
            "    sys.stdout.write(json.dumps({'jsonrpc':'2.0','id':msg['id'],'result':r})+'\\n')\n"
            "    sys.stdout.flush()\n"
        )
        command = f'"{sys.executable}" -c "{server_script}"'

        settings = _settings(mcp_server_url=command, mcp_transport="stdio")
        provider = LLMProvider(
            api_key="sk-test",
            base_url=settings.llm_base_url,
            model_id="gpt-6-sol",
            transport=mock_transport(
                [
                    make_completion(
                        content="",
                        tool_calls=[make_tool_call("read_file", {"path": "notes.txt"})],
                        finish_reason="tool_calls",
                    ),
                    make_completion(content="Your notes are empty."),
                ]
            ),
        )

        gateway = Gateway(settings=settings, provider=provider, connect_mcp=True)
        try:
            await gateway.start()
            assert gateway.client is not None, "MCP client failed to connect"
            result = await gateway.chat([{"role": "user", "content": "read my notes"}])
        finally:
            await gateway.stop()

        assert result.content == "Your notes are empty."
        assert result.used_tools == ["read_file"]


class TestMCPClientDirect:
    async def test_list_and_call_over_stdio(self) -> None:
        server_script = (
            "import json,sys\n"
            "for line in sys.stdin:\n"
            "    line=line.strip()\n"
            "    if not line: continue\n"
            "    msg=json.loads(line)\n"
            "    if 'id' not in msg: continue\n"
            "    m=msg['method']\n"
            "    if m=='initialize':\n"
            "        r={'serverInfo':{'name':'direct','version':'1'}}\n"
            "    elif m=='tools/list':\n"
            "        r={'tools':[{'name':'ping','inputSchema':{'type':'object'}}]}\n"
            "    elif m=='tools/call':\n"
            "        r={'content':[{'type':'text','text':'pong'}]}\n"
            "    else:\n"
            "        r={}\n"
            "    sys.stdout.write(json.dumps({'jsonrpc':'2.0','id':msg['id'],'result':r})+'\\n')\n"
            "    sys.stdout.flush()\n"
        )
        command = f'"{sys.executable}" -c "{server_script}"'
        async with MCPClient(url=command, transport="stdio", timeout=20) as client:
            tools = await client.list_tools()
            result = await client.call_tool("ping", {})
        assert [t.name for t in tools] == ["ping"]
        assert result.content == "pong"


class TestToolSchemaConversion:
    async def test_mcp_tools_become_openai_functions(self) -> None:
        client = FakeMCPClient(
            tools=[
                MCPTool(
                    name="read_file",
                    description="Read a file",
                    input_schema={
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                )
            ]
        )
        gateway = _gateway(make_completion(), fake_client=client)
        tools = await gateway.list_tools()
        assert tools[0]["type"] == "function"
        assert tools[0]["function"]["parameters"]["required"] == ["path"]


@pytest.mark.parametrize("model_alias", ["sol", "gpt-6", "gpt6", "gpt-6-sol"])
async def test_model_aliases_all_work(model_alias: str, fake_client) -> None:
    gateway = _gateway(make_completion(content="ok"), fake_client=fake_client)
    result = await gateway.chat([{"role": "user", "content": "hi"}], model_id=model_alias)
    assert result.content == "ok"
