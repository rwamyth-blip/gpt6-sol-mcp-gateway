"""Unit tests for the MCP client — transports, framing, and error handling."""

from __future__ import annotations

import json

import httpx
import pytest

from mcp_for_copilot.mcp_client import (
    MCPClient,
    MCPClientError,
    MCPTool,
    _flatten_content,
    default_stdio_command,
)


class TestFlattenContent:
    def test_text_blocks_are_joined(self) -> None:
        blocks = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
        assert _flatten_content(blocks) == "a\nb"

    def test_plain_string_passes_through(self) -> None:
        assert _flatten_content("hello") == "hello"

    def test_none_becomes_empty(self) -> None:
        assert _flatten_content(None) == ""

    def test_single_dict_is_wrapped(self) -> None:
        assert _flatten_content({"type": "text", "text": "x"}) == "x"

    def test_image_block_is_placeholder(self) -> None:
        assert "image" in _flatten_content([{"type": "image", "mimeType": "image/png"}])

    def test_resource_block_uses_text(self) -> None:
        block = {"type": "resource", "resource": {"text": "payload"}}
        assert _flatten_content([block]) == "payload"

    def test_unknown_block_is_json_encoded(self) -> None:
        assert "weird" in _flatten_content([{"type": "weird", "v": 1}])


class TestMCPTool:
    def test_from_raw_camel_case(self) -> None:
        tool = MCPTool.from_raw(
            {"name": "read_file", "description": "d", "inputSchema": {"type": "object"}}
        )
        assert tool.name == "read_file"
        assert tool.input_schema == {"type": "object"}

    def test_from_raw_snake_case(self) -> None:
        tool = MCPTool.from_raw({"name": "x", "input_schema": {"type": "object"}})
        assert tool.input_schema == {"type": "object"}

    def test_from_raw_missing_fields(self) -> None:
        tool = MCPTool.from_raw({})
        assert tool.name == ""
        assert tool.input_schema == {}


class TestMCPClientInit:
    def test_rejects_unknown_transport(self) -> None:
        with pytest.raises(MCPClientError, match="Unsupported MCP transport"):
            MCPClient(url="x", transport="carrier-pigeon")

    def test_rejects_empty_url(self) -> None:
        with pytest.raises(MCPClientError, match="url/command is required"):
            MCPClient(url="", transport="stdio")

    def test_transport_is_normalised(self) -> None:
        assert MCPClient(url="x", transport="STDIO").transport == "stdio"

    def test_default_stdio_command_mentions_the_module(self) -> None:
        assert "mcp_for_copilot.gateway.server" in default_stdio_command()


def _sse_body(payload: dict) -> str:
    return f"event: message\ndata: {json.dumps(payload)}\n\n"


def _jsonrpc_reply(body: dict, result: dict | None = None) -> httpx.Response:
    """Reply to a JSON-RPC request, or 202 to a notification (no ``id``)."""
    if "id" not in body:
        return httpx.Response(202)
    return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": result or {}})


class TestHTTPTransport:
    async def test_initialize_and_list_tools(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content.decode())
            if body["method"] == "initialize":
                return _jsonrpc_reply(body, {"serverInfo": {"name": "test", "version": "1"}})
            if body["method"] == "tools/list":
                return _jsonrpc_reply(
                    body,
                    {
                        "tools": [
                            {
                                "name": "read_file",
                                "description": "d",
                                "inputSchema": {"type": "object"},
                            }
                        ]
                    },
                )
            return _jsonrpc_reply(body)

        client = MCPClient(
            url="https://mcp.example.test/mcp",
            transport="http",
            transport_impl=httpx.MockTransport(handler),
        )
        async with client:
            tools = await client.list_tools()
        assert [t.name for t in tools] == ["read_file"]
        assert client.server_info["serverInfo"]["name"] == "test"

    async def test_call_tool_flattens_content(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content.decode())
            if body["method"] == "tools/call":
                return _jsonrpc_reply(body, {"content": [{"type": "text", "text": "file body"}]})
            return _jsonrpc_reply(body)

        client = MCPClient(
            url="https://mcp.example.test/mcp",
            transport="http",
            transport_impl=httpx.MockTransport(handler),
        )
        async with client:
            result = await client.call_tool("read_file", {"path": "a.txt"})
        assert result.content == "file body"
        assert result.is_error is False

    async def test_jsonrpc_error_is_raised(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content.decode())
            if body["method"] == "tools/call":
                return httpx.Response(
                    200,
                    json={
                        "jsonrpc": "2.0",
                        "id": body["id"],
                        "error": {"code": -32601, "message": "Method not found"},
                    },
                )
            return _jsonrpc_reply(body)

        client = MCPClient(
            url="https://mcp.example.test/mcp",
            transport="http",
            transport_impl=httpx.MockTransport(handler),
        )
        async with client:
            with pytest.raises(MCPClientError, match="Method not found"):
                await client.call_tool("nope", {})

    async def test_http_error_status_is_raised(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content.decode())
            if body["method"] == "tools/list":
                return httpx.Response(500, text="boom")
            return _jsonrpc_reply(body)

        client = MCPClient(
            url="https://mcp.example.test/mcp",
            transport="http",
            transport_impl=httpx.MockTransport(handler),
        )
        async with client:
            with pytest.raises(MCPClientError, match="HTTP 500"):
                await client.list_tools()

    async def test_sse_response_is_parsed(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content.decode())
            if body["method"] == "tools/list":
                return httpx.Response(
                    200,
                    text=_sse_body(
                        {
                            "jsonrpc": "2.0",
                            "id": body["id"],
                            "result": {"tools": [{"name": "t", "inputSchema": {}}]},
                        }
                    ),
                    headers={"content-type": "text/event-stream"},
                )
            return _jsonrpc_reply(body)

        client = MCPClient(
            url="https://mcp.example.test/mcp",
            transport="sse",
            transport_impl=httpx.MockTransport(handler),
        )
        async with client:
            tools = await client.list_tools()
        assert [t.name for t in tools] == ["t"]

    async def test_auth_header_is_sent(self) -> None:
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request.headers.get("authorization", ""))
            body = json.loads(request.content.decode())
            return _jsonrpc_reply(body)

        client = MCPClient(
            url="https://mcp.example.test/mcp",
            transport="http",
            auth_token="tok-123",
            transport_impl=httpx.MockTransport(handler),
        )
        async with client:
            pass
        assert seen
        assert all(h == "Bearer tok-123" for h in seen)


class TestSSEParsing:
    def test_returns_first_jsonrpc_payload(self) -> None:
        body = _sse_body({"jsonrpc": "2.0", "id": 1, "result": {"ok": True}})
        assert MCPClient._parse_sse(body)["result"] == {"ok": True}

    def test_skips_done_marker(self) -> None:
        body = "data: [DONE]\n\n" + _sse_body({"jsonrpc": "2.0", "id": 1, "result": {}})
        assert MCPClient._parse_sse(body)["result"] == {}

    def test_raises_when_no_payload(self) -> None:
        with pytest.raises(MCPClientError, match="no JSON-RPC response"):
            MCPClient._parse_sse("event: ping\n\n")


class TestStdioTransport:
    async def test_missing_binary_raises(self) -> None:
        client = MCPClient(url="definitely-not-a-real-binary-xyz", transport="stdio")
        with pytest.raises(MCPClientError, match="Failed to start MCP server"):
            await client.connect()

    async def test_round_trip_with_a_real_subprocess(self) -> None:
        """Spawn a tiny Python MCP server and complete a real handshake."""
        script = (
            "import json,sys\n"
            "for line in sys.stdin:\n"
            "    line=line.strip()\n"
            "    if not line: continue\n"
            "    msg=json.loads(line)\n"
            "    if 'id' not in msg: continue\n"
            "    m=msg['method']\n"
            "    if m=='initialize':\n"
            "        r={'serverInfo':{'name':'mini','version':'1'}}\n"
            "    elif m=='tools/list':\n"
            "        r={'tools':[{'name':'read_file','inputSchema':{'type':'object'}}]}\n"
            "    elif m=='tools/call':\n"
            "        r={'content':[{'type':'text','text':'ok:'+msg['params']['name']}]}\n"
            "    else:\n"
            "        r={}\n"
            "    sys.stdout.write(json.dumps({'jsonrpc':'2.0','id':msg['id'],'result':r})+'\\n')\n"
            "    sys.stdout.flush()\n"
        )
        import sys as _sys

        command = f'"{_sys.executable}" -c "{script}"'
        client = MCPClient(url=command, transport="stdio", timeout=20)
        async with client:
            tools = await client.list_tools()
            result = await client.call_tool("read_file", {"path": "a"})
        assert [t.name for t in tools] == ["read_file"]
        assert result.content == "ok:read_file"
