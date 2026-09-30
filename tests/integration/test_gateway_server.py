"""Integration tests for the stdio MCP server's tool handlers."""

from __future__ import annotations

import json

from gpt6_sol_mcp.gateway.facade import Gateway
from gpt6_sol_mcp.gateway.server import SERVER_NAME, build_server
from gpt6_sol_mcp.provider import LLMProvider

from ..conftest import FakeMCPClient, make_completion, mock_transport


def _gateway(fake_client: FakeMCPClient | None = None) -> Gateway:
    from gpt6_sol_mcp.config import Settings

    settings = Settings(
        llm_api_key="sk-test",
        llm_model_id="gpt-6-sol",
        llm_base_url="https://api.example.test/v1",
        mcp_server_url="",
    )
    provider = LLMProvider(
        api_key="sk-test",
        base_url=settings.llm_base_url,
        model_id="gpt-6-sol",
        transport=mock_transport(make_completion(content="Answer from the model")),
    )
    gateway = Gateway(settings=settings, provider=provider, connect_mcp=False)
    gateway.client = fake_client
    gateway.orchestrator.client = fake_client
    gateway._connected = True
    return gateway


async def _call(server, name: str, arguments: dict | None = None) -> list:
    """Invoke the registered ``tools/call`` handler directly."""
    from mcp.types import CallToolRequestParams

    handler = server._request_handlers["tools/call"].handler
    result = await handler(None, CallToolRequestParams(name=name, arguments=arguments or {}))
    return result.content


async def _list_tools(server) -> list:
    """Invoke the registered ``tools/list`` handler directly."""
    handler = server._request_handlers["tools/list"].handler
    result = await handler(None, None)
    return result.tools


class TestServerMetadata:
    def test_server_name(self) -> None:
        assert SERVER_NAME == "gpt6-sol-mcp-gateway"

    async def test_lists_all_tools(self) -> None:
        server = build_server(_gateway())
        names = [t.name for t in await _list_tools(server)]
        assert names == [
            "gpt6_chat",
            "gpt6_models",
            "gpt6_status",
            "gpt6_tools",
            "gpt6_plan",
            "gpt6_debug_marathon",
            "gpt6_debug_marathon_status",
            "copilot_install",
            "hubspot_track",
            "vihokai_deploy",
        ]

    async def test_chat_tool_schema_requires_prompt(self) -> None:
        server = build_server(_gateway())
        chat = next(t for t in await _list_tools(server) if t.name == "gpt6_chat")
        assert chat.input_schema["required"] == ["prompt"]


class TestChatTool:
    async def test_returns_model_answer(self) -> None:
        server = build_server(_gateway())
        content = await _call(server, "gpt6_chat", {"prompt": "hello"})
        payload = json.loads(content[0].text)
        assert payload["content"] == "Answer from the model"
        assert payload["model"] == "gpt-6-sol"

    async def test_payload_includes_empty_plan_by_default(self) -> None:
        server = build_server(_gateway())
        content = await _call(server, "gpt6_chat", {"prompt": "hello"})
        payload = json.loads(content[0].text)
        assert payload["plan"] == []
        assert payload["plan_progress"] == {
            "pending": 0,
            "in_progress": 0,
            "done": 0,
            "blocked": 0,
        }

    async def test_missing_prompt_is_an_error_string(self) -> None:
        server = build_server(_gateway())
        content = await _call(server, "gpt6_chat", {})
        assert content[0].text.startswith("Error:")

    async def test_blank_prompt_is_an_error_string(self) -> None:
        server = build_server(_gateway())
        content = await _call(server, "gpt6_chat", {"prompt": "   "})
        assert content[0].text.startswith("Error:")


class TestModelsTool:
    async def test_lists_models_with_aliases(self) -> None:
        server = build_server(_gateway())
        content = await _call(server, "gpt6_models")
        payload = json.loads(content[0].text)
        assert "gpt-6-sol" in payload
        assert "sol" in payload["gpt-6-sol"]["aliases"]


class TestStatusTool:
    async def test_status_has_no_secrets(self) -> None:
        server = build_server(_gateway())
        content = await _call(server, "gpt6_status")
        assert "sk-test" not in content[0].text
        payload = json.loads(content[0].text)
        assert payload["llm"]["api_key_present"] is True


class TestToolsTool:
    async def test_lists_allowed_tool_names(self, fake_client: FakeMCPClient) -> None:
        server = build_server(_gateway(fake_client))
        content = await _call(server, "gpt6_tools")
        names = json.loads(content[0].text)
        assert "read_file" in names
        assert "write_file" not in names  # not in the read-only default allowlist


class TestDebugMarathonTools:
    async def test_submit_tool_returns_job_id(self, monkeypatch) -> None:
        gateway = _gateway()
        monkeypatch.setattr(
            gateway.debug_marathon,
            "submit",
            lambda tasks: {"job_id": "job-1", "status": "queued", "total": len(tasks)},
        )
        content = await _call(
            build_server(gateway),
            "gpt6_debug_marathon",
            {"tasks": [{"question": "Find the crash"}]},
        )
        payload = json.loads(content[0].text)
        assert payload == {"job_id": "job-1", "status": "queued", "total": 1}

    async def test_status_tool_returns_job(self, monkeypatch) -> None:
        gateway = _gateway()
        monkeypatch.setattr(
            gateway.debug_marathon,
            "get",
            lambda job_id: {"job_id": job_id, "status": "running"},
        )
        content = await _call(
            build_server(gateway),
            "gpt6_debug_marathon_status",
            {"job_id": "job-1"},
        )
        assert json.loads(content[0].text) == {"job_id": "job-1", "status": "running"}


class TestPlanTool:
    async def test_plan_is_empty_before_any_chat(self) -> None:
        server = build_server(_gateway())
        content = await _call(server, "gpt6_plan")
        assert json.loads(content[0].text) == {"plan": [], "plan_progress": {}}

    async def test_plan_reflects_the_last_chat(self) -> None:
        gateway = _gateway()
        server = build_server(gateway)
        await _call(server, "gpt6_chat", {"prompt": "hello"})
        content = await _call(server, "gpt6_plan")
        payload = json.loads(content[0].text)
        assert payload["plan"] == []
        assert payload["plan_progress"] == {
            "pending": 0,
            "in_progress": 0,
            "done": 0,
            "blocked": 0,
        }

    async def test_plan_tool_does_not_require_arguments(self) -> None:
        server = build_server(_gateway())
        plan = next(t for t in await _list_tools(server) if t.name == "gpt6_plan")
        assert plan.input_schema["properties"] == {}


class TestUnknownTool:
    async def test_unknown_tool_returns_error_string(self) -> None:
        server = build_server(_gateway())
        content = await _call(server, "not_a_tool")
        assert "unknown tool" in content[0].text


class TestErrorHandling:
    async def test_handler_exception_becomes_error_string(self) -> None:
        class ExplodingGateway(Gateway):
            async def chat(self, *args, **kwargs):
                raise RuntimeError("model exploded")

        gateway = _gateway()
        gateway.chat = ExplodingGateway.chat.__get__(gateway)  # type: ignore[method-assign]
        server = build_server(gateway)
        content = await _call(server, "gpt6_chat", {"prompt": "hi"})
        assert content[0].text.startswith("Error:")
        assert "model exploded" in content[0].text
