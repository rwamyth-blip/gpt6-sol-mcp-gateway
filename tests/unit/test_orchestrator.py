"""Unit tests for the orchestrator — the three-gate tool loop."""

from __future__ import annotations

from mcp_for_copilot.approval import ApprovalLayer, allow_all_approver, deny_all_approver
from mcp_for_copilot.orchestrator import LLMMCPOrchestrator
from mcp_for_copilot.provider import LLMProvider
from mcp_for_copilot.tool_router import ToolRouter

from ..conftest import FakeMCPClient, make_completion, make_tool_call, mock_transport


def _provider(responses, capture=None) -> LLMProvider:
    return LLMProvider(
        api_key="sk-test",
        model_id="gpt-6-sol",
        transport=mock_transport(responses, capture=capture),
    )


def _orchestrator(provider, client, *, allowed=("read_file", "write_file"), approval=None):
    router = ToolRouter(allowed_tools=list(allowed), require_approval=True)
    return LLMMCPOrchestrator(
        provider=provider,
        client=client,
        router=router,
        approval=approval or ApprovalLayer(),
    )


class TestNoToolPath:
    async def test_plain_answer_returns_immediately(self, fake_client) -> None:
        provider = _provider(make_completion(content="Just an answer"))
        orchestrator = _orchestrator(provider, fake_client)
        result = await orchestrator.run([{"role": "user", "content": "hi"}])
        assert result.content == "Just an answer"
        assert result.rounds == 1
        assert result.used_tools == []

    async def test_no_client_means_no_tools(self) -> None:
        provider = _provider(make_completion(content="ok"))
        orchestrator = LLMMCPOrchestrator(provider=provider, client=None)
        result = await orchestrator.run([{"role": "user", "content": "hi"}])
        assert result.content == "ok"


class TestToolLoop:
    async def test_read_only_tool_executes(self, fake_client) -> None:
        provider = _provider(
            [
                make_completion(
                    content="",
                    tool_calls=[make_tool_call("read_file", {"path": "a.txt"})],
                    finish_reason="tool_calls",
                ),
                make_completion(content="The file says hello."),
            ]
        )
        orchestrator = _orchestrator(provider, fake_client)
        result = await orchestrator.run([{"role": "user", "content": "read a.txt"}])

        assert result.content == "The file says hello."
        assert result.used_tools == ["read_file"]
        assert fake_client.calls == [("read_file", {"path": "a.txt"})]

    async def test_tool_result_is_fed_back_to_the_model(self, fake_client) -> None:
        captured: list[dict] = []
        provider = _provider(
            [
                make_completion(
                    content="",
                    tool_calls=[make_tool_call("read_file", {"path": "a.txt"})],
                    finish_reason="tool_calls",
                ),
                make_completion(content="done"),
            ],
            capture=captured,
        )
        orchestrator = _orchestrator(provider, fake_client)
        await orchestrator.run([{"role": "user", "content": "read a.txt"}])

        second_call_messages = captured[1]["messages"]
        tool_messages = [m for m in second_call_messages if m["role"] == "tool"]
        assert len(tool_messages) == 1
        assert "untrusted data" in tool_messages[0]["content"]

    async def test_round_cap_is_enforced(self, fake_client) -> None:
        # The model always asks for a tool, so the cap must stop the loop.
        provider = _provider(
            make_completion(
                content="",
                tool_calls=[make_tool_call("read_file", {"path": "a.txt"})],
                finish_reason="tool_calls",
            )
        )
        orchestrator = _orchestrator(provider, fake_client)
        result = await orchestrator.run([{"role": "user", "content": "loop"}], max_rounds=2)
        assert result.rounds == 3  # cap + 1 final attempt
        assert len(fake_client.calls) == 2


class TestGateOneAllowlist:
    async def test_tool_outside_allowlist_is_blocked(self, fake_client) -> None:
        provider = _provider(
            [
                make_completion(
                    content="",
                    tool_calls=[make_tool_call("delete_file", {"path": "a"})],
                    finish_reason="tool_calls",
                ),
                make_completion(content="I could not delete it."),
            ]
        )
        orchestrator = _orchestrator(provider, fake_client, allowed=("read_file",))
        result = await orchestrator.run([{"role": "user", "content": "delete a"}])

        assert fake_client.calls == []
        assert result.invocations[0].allowed is False
        assert "allowlist" in result.invocations[0].reason

    async def test_invalid_arguments_are_blocked(self, fake_client) -> None:
        provider = _provider(
            [
                make_completion(
                    content="",
                    tool_calls=[make_tool_call("read_file", {})],  # path is required
                    finish_reason="tool_calls",
                ),
                make_completion(content="Missing path."),
            ]
        )
        orchestrator = _orchestrator(provider, fake_client)
        result = await orchestrator.run([{"role": "user", "content": "read"}])

        assert fake_client.calls == []
        assert "invalid arguments" in result.invocations[0].reason

    async def test_malformed_json_arguments_are_blocked(self, fake_client) -> None:
        provider = _provider(
            [
                make_completion(
                    content="",
                    tool_calls=[make_tool_call("read_file", "{not json")],
                    finish_reason="tool_calls",
                ),
                make_completion(content="Bad call."),
            ]
        )
        orchestrator = _orchestrator(provider, fake_client)
        result = await orchestrator.run([{"role": "user", "content": "read"}])
        assert fake_client.calls == []
        assert result.invocations[0].allowed is False


class TestGateTwoApproval:
    async def test_risky_tool_without_approver_is_denied(self, fake_client) -> None:
        provider = _provider(
            [
                make_completion(
                    content="",
                    tool_calls=[make_tool_call("write_file", {"path": "a", "content": "x"})],
                    finish_reason="tool_calls",
                ),
                make_completion(content="Not approved."),
            ]
        )
        orchestrator = _orchestrator(provider, fake_client)  # no approver -> fail closed
        result = await orchestrator.run([{"role": "user", "content": "write a"}])

        assert fake_client.calls == []
        assert result.invocations[0].approved is False

    async def test_risky_tool_with_allow_approver_executes(self, fake_client) -> None:
        provider = _provider(
            [
                make_completion(
                    content="",
                    tool_calls=[make_tool_call("write_file", {"path": "a", "content": "x"})],
                    finish_reason="tool_calls",
                ),
                make_completion(content="Written."),
            ]
        )
        orchestrator = _orchestrator(
            provider, fake_client, approval=ApprovalLayer(allow_all_approver())
        )
        result = await orchestrator.run([{"role": "user", "content": "write a"}])

        assert fake_client.calls == [("write_file", {"path": "a", "content": "x"})]
        assert result.invocations[0].approved is True

    async def test_risky_tool_with_deny_approver_is_blocked(self, fake_client) -> None:
        provider = _provider(
            [
                make_completion(
                    content="",
                    tool_calls=[make_tool_call("write_file", {"path": "a", "content": "x"})],
                    finish_reason="tool_calls",
                ),
                make_completion(content="Denied."),
            ]
        )
        orchestrator = _orchestrator(
            provider, fake_client, approval=ApprovalLayer(deny_all_approver())
        )
        result = await orchestrator.run([{"role": "user", "content": "write a"}])
        assert fake_client.calls == []
        assert result.invocations[0].approved is False

    async def test_read_only_tool_skips_approval(self, fake_client) -> None:
        provider = _provider(
            [
                make_completion(
                    content="",
                    tool_calls=[make_tool_call("read_file", {"path": "a"})],
                    finish_reason="tool_calls",
                ),
                make_completion(content="ok"),
            ]
        )
        orchestrator = _orchestrator(provider, fake_client)
        result = await orchestrator.run([{"role": "user", "content": "read a"}])
        assert result.invocations[0].approved is None
        assert result.invocations[0].executed is True


class TestGateThreeDispatch:
    async def test_tool_error_is_reported_not_raised(self, fake_client) -> None:
        from mcp_for_copilot.mcp_client import MCPCallResult

        fake_client._results["read_file"] = MCPCallResult(
            tool="read_file", content="permission denied", is_error=True
        )
        provider = _provider(
            [
                make_completion(
                    content="",
                    tool_calls=[make_tool_call("read_file", {"path": "a"})],
                    finish_reason="tool_calls",
                ),
                make_completion(content="I could not read it."),
            ]
        )
        orchestrator = _orchestrator(provider, fake_client)
        result = await orchestrator.run([{"role": "user", "content": "read a"}])

        assert result.content == "I could not read it."
        assert result.invocations[0].is_error is True

    async def test_client_exception_is_captured(self, fake_client) -> None:
        from mcp_for_copilot.mcp_client import MCPClientError

        async def boom(name, arguments=None):
            raise MCPClientError("transport died")

        fake_client.call_tool = boom  # type: ignore[method-assign]
        provider = _provider(
            [
                make_completion(
                    content="",
                    tool_calls=[make_tool_call("read_file", {"path": "a"})],
                    finish_reason="tool_calls",
                ),
                make_completion(content="Tool failed."),
            ]
        )
        orchestrator = _orchestrator(provider, fake_client)
        result = await orchestrator.run([{"role": "user", "content": "read a"}])
        assert result.invocations[0].executed is False
        assert "transport died" in result.invocations[0].reason

    async def test_no_client_configured(self) -> None:
        provider = _provider(
            [
                make_completion(
                    content="",
                    tool_calls=[make_tool_call("read_file", {"path": "a"})],
                    finish_reason="tool_calls",
                ),
                make_completion(content="No tools available."),
            ]
        )
        router = ToolRouter(allowed_tools=["read_file"], require_approval=True)
        from mcp_for_copilot.mcp_client import MCPTool

        router.register(MCPTool(name="read_file", input_schema={}))
        orchestrator = LLMMCPOrchestrator(
            provider=provider, client=None, router=router, approval=ApprovalLayer()
        )
        result = await orchestrator.run([{"role": "user", "content": "read a"}])
        assert "no MCP client configured" in result.invocations[0].reason


class TestProviderFailure:
    async def test_provider_error_becomes_content(self, fake_client) -> None:
        import httpx

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="upstream exploded")

        provider = LLMProvider(
            api_key="sk-test",
            model_id="gpt-6-sol",
            transport=httpx.MockTransport(handler),
        )
        orchestrator = _orchestrator(provider, fake_client)
        result = await orchestrator.run([{"role": "user", "content": "hi"}])
        assert "Provider error" in result.content


class TestSystemPrompt:
    async def test_system_prompt_is_prepended(self, fake_client) -> None:
        captured: list[dict] = []
        provider = _provider(make_completion(content="ok"), capture=captured)
        orchestrator = _orchestrator(provider, fake_client)
        await orchestrator.run([{"role": "user", "content": "hi"}], system="You are terse.")
        assert captured[0]["messages"][0]["role"] == "system"
        assert "You are terse." in captured[0]["messages"][0]["content"]

    async def test_default_system_prompt_when_none_given(self, fake_client) -> None:
        captured: list[dict] = []
        provider = _provider(make_completion(content="ok"), capture=captured)
        orchestrator = _orchestrator(provider, fake_client)
        await orchestrator.run([{"role": "user", "content": "hi"}])
        assert "tool-using assistant" in captured[0]["messages"][0]["content"]


class TestToolSchemas:
    async def test_allowed_tools_are_sent_to_the_model(self, fake_client) -> None:
        captured: list[dict] = []
        provider = _provider(make_completion(content="ok"), capture=captured)
        orchestrator = _orchestrator(provider, fake_client, allowed=("read_file",))
        await orchestrator.run([{"role": "user", "content": "hi"}])

        names = [t["function"]["name"] for t in captured[0]["tools"]]
        assert names == ["read_file"]

    async def test_no_tools_means_no_tools_key(self, fake_client) -> None:
        captured: list[dict] = []
        provider = _provider(make_completion(content="ok"), capture=captured)
        orchestrator = _orchestrator(provider, fake_client, allowed=())
        await orchestrator.run([{"role": "user", "content": "hi"}])
        assert "tools" not in captured[0]

    async def test_list_tools_failure_is_tolerated(self) -> None:
        from mcp_for_copilot.mcp_client import MCPClientError

        class BrokenClient(FakeMCPClient):
            async def list_tools(self):
                raise MCPClientError("cannot list")

        captured: list[dict] = []
        provider = _provider(make_completion(content="ok"), capture=captured)
        orchestrator = _orchestrator(provider, BrokenClient())
        result = await orchestrator.run([{"role": "user", "content": "hi"}])
        assert result.content == "ok"
        assert "tools" not in captured[0]
