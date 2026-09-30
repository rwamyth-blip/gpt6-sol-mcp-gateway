"""Unit tests for the orchestrator's advisory plan state (``MCP_PLAN_MODE``).

The plan is the model's own statement of intent. These tests pin three
properties that matter:

1. It is **opt-in** -- with plan mode off nothing changes.
2. It is **advisory** -- ``update_plan`` never reaches an MCP server and never
   bypasses the allowlist or the approval gate.
3. It is **robust** -- a malformed payload from the model cannot break the turn.
"""

from __future__ import annotations

from mcp_for_copilot.approval import ApprovalLayer, deny_all_approver
from mcp_for_copilot.config import Settings
from mcp_for_copilot.orchestrator import (
    PLAN_TOOL_NAME,
    LLMMCPOrchestrator,
    _parse_plan,
)
from mcp_for_copilot.provider import LLMProvider
from mcp_for_copilot.tool_router import ToolRouter

from ..conftest import make_completion, make_tool_call, mock_transport


def _settings(*, plan_mode: bool) -> Settings:
    return Settings(
        llm_api_key="sk-test",
        mcp_server_url="",
        mcp_transport="stdio",
        mcp_plan_mode=plan_mode,
        mcp_max_tool_rounds=3,
    )


def _provider(responses, capture=None) -> LLMProvider:
    return LLMProvider(
        api_key="sk-test",
        model_id="gpt-6-sol",
        transport=mock_transport(responses, capture=capture),
    )


def _orchestrator(provider, client, *, plan_mode: bool, approval=None):
    settings = _settings(plan_mode=plan_mode)
    router = ToolRouter(allowed_tools=["read_file"], require_approval=True)
    return LLMMCPOrchestrator(
        provider=provider,
        client=client,
        router=router,
        approval=approval or ApprovalLayer(),
        settings=settings,
    )


def _plan_call(steps, call_id: str = "call_plan") -> dict:
    return make_tool_call(PLAN_TOOL_NAME, {"steps": steps}, call_id=call_id)


class TestParsePlan:
    def test_accepts_dicts_with_status(self) -> None:
        steps = _parse_plan(
            [
                {"description": "read the file", "status": "done"},
                {"description": "summarise", "status": "in_progress"},
            ]
        )
        assert [(s.description, s.status) for s in steps] == [
            ("read the file", "done"),
            ("summarise", "in_progress"),
        ]

    def test_accepts_bare_strings(self) -> None:
        steps = _parse_plan(["first", "second"])
        assert [s.description for s in steps] == ["first", "second"]
        assert all(s.status == "pending" for s in steps)

    def test_unknown_status_is_normalised(self) -> None:
        steps = _parse_plan([{"description": "x", "status": "nonsense"}])
        assert steps[0].status == "pending"

    def test_non_list_payload_is_dropped(self) -> None:
        assert _parse_plan("not a list") == []
        assert _parse_plan(None) == []
        assert _parse_plan({"description": "x"}) == []

    def test_blank_descriptions_are_dropped(self) -> None:
        assert _parse_plan([{"description": "   "}, {"status": "done"}]) == []

    def test_accepts_step_key_alias(self) -> None:
        steps = _parse_plan([{"step": "aliased"}])
        assert steps[0].description == "aliased"


class TestPlanModeOff:
    async def test_plan_tool_is_not_offered(self, fake_client) -> None:
        captured: list[dict] = []
        provider = _provider(make_completion(content="ok"), capture=captured)
        orchestrator = _orchestrator(provider, fake_client, plan_mode=False)
        await orchestrator.run([{"role": "user", "content": "hi"}])

        names = [t["function"]["name"] for t in captured[0].get("tools") or []]
        assert PLAN_TOOL_NAME not in names

    async def test_system_prompt_has_no_plan_rules(self, fake_client) -> None:
        captured: list[dict] = []
        provider = _provider(make_completion(content="ok"), capture=captured)
        orchestrator = _orchestrator(provider, fake_client, plan_mode=False)
        await orchestrator.run([{"role": "user", "content": "hi"}])

        system = captured[0]["messages"][0]["content"]
        assert "update_plan" not in system

    async def test_result_plan_is_empty(self, fake_client) -> None:
        provider = _provider(make_completion(content="ok"))
        orchestrator = _orchestrator(provider, fake_client, plan_mode=False)
        result = await orchestrator.run([{"role": "user", "content": "hi"}])
        assert result.plan == []
        assert result.plan_progress["pending"] == 0


class TestPlanModeOn:
    async def test_plan_tool_is_offered(self, fake_client) -> None:
        captured: list[dict] = []
        provider = _provider(make_completion(content="ok"), capture=captured)
        orchestrator = _orchestrator(provider, fake_client, plan_mode=True)
        await orchestrator.run([{"role": "user", "content": "hi"}])

        names = [t["function"]["name"] for t in captured[0].get("tools") or []]
        assert PLAN_TOOL_NAME in names

    async def test_plan_tool_is_offered_without_any_mcp_client(self) -> None:
        captured: list[dict] = []
        provider = _provider(make_completion(content="ok"), capture=captured)
        orchestrator = LLMMCPOrchestrator(
            provider=provider, client=None, settings=_settings(plan_mode=True)
        )
        await orchestrator.run([{"role": "user", "content": "hi"}])

        names = [t["function"]["name"] for t in captured[0].get("tools") or []]
        assert names == [PLAN_TOOL_NAME]

    async def test_system_prompt_mentions_the_plan_tool(self, fake_client) -> None:
        captured: list[dict] = []
        provider = _provider(make_completion(content="ok"), capture=captured)
        orchestrator = _orchestrator(provider, fake_client, plan_mode=True)
        await orchestrator.run([{"role": "user", "content": "hi"}])

        system = captured[0]["messages"][0]["content"]
        assert "update_plan" in system
        assert "in_progress" in system

    async def test_plan_is_recorded_on_the_result(self, fake_client) -> None:
        provider = _provider(
            [
                make_completion(
                    content="",
                    tool_calls=[
                        _plan_call(
                            [
                                {"description": "read a.txt", "status": "in_progress"},
                                {"description": "summarise", "status": "pending"},
                            ]
                        )
                    ],
                    finish_reason="tool_calls",
                ),
                make_completion(content="done"),
            ]
        )
        orchestrator = _orchestrator(provider, fake_client, plan_mode=True)
        result = await orchestrator.run([{"role": "user", "content": "read a.txt"}])

        assert [s.description for s in result.plan] == ["read a.txt", "summarise"]
        assert result.plan_progress["in_progress"] == 1
        assert result.plan_progress["pending"] == 1
        assert result.plan_progress["done"] == 0

    async def test_plan_call_never_reaches_the_mcp_client(self, fake_client) -> None:
        provider = _provider(
            [
                make_completion(
                    content="",
                    tool_calls=[_plan_call([{"description": "step one"}])],
                    finish_reason="tool_calls",
                ),
                make_completion(content="done"),
            ]
        )
        orchestrator = _orchestrator(provider, fake_client, plan_mode=True)
        await orchestrator.run([{"role": "user", "content": "hi"}])

        assert fake_client.calls == []

    async def test_plan_call_is_audited_as_executed(self, fake_client) -> None:
        provider = _provider(
            [
                make_completion(
                    content="",
                    tool_calls=[_plan_call([{"description": "step one"}])],
                    finish_reason="tool_calls",
                ),
                make_completion(content="done"),
            ]
        )
        orchestrator = _orchestrator(provider, fake_client, plan_mode=True)
        result = await orchestrator.run([{"role": "user", "content": "hi"}])

        assert result.used_tools == [PLAN_TOOL_NAME]
        invocation = result.invocations[0]
        assert invocation.allowed is True
        assert invocation.approved is True
        assert invocation.executed is True
        assert invocation.is_error is False

    async def test_plan_survives_a_deny_all_approver(self, fake_client) -> None:
        """The plan is not a tool call, so the approval gate must not block it."""
        provider = _provider(
            [
                make_completion(
                    content="",
                    tool_calls=[_plan_call([{"description": "step one"}])],
                    finish_reason="tool_calls",
                ),
                make_completion(content="done"),
            ]
        )
        orchestrator = _orchestrator(
            provider, fake_client, plan_mode=True, approval=deny_all_approver()
        )
        result = await orchestrator.run([{"role": "user", "content": "hi"}])

        assert [s.description for s in result.plan] == ["step one"]
        assert result.invocations[0].executed is True

    async def test_plan_is_reminded_to_the_model_next_round(self, fake_client) -> None:
        captured: list[dict] = []
        provider = _provider(
            [
                make_completion(
                    content="",
                    tool_calls=[_plan_call([{"description": "read a.txt"}])],
                    finish_reason="tool_calls",
                ),
                make_completion(content="done"),
            ],
            capture=captured,
        )
        orchestrator = _orchestrator(provider, fake_client, plan_mode=True)
        await orchestrator.run([{"role": "user", "content": "hi"}])

        second_round = captured[1]["messages"]
        reminders = [
            m for m in second_round if m["role"] == "system" and "current plan" in m["content"]
        ]
        assert len(reminders) == 1
        assert "read a.txt" in reminders[0]["content"]

    async def test_plan_is_replaced_not_appended(self, fake_client) -> None:
        provider = _provider(
            [
                make_completion(
                    content="",
                    tool_calls=[_plan_call([{"description": "old step"}], call_id="c1")],
                    finish_reason="tool_calls",
                ),
                make_completion(
                    content="",
                    tool_calls=[
                        _plan_call(
                            [
                                {"description": "old step", "status": "done"},
                                {"description": "new step", "status": "in_progress"},
                            ],
                            call_id="c2",
                        )
                    ],
                    finish_reason="tool_calls",
                ),
                make_completion(content="done"),
            ]
        )
        orchestrator = _orchestrator(provider, fake_client, plan_mode=True)
        result = await orchestrator.run([{"role": "user", "content": "hi"}])

        assert [s.description for s in result.plan] == ["old step", "new step"]
        assert result.plan_progress["done"] == 1

    async def test_plan_is_reset_between_turns(self, fake_client) -> None:
        provider = _provider(
            [
                make_completion(
                    content="",
                    tool_calls=[_plan_call([{"description": "turn one step"}])],
                    finish_reason="tool_calls",
                ),
                make_completion(content="done"),
                make_completion(content="plain answer"),
            ]
        )
        orchestrator = _orchestrator(provider, fake_client, plan_mode=True)
        first = await orchestrator.run([{"role": "user", "content": "hi"}])
        second = await orchestrator.run([{"role": "user", "content": "again"}])

        assert len(first.plan) == 1
        assert second.plan == []


class TestMalformedPlan:
    async def test_non_list_steps_is_an_error_result_not_a_crash(self, fake_client) -> None:
        provider = _provider(
            [
                make_completion(
                    content="",
                    tool_calls=[make_tool_call(PLAN_TOOL_NAME, {"steps": "oops"})],
                    finish_reason="tool_calls",
                ),
                make_completion(content="recovered"),
            ]
        )
        orchestrator = _orchestrator(provider, fake_client, plan_mode=True)
        result = await orchestrator.run([{"role": "user", "content": "hi"}])

        assert result.content == "recovered"
        assert result.plan == []
        invocation = result.invocations[0]
        assert invocation.is_error is True
        assert invocation.executed is False
        assert "list" in invocation.reason

    async def test_missing_steps_key_is_an_error_result(self, fake_client) -> None:
        provider = _provider(
            [
                make_completion(
                    content="",
                    tool_calls=[make_tool_call(PLAN_TOOL_NAME, {})],
                    finish_reason="tool_calls",
                ),
                make_completion(content="recovered"),
            ]
        )
        orchestrator = _orchestrator(provider, fake_client, plan_mode=True)
        result = await orchestrator.run([{"role": "user", "content": "hi"}])

        assert result.invocations[0].is_error is True
        assert result.content == "recovered"

    async def test_empty_step_list_leaves_the_plan_untouched(self, fake_client) -> None:
        provider = _provider(
            [
                make_completion(
                    content="",
                    tool_calls=[_plan_call([{"description": "keep me"}], call_id="c1")],
                    finish_reason="tool_calls",
                ),
                make_completion(
                    content="",
                    tool_calls=[_plan_call([], call_id="c2")],
                    finish_reason="tool_calls",
                ),
                make_completion(content="done"),
            ]
        )
        orchestrator = _orchestrator(provider, fake_client, plan_mode=True)
        result = await orchestrator.run([{"role": "user", "content": "hi"}])

        assert [s.description for s in result.plan] == ["keep me"]

    async def test_plan_and_real_tool_in_the_same_round(self, fake_client) -> None:
        provider = _provider(
            [
                make_completion(
                    content="",
                    tool_calls=[
                        _plan_call([{"description": "read a.txt"}], call_id="c1"),
                        make_tool_call("read_file", {"path": "a.txt"}, call_id="c2"),
                    ],
                    finish_reason="tool_calls",
                ),
                make_completion(content="done"),
            ]
        )
        orchestrator = _orchestrator(provider, fake_client, plan_mode=True)
        result = await orchestrator.run([{"role": "user", "content": "hi"}])

        assert fake_client.calls == [("read_file", {"path": "a.txt"})]
        assert result.used_tools == [PLAN_TOOL_NAME, "read_file"]
        assert [s.description for s in result.plan] == ["read a.txt"]
