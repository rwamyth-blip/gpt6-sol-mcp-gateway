"""Unit tests for orchestrator loop detection.

Loop detection mirrors the CLI behaviour in Cline: a soft threshold warns the
model, a hard threshold stops the turn. Only *identical* calls count, so a
model that varies its arguments is making progress and must not be stopped.
"""

from __future__ import annotations

from mcp_for_copilot.approval import ApprovalLayer
from mcp_for_copilot.config import Settings
from mcp_for_copilot.orchestrator import LLMMCPOrchestrator, LoopDetector
from mcp_for_copilot.provider import LLMProvider
from mcp_for_copilot.tool_router import ToolRouter

from ..conftest import make_completion, make_tool_call, mock_transport


def _provider(responses, capture=None) -> LLMProvider:
    return LLMProvider(
        api_key="sk-test",
        model_id="gpt-6-sol",
        transport=mock_transport(responses, capture=capture),
    )


def _orchestrator(provider, client, *, settings=None, allowed=("read_file",)) -> LLMMCPOrchestrator:
    router = ToolRouter(allowed_tools=list(allowed), require_approval=True)
    return LLMMCPOrchestrator(
        provider=provider,
        client=client,
        router=router,
        approval=ApprovalLayer(),
        settings=settings,
    )


def _settings(**overrides) -> Settings:
    base = {
        "llm_api_key": "sk-test",
        "mcp_max_tool_rounds": 10,
        "mcp_loop_detection": True,
        "mcp_loop_soft_threshold": 3,
        "mcp_loop_hard_threshold": 5,
    }
    base.update(overrides)
    return Settings(**base)


class TestLoopDetectorUnit:
    def test_identical_calls_are_counted(self) -> None:
        detector = LoopDetector(soft_threshold=3, hard_threshold=5)
        assert detector.record("read_file", {"path": "a"}) == (False, False, 1)
        assert detector.record("read_file", {"path": "a"}) == (False, False, 2)
        assert detector.record("read_file", {"path": "a"}) == (True, False, 3)
        assert detector.record("read_file", {"path": "a"}) == (False, False, 4)
        assert detector.record("read_file", {"path": "a"}) == (False, True, 5)

    def test_warning_fires_only_once(self) -> None:
        detector = LoopDetector(soft_threshold=2, hard_threshold=4)
        detector.record("read_file", {"path": "a"})
        assert detector.record("read_file", {"path": "a"})[0] is True
        # The second soft hit must not warn again.
        assert detector.record("read_file", {"path": "a"})[0] is False

    def test_different_arguments_are_not_a_loop(self) -> None:
        detector = LoopDetector(soft_threshold=2, hard_threshold=3)
        for index in range(10):
            warn, stop, _ = detector.record("read_file", {"path": f"{index}.txt"})
            assert warn is False
            assert stop is False

    def test_key_order_does_not_change_the_signature(self) -> None:
        detector = LoopDetector(soft_threshold=2, hard_threshold=3)
        detector.record("write_file", {"path": "a", "content": "b"})
        warn, _, count = detector.record("write_file", {"content": "b", "path": "a"})
        assert count == 2
        assert warn is True

    def test_different_tools_are_tracked_separately(self) -> None:
        detector = LoopDetector(soft_threshold=2, hard_threshold=3)
        detector.record("read_file", {"path": "a"})
        warn, stop, count = detector.record("list_files", {"path": "a"})
        assert (warn, stop, count) == (False, False, 1)

    def test_hard_threshold_is_forced_above_soft(self) -> None:
        detector = LoopDetector(soft_threshold=5, hard_threshold=2)
        assert detector.hard_threshold == 6

    def test_repeated_reports_only_duplicates(self) -> None:
        detector = LoopDetector(soft_threshold=2, hard_threshold=5)
        detector.record("read_file", {"path": "a"})
        detector.record("read_file", {"path": "a"})
        detector.record("list_files", {"path": "b"})
        assert list(detector.repeated.values()) == [2]


class TestLoopDetectionInRun:
    async def test_hard_threshold_stops_the_turn(self, fake_client) -> None:
        # The model asks for the same call forever; the hard threshold must
        # stop it well before the round cap.
        provider = _provider(
            make_completion(
                content="",
                tool_calls=[make_tool_call("read_file", {"path": "a.txt"})],
                finish_reason="tool_calls",
            )
        )
        orchestrator = _orchestrator(provider, fake_client, settings=_settings())
        result = await orchestrator.run([{"role": "user", "content": "loop"}])

        assert result.loop_stopped is True
        assert len(fake_client.calls) == 5
        assert "repeated" in result.content.lower()

    async def test_soft_threshold_injects_a_warning(self, fake_client) -> None:
        captured: list[dict] = []
        provider = _provider(
            make_completion(
                content="",
                tool_calls=[make_tool_call("read_file", {"path": "a.txt"})],
                finish_reason="tool_calls",
            ),
            capture=captured,
        )
        orchestrator = _orchestrator(provider, fake_client, settings=_settings())
        await orchestrator.run([{"role": "user", "content": "loop"}])

        # The warning is injected once into the history, so it appears in the
        # request that follows the 3rd identical call and in every later one.
        first_warning_request = next(
            index
            for index, request in enumerate(captured)
            if any(
                message["role"] == "system"
                and "same tool with the same arguments" in message["content"]
                for message in request["messages"]
            )
        )
        assert first_warning_request == 3
        # Exactly one warning message is ever added to the history.
        assert (
            sum(
                1
                for message in captured[-1]["messages"]
                if message["role"] == "system"
                and "same tool with the same arguments" in message["content"]
            )
            == 1
        )

    async def test_varying_arguments_never_stop_the_turn(self, fake_client) -> None:
        responses = [
            make_completion(
                content="",
                tool_calls=[make_tool_call("read_file", {"path": f"{i}.txt"}, call_id=f"c{i}")],
                finish_reason="tool_calls",
            )
            for i in range(6)
        ]
        responses.append(make_completion(content="finished"))
        provider = _provider(responses)
        orchestrator = _orchestrator(provider, fake_client, settings=_settings())
        result = await orchestrator.run([{"role": "user", "content": "read many"}])

        assert result.loop_stopped is False
        assert result.content == "finished"
        assert len(fake_client.calls) == 6

    async def test_detection_can_be_disabled(self, fake_client) -> None:
        provider = _provider(
            make_completion(
                content="",
                tool_calls=[make_tool_call("read_file", {"path": "a.txt"})],
                finish_reason="tool_calls",
            )
        )
        orchestrator = _orchestrator(
            provider, fake_client, settings=_settings(mcp_loop_detection=False)
        )
        result = await orchestrator.run([{"role": "user", "content": "loop"}], max_rounds=3)

        assert result.loop_stopped is False
        assert len(fake_client.calls) == 3

    async def test_state_does_not_leak_between_turns(self, fake_client) -> None:
        provider = _provider(
            make_completion(
                content="",
                tool_calls=[make_tool_call("read_file", {"path": "a.txt"})],
                finish_reason="tool_calls",
            )
        )
        orchestrator = _orchestrator(provider, fake_client, settings=_settings())
        first = await orchestrator.run([{"role": "user", "content": "one"}])
        assert first.loop_stopped is True

        # A fresh turn must start with a clean counter, not inherit the stop.
        second = await orchestrator.run([{"role": "user", "content": "two"}], max_rounds=1)
        assert second.loop_stopped is False
