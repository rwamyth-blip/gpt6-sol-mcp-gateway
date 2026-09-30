"""Unit tests for the debug marathon's plan and verification markers.

The marathon never executes code, so the interesting properties are:

1. ``verification`` defaults to ``not_executed`` and only upgrades when a stage
   explicitly says it was handed real execution output.
2. The advisory plan is parsed out of a single marker line and never leaks into
   the human-readable answer.
3. A stage that omits the markers entirely still behaves exactly as before.
"""

from __future__ import annotations

import asyncio

from mcp_for_copilot.debug_marathon import (
    MAX_PLAN_STEPS,
    VERIFICATION_MARKER,
    DebugMarathon,
    _parse_plan,
    _parse_verification,
    _strip_markers,
    _task_verification,
)
from mcp_for_copilot.provider import ProviderResponse


def _task(title: str, priority: int = 3, difficulty: int = 3, complexity: int = 3) -> dict:
    return {
        "title": title,
        "question": f"Investigate {title}",
        "priority": priority,
        "difficulty": difficulty,
        "complexity": complexity,
    }


class _ScriptedProvider:
    """Replays one canned answer per model, in call order."""

    def __init__(self, answers: dict[str, str]) -> None:
        self.answers = answers
        self.models: list[str] = []

    async def complete(self, messages, *, model_id, reasoning_effort):
        self.models.append(model_id)
        return ProviderResponse(content=self.answers[model_id], model=model_id)


def _run(provider: _ScriptedProvider, tasks: list[dict]) -> dict:
    async def go():
        marathon = DebugMarathon(provider)  # type: ignore[arg-type]
        submitted = marathon.submit(tasks)
        await marathon._worker
        return marathon.get(submitted["job_id"])

    return asyncio.run(go())


class TestParseVerification:
    def test_defaults_to_not_executed_when_absent(self) -> None:
        assert _parse_verification("just an answer") == "not_executed"

    def test_reads_the_marker(self) -> None:
        raw = f"answer\n{VERIFICATION_MARKER}: executed"
        assert _parse_verification(raw) == "executed"

    def test_is_case_insensitive(self) -> None:
        raw = f"answer\n{VERIFICATION_MARKER}: NOT_EXECUTED"
        assert _parse_verification(raw) == "not_executed"

    def test_unknown_value_falls_back_to_the_safe_default(self) -> None:
        raw = f"answer\n{VERIFICATION_MARKER}: probably_fine"
        assert _parse_verification(raw) == "not_executed"


class TestParsePlan:
    def test_reads_status_prefixed_steps(self) -> None:
        raw = "MARATHON_PLAN: done: read the log; in_progress: patch the parser"
        assert _parse_plan(raw) == [
            {"description": "read the log", "status": "done"},
            {"description": "patch the parser", "status": "in_progress"},
        ]

    def test_unprefixed_items_become_pending(self) -> None:
        assert _parse_plan("MARATHON_PLAN: just do the thing") == [
            {"description": "just do the thing", "status": "pending"}
        ]

    def test_absent_marker_yields_no_plan(self) -> None:
        assert _parse_plan("no plan here") == []

    def test_step_count_is_capped(self) -> None:
        raw = "MARATHON_PLAN: " + "; ".join(f"step {i}" for i in range(20))
        assert len(_parse_plan(raw)) == MAX_PLAN_STEPS

    def test_blank_chunks_are_skipped(self) -> None:
        assert _parse_plan("MARATHON_PLAN: done: a;; ; done: b") == [
            {"description": "a", "status": "done"},
            {"description": "b", "status": "done"},
        ]


class TestStripMarkers:
    def test_removes_both_marker_lines(self) -> None:
        raw = (
            "The parser drops the last line.\n"
            "MARATHON_PLAN: done: read the log\n"
            f"{VERIFICATION_MARKER}: not_executed"
        )
        assert _strip_markers(raw) == "The parser drops the last line."

    def test_leaves_a_plain_answer_untouched(self) -> None:
        assert _strip_markers("  plain answer  ") == "plain answer"


class TestTaskVerification:
    def test_defaults_to_not_executed(self) -> None:
        assert _task_verification([{"verification": "not_executed"}]) == "not_executed"

    def test_any_executed_stage_upgrades_the_task(self) -> None:
        stages = [{"verification": "not_executed"}, {"verification": "executed"}]
        assert _task_verification(stages) == "executed"

    def test_empty_stage_list_is_not_executed(self) -> None:
        assert _task_verification([]) == "not_executed"


class TestMarathonIntegration:
    def test_plan_and_verification_are_recorded_per_stage(self) -> None:
        provider = _ScriptedProvider(
            {
                "gpt-6-luna": (
                    "Triage: the parser drops the last line.\n"
                    "MARATHON_PLAN: done: read the log; in_progress: patch the parser\n"
                    f"{VERIFICATION_MARKER}: not_executed\n"
                    "MARATHON_STATUS: SOLVED"
                )
            }
        )
        job = _run(provider, [_task("simple", 1, 1, 1)])

        result = job["results"][0]
        stage = result["stages"][0]
        assert stage["plan"] == [
            {"description": "read the log", "status": "done"},
            {"description": "patch the parser", "status": "in_progress"},
        ]
        assert stage["verification"] == "not_executed"
        assert result["plan"] == stage["plan"]
        assert result["verification"] == "not_executed"

    def test_markers_do_not_leak_into_the_answer(self) -> None:
        provider = _ScriptedProvider(
            {
                "gpt-6-luna": (
                    "Triage: the parser drops the last line.\n"
                    "MARATHON_PLAN: done: read the log\n"
                    f"{VERIFICATION_MARKER}: not_executed\n"
                    "MARATHON_STATUS: SOLVED"
                )
            }
        )
        job = _run(provider, [_task("simple", 1, 1, 1)])

        answer = job["results"][0]["final_answer"]
        assert answer == "Triage: the parser drops the last line."
        assert "MARATHON_PLAN" not in answer
        assert VERIFICATION_MARKER not in answer

    def test_verification_upgrades_when_a_stage_reports_execution(self) -> None:
        provider = _ScriptedProvider(
            {
                "gpt-6-luna": (
                    f"Triage done.\n{VERIFICATION_MARKER}: executed\nMARATHON_STATUS: SOLVED"
                )
            }
        )
        job = _run(provider, [_task("simple", 1, 1, 1)])

        assert job["results"][0]["verification"] == "executed"
        assert job["verification"] == "executed"

    def test_job_verification_defaults_to_not_executed(self) -> None:
        provider = _ScriptedProvider({"gpt-6-luna": "Triage done.\nMARATHON_STATUS: SOLVED"})
        job = _run(provider, [_task("simple", 1, 1, 1)])

        assert job["verification"] == "not_executed"
        assert job["results"][0]["verification"] == "not_executed"

    def test_stage_without_markers_still_works(self) -> None:
        """Backward compatibility: the old answer shape must keep working."""
        provider = _ScriptedProvider(
            {"gpt-6-luna": "review from gpt-6-luna\nMARATHON_STATUS: SOLVED"}
        )
        job = _run(provider, [_task("simple", 1, 1, 1)])

        result = job["results"][0]
        assert result["status"] == "completed"
        assert result["final_answer"] == "review from gpt-6-luna"
        assert result["plan"] == []
        assert result["stages"][0]["plan"] == []

    def test_failed_stage_carries_empty_plan_and_safe_verification(self) -> None:
        class _BoomProvider:
            async def complete(self, messages, *, model_id, reasoning_effort):
                raise RuntimeError("provider exploded")

        async def go():
            marathon = DebugMarathon(_BoomProvider())  # type: ignore[arg-type]
            submitted = marathon.submit([_task("simple", 1, 1, 1)])
            await marathon._worker
            return marathon.get(submitted["job_id"])

        job = asyncio.run(go())
        stage = job["results"][0]["stages"][0]
        assert stage["status"] == "failed"
        assert stage["plan"] == []
        assert stage["verification"] == "not_executed"
        assert job["results"][0]["verification"] == "not_executed"

    def test_plan_comes_from_the_last_successful_stage(self) -> None:
        provider = _ScriptedProvider(
            {
                "gpt-6-luna": ("Triage.\nMARATHON_PLAN: done: triage\nMARATHON_STATUS: ESCALATE"),
                "gpt-6-sol": (
                    "Root cause.\nMARATHON_PLAN: done: triage; done: root cause\n"
                    "MARATHON_STATUS: SOLVED"
                ),
            }
        )
        job = _run(provider, [_task("simple", 1, 1, 1)])

        assert provider.models == ["gpt-6-luna", "gpt-6-sol"]
        assert job["results"][0]["plan"] == [
            {"description": "triage", "status": "done"},
            {"description": "root cause", "status": "done"},
        ]
