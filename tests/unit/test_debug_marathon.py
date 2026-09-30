from __future__ import annotations

from mcp_for_copilot.debug_marathon import DebugMarathon, _estimate_cost_usd, prioritize_tasks
from mcp_for_copilot.provider import ProviderResponse


def _task(title: str, priority: int, difficulty: int, complexity: int) -> dict:
    return {
        "title": title,
        "question": f"Investigate {title}",
        "priority": priority,
        "difficulty": difficulty,
        "complexity": complexity,
    }


def test_tasks_sort_by_priority_difficulty_complexity_and_stably() -> None:
    ordered = prioritize_tasks(
        [
            _task("medium", 4, 2, 5),
            _task("hard", 4, 5, 1),
            _task("urgent", 5, 1, 1),
            _task("complex", 4, 2, 5),
        ]
    )

    assert [task["title"] for task in ordered] == ["urgent", "hard", "medium", "complex"]
    assert [task["queue_position"] for task in ordered] == [1, 2, 3, 4]
    assert ordered[0]["priority_score"] == 511


def test_marathon_calls_luna_then_sol_then_astra() -> None:
    class FakeProvider:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, str]] = []

        async def complete(self, messages, *, model_id, reasoning_effort):
            self.calls.append((model_id, reasoning_effort, messages[-1]["content"]))
            return ProviderResponse(
                content=f"review from {model_id}\nMARATHON_STATUS: SOLVED",
                model=model_id,
            )

    async def run():
        provider = FakeProvider()
        marathon = DebugMarathon(provider)  # type: ignore[arg-type]
        submitted = marathon.submit(
            [
                _task("simple", 1, 2, 2),
                _task("medium", 3, 3, 4),
                _task("hard", 5, 5, 5),
            ]
        )
        await marathon._worker
        return provider.calls, marathon.get(submitted["job_id"])

    import asyncio

    calls, job = asyncio.run(run())

    assert [model for model, _, _ in calls] == [
        "gpt-6-luna",
        "gpt-6-sol",
        "gpt-6-astra",
        "gpt-6-luna",
        "gpt-6-sol",
        "gpt-6-luna",
    ]
    assert [effort for _, effort, _ in calls] == ["low", "high", "high", "low", "high", "low"]
    assert "review from gpt-6-luna" in calls[1][2]
    assert "review from gpt-6-sol" in calls[2][2]
    assert "review from gpt-6-luna" in calls[4][2]
    assert job["status"] == "completed"
    assert [len(result["models_used"]) for result in job["results"]] == [3, 2, 1]
    assert job["results"][0]["final_answer"] == "review from gpt-6-astra"


def test_uncertain_luna_escalates_even_for_a_simple_task() -> None:
    class FakeProvider:
        def __init__(self) -> None:
            self.models: list[str] = []

        async def complete(self, messages, *, model_id, reasoning_effort):
            self.models.append(model_id)
            decision = "ESCALATE" if model_id == "gpt-6-luna" else "SOLVED"
            return ProviderResponse(
                content=f"analysis from {model_id}\nMARATHON_STATUS: {decision}",
                model=model_id,
            )

    async def run():
        provider = FakeProvider()
        marathon = DebugMarathon(provider)  # type: ignore[arg-type]
        submitted = marathon.submit([_task("simple", 2, 1, 1)])
        await marathon._worker
        return provider.models, marathon.get(submitted["job_id"])

    import asyncio

    models, job = asyncio.run(run())

    assert models == ["gpt-6-luna", "gpt-6-sol"]
    assert job["results"][0]["status"] == "completed"


def test_rejects_invalid_scores_and_empty_batches() -> None:
    for tasks in ([], [_task("too urgent", 6, 1, 1)]):
        try:
            prioritize_tasks(tasks)
        except ValueError:
            continue
        raise AssertionError("invalid debug marathon input was accepted")


def test_estimated_cost_tracks_the_model_catalog_ratio() -> None:
    one_million_each = {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000}

    luna = _estimate_cost_usd("gpt-6-luna", one_million_each)
    sol = _estimate_cost_usd("gpt-6-sol", one_million_each)
    astra = _estimate_cost_usd("gpt-6-astra", one_million_each)

    assert (luna, sol, astra) == (0.6, 12.0, 60.0)
