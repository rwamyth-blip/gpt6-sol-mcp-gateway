"""Prioritized, sequential debug runs through the GPT-6 model ladder."""

from __future__ import annotations

import asyncio
import copy
import re
from contextlib import suppress
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from .logging_utils import redact
from .provider import KNOWN_MODELS, LLMProvider

MODEL_STAGES: tuple[tuple[str, str, str], ...] = (
    ("gpt-6-luna", "low", "triage the issue and produce a diagnostic plan"),
    ("gpt-6-sol", "high", "identify the root cause and propose a minimal safe fix"),
    ("gpt-6-astra", "high", "review the diagnosis and fix for gaps and regressions"),
)
MAX_TASKS_PER_JOB = 10
MAX_RETAINED_JOBS = 100
TARGET_TASK_MIX = {"luna_only": 0.80, "luna_then_sol": 0.15, "full_ladder": 0.05}

# The marathon never executes code, so every stage must say so explicitly. The
# marker is parsed out of the answer and surfaced as ``verification`` on the
# task result, replacing the old blanket "model_self_reported" string.
VERIFICATION_MARKER = "MARATHON_VERIFICATION"
VERIFICATION_VALUES = ("not_executed", "executed")
_DEFAULT_VERIFICATION = "not_executed"

# A stage may publish a short plan. It is advisory: it is recorded for the
# audit trail and never gates anything, because the marathon runs no tools.
PLAN_MARKER = "MARATHON_PLAN"
MAX_PLAN_STEPS = 6
PLAN_STATUSES = ("pending", "in_progress", "done", "blocked")

_VERIFICATION_RE = re.compile(rf"(?im)^\s*{VERIFICATION_MARKER}\s*:\s*([A-Za-z_]+)\s*$")
_PLAN_RE = re.compile(rf"(?im)^\s*{PLAN_MARKER}\s*:\s*(.+?)\s*$")


def _parse_verification(raw: str) -> str:
    """Read the stage's self-declared verification state.

    Anything unrecognised falls back to ``not_executed``: the safe default is
    to assume nothing ran, never to assume something did.
    """
    match = _VERIFICATION_RE.search(raw)
    if not match:
        return _DEFAULT_VERIFICATION
    value = match.group(1).strip().lower()
    return value if value in VERIFICATION_VALUES else _DEFAULT_VERIFICATION


def _parse_plan(raw: str) -> list[dict[str, str]]:
    """Read a stage's advisory plan from a single ``MARATHON_PLAN:`` line.

    The line is a semicolon-separated list of ``status: description`` items,
    e.g. ``MARATHON_PLAN: done: read the log; in_progress: patch the parser``.
    Items without a recognised status prefix are treated as ``pending``.
    """
    match = _PLAN_RE.search(raw)
    if not match:
        return []
    steps: list[dict[str, str]] = []
    for chunk in match.group(1).split(";"):
        item = chunk.strip()
        if not item:
            continue
        status, _, description = item.partition(":")
        status = status.strip().lower()
        if status in PLAN_STATUSES and description.strip():
            description = description.strip()
        else:
            status, description = "pending", item
        steps.append({"description": description, "status": status})
        if len(steps) >= MAX_PLAN_STEPS:
            break
    return steps


def _strip_markers(raw: str) -> str:
    """Remove the machine-readable marker lines from a stage answer."""
    cleaned = _VERIFICATION_RE.sub("", raw)
    cleaned = _PLAN_RE.sub("", cleaned)
    return cleaned.strip()


def _task_verification(stages: list[dict[str, Any]]) -> str:
    """Summarise the task's verification state from its stages.

    The marathon never runs code, so the honest answer is almost always
    ``not_executed``. A stage only upgrades the task to ``executed`` when it
    explicitly declared that it was handed real execution output.
    """
    if any(stage.get("verification") == "executed" for stage in stages):
        return "executed"
    return _DEFAULT_VERIFICATION


def prioritize_tasks(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate and stably order tasks by priority, difficulty, and complexity."""
    if not 1 <= len(tasks) <= MAX_TASKS_PER_JOB:
        raise ValueError(f"tasks must contain 1 to {MAX_TASKS_PER_JOB} items")

    normalized: list[dict[str, Any]] = []
    for task in tasks:
        if not isinstance(task, dict):
            raise ValueError("each task must be an object")
        question = str(task.get("question") or "").strip()
        if not question or len(question) > 10000:
            raise ValueError("each task needs a question of 1 to 10000 characters")
        item: dict[str, Any] = {"question": question}
        title = str(task.get("title") or "").strip()
        if title:
            if len(title) > 200:
                raise ValueError("task title must be 200 characters or fewer")
            item["title"] = title
        for field in ("priority", "difficulty", "complexity"):
            score = task.get(field, 3)
            if isinstance(score, bool) or not isinstance(score, int):
                raise ValueError(f"{field} must be an integer from 1 to 5")
            if not 1 <= score <= 5:
                raise ValueError(f"{field} must be an integer from 1 to 5")
            item[field] = score
        normalized.append(item)

    ranked = sorted(
        enumerate(normalized),
        key=lambda pair: (
            -pair[1]["priority"],
            -pair[1]["difficulty"],
            -pair[1]["complexity"],
            pair[0],
        ),
    )
    return [
        {
            **task,
            "task_id": str(uuid4()),
            "queue_position": position,
            "priority_score": (
                task["priority"] * 100 + task["difficulty"] * 10 + task["complexity"]
            ),
        }
        for position, (_, task) in enumerate(ranked, start=1)
    ]


def _required_stages(task: dict[str, Any]) -> int:
    level = max(task["difficulty"], task["complexity"])
    return 1 if level <= 2 else 2 if level <= 4 else 3


def _token_count(usage: dict[str, Any], key: str) -> int:
    try:
        return max(0, int(usage.get(key) or 0))
    except (TypeError, ValueError):
        return 0


def _estimate_cost_usd(model: str, usage: dict[str, Any]) -> float:
    spec = KNOWN_MODELS[model]
    prompt_tokens = _token_count(usage, "prompt_tokens")
    completion_tokens = _token_count(usage, "completion_tokens")
    return (
        float(
            prompt_tokens * spec["input_price_per_mtok"]
            + completion_tokens * spec["output_price_per_mtok"]
        )
        / 1_000_000
    )


class DebugMarathon:
    """Own an in-process FIFO job queue and execute tasks one at a time."""

    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider
        self._jobs: dict[str, dict[str, Any]] = {}
        self._queue: asyncio.Queue[tuple[str, list[dict[str, Any]]]] = asyncio.Queue()
        self._worker: asyncio.Task[None] | None = None

    def submit(self, tasks: list[dict[str, Any]]) -> dict[str, Any]:
        ordered = prioritize_tasks(tasks)
        self._prune_jobs()
        job_id = str(uuid4())
        self._jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "created_at": _now(),
            "started_at": None,
            "finished_at": None,
            "total": len(ordered),
            "processed": 0,
            "completed": 0,
            "current_task_id": None,
            "current_model": None,
            "model_order": [model for model, _, _ in MODEL_STAGES],
            "target_task_mix": dict(TARGET_TASK_MIX),
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "estimated_cost_usd": 0.0,
            "model_reported_success_rate": None,
            "verification": _DEFAULT_VERIFICATION,
            "queue": [
                {
                    key: task[key]
                    for key in (
                        "task_id",
                        "queue_position",
                        "title",
                        "priority",
                        "difficulty",
                        "complexity",
                        "priority_score",
                    )
                    if key in task
                }
                for task in ordered
            ],
            "results": [],
        }
        self._queue.put_nowait((job_id, ordered))
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._run_queue())
        return self.get(job_id) or {}

    def get(self, job_id: str) -> dict[str, Any] | None:
        job = self._jobs.get(job_id)
        if not job:
            return None
        result = copy.deepcopy(job)
        evaluated = result["results"]
        result["model_reported_evaluated_tasks"] = len(evaluated)
        result["model_reported_success_rate"] = (
            sum(item["status"] == "completed" for item in evaluated) / len(evaluated)
            if evaluated
            else None
        )
        return result

    async def stop(self) -> None:
        worker = self._worker
        if worker is not None and not worker.done():
            worker.cancel()
            with suppress(asyncio.CancelledError):
                await worker
        while not self._queue.empty():
            self._queue.get_nowait()
            self._queue.task_done()
        for job in self._jobs.values():
            if job["status"] in {"queued", "running"}:
                job["status"] = "interrupted"
                job["current_task_id"] = None
                job["current_model"] = None
                job["finished_at"] = _now()
        self._worker = None

    def _prune_jobs(self) -> None:
        if len(self._jobs) < MAX_RETAINED_JOBS:
            return
        terminal = {"completed", "partial", "failed", "needs_review", "interrupted"}
        for job_id, job in list(self._jobs.items()):
            if job["status"] in terminal:
                del self._jobs[job_id]
                if len(self._jobs) < MAX_RETAINED_JOBS:
                    break

    async def _run_queue(self) -> None:
        while not self._queue.empty():
            job_id, tasks = self._queue.get_nowait()
            try:
                await self._run_job(job_id, tasks)
            except Exception:
                job = self._jobs[job_id]
                job["status"] = "failed"
                job["error"] = "marathon_worker_failed"
                job["current_task_id"] = None
                job["current_model"] = None
                job["finished_at"] = _now()
            finally:
                self._queue.task_done()
        self._worker = None

    async def _run_job(self, job_id: str, tasks: list[dict[str, Any]]) -> None:
        job = self._jobs[job_id]
        job["status"] = "running"
        job["started_at"] = _now()
        for task in tasks:
            job["current_task_id"] = task["task_id"]
            result = await self._run_task(task, job)
            job["results"].append(result)
            job["processed"] += 1
            if result["status"] == "completed":
                job["completed"] += 1
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                job["usage"][key] += result["usage"][key]
            job["estimated_cost_usd"] += result["estimated_cost_usd"]
            job["estimated_cost_usd"] = round(job["estimated_cost_usd"], 8)
        statuses = [result["status"] for result in job["results"]]
        job["status"] = (
            "completed"
            if all(status == "completed" for status in statuses)
            else "partial"
            if any(status != "failed" for status in statuses)
            else "failed"
        )
        # The marathon never executes code, so the job-level verification is
        # only ever upgraded when a stage explicitly reported real output.
        job["verification"] = (
            "executed"
            if any(result["verification"] == "executed" for result in job["results"])
            else _DEFAULT_VERIFICATION
        )
        job["current_task_id"] = None
        job["current_model"] = None
        job["finished_at"] = _now()

    async def _run_task(self, task: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
        context = (
            f"Debug issue: {task.get('title') or task['question']}\n"
            f"Details: {task['question']}\n"
            f"Priority: {task['priority']}/5; difficulty: {task['difficulty']}/5; "
            f"complexity: {task['complexity']}/5."
        )
        stages: list[dict[str, Any]] = []
        previous = ""
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        estimated_cost = 0.0
        required_stages = _required_stages(task)
        for index, (model, effort, role) in enumerate(MODEL_STAGES):
            job["current_model"] = model
            user_content = context
            if previous:
                user_content += f"\n\nPrevious stage output:\n{previous[:12000]}"
            try:
                response = await self.provider.complete(
                    [
                        {
                            "role": "system",
                            "content": (
                                f"You are the {model} stage in a debugging review. "
                                f"Your role is to {role}. Use only supplied facts. "
                                "Do not claim to have executed code or tests. End with "
                                "exactly one line: MARATHON_STATUS: SOLVED if the "
                                "diagnosis is sufficiently supported, otherwise "
                                "MARATHON_STATUS: ESCALATE. Then add exactly one line "
                                f"{VERIFICATION_MARKER}: not_executed unless you were "
                                "actually given execution output, in which case use "
                                f"{VERIFICATION_MARKER}: executed. Optionally add one "
                                f"line {PLAN_MARKER}: <status>: <step>; <status>: "
                                "<step> using the statuses pending, in_progress, done "
                                "or blocked."
                            ),
                        },
                        {"role": "user", "content": user_content},
                    ],
                    model_id=model,
                    reasoning_effort=effort,
                )
                raw_answer = (response.content or "").strip()
                match = re.search(
                    r"(?im)^\s*MARATHON_STATUS\s*:\s*(SOLVED|ESCALATE)\s*$",
                    raw_answer,
                )
                verification = _parse_verification(raw_answer)
                plan = _parse_plan(raw_answer)
                answer = _strip_markers(
                    re.sub(
                        r"(?im)^\s*MARATHON_STATUS\s*:\s*(?:SOLVED|ESCALATE)\s*$",
                        "",
                        raw_answer,
                    )
                )
                ok = bool(answer)
                decision = match.group(1).lower() if match else "uncertain"
                usage = response.usage or {}
                stage_usage = {
                    "prompt_tokens": _token_count(usage, "prompt_tokens"),
                    "completion_tokens": _token_count(usage, "completion_tokens"),
                    "total_tokens": _token_count(usage, "total_tokens"),
                }
                if not stage_usage["total_tokens"]:
                    stage_usage["total_tokens"] = (
                        stage_usage["prompt_tokens"] + stage_usage["completion_tokens"]
                    )
                stage_cost = _estimate_cost_usd(model, stage_usage)
                for key in total_usage:
                    total_usage[key] += stage_usage[key]
                estimated_cost += stage_cost
                stages.append(
                    {
                        "model": model,
                        "status": "completed" if ok else "failed",
                        "decision": decision,
                        "answer": answer or None,
                        "error": None if ok else "empty_response",
                        "verification": verification,
                        "plan": plan,
                        "usage": stage_usage,
                        "estimated_cost_usd": round(stage_cost, 8),
                    }
                )
                if ok:
                    previous = answer
                if ok and index + 1 >= required_stages and decision == "solved":
                    break
            except Exception as exc:
                stages.append(
                    {
                        "model": model,
                        "status": "failed",
                        "decision": "uncertain",
                        "answer": None,
                        "error": redact(str(exc)) or type(exc).__name__,
                        "verification": _DEFAULT_VERIFICATION,
                        "plan": [],
                        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                        "estimated_cost_usd": 0.0,
                    }
                )
        successes = [stage for stage in stages if stage["status"] == "completed"]
        resolved = bool(successes and successes[-1]["decision"] == "solved")
        status = "completed" if resolved else "needs_review" if successes else "failed"
        return {
            "task_id": task["task_id"],
            "queue_position": task["queue_position"],
            "title": task.get("title"),
            "priority": task["priority"],
            "difficulty": task["difficulty"],
            "complexity": task["complexity"],
            "priority_score": task["priority_score"],
            "status": status,
            "verification": _task_verification(stages),
            "plan": successes[-1]["plan"] if successes else [],
            "models_used": [stage["model"] for stage in stages],
            "usage": total_usage,
            "estimated_cost_usd": round(estimated_cost, 8),
            "stages": stages,
            "final_answer": successes[-1]["answer"] if successes else None,
        }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
