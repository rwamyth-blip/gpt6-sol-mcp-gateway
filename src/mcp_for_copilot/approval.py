"""Approval layer — the human-in-the-loop gate for risky tools.

The orchestrator asks this layer before dispatching any tool the router marked
as needing approval. The layer **fails closed**: if no approver is configured,
every request is denied rather than silently allowed.

Three approvers ship with the package:

``allow_all_approver``
    Approves everything. For tests and trusted local runs only.
``deny_all_approver``
    Denies everything. The default when nothing is configured.
``static_approver``
    Approves or denies a fixed set of tool names.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from .logging_utils import log_info, log_warning, redact


@dataclass
class ApprovalRequest:
    """What the orchestrator asks the approver to decide on."""

    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    round_index: int = 0


@dataclass
class ApprovalDecision:
    """The approver's verdict."""

    approved: bool
    reason: str = ""
    approver: str = ""

    @classmethod
    def allow(cls, reason: str = "approved", approver: str = "") -> ApprovalDecision:
        return cls(True, reason, approver)

    @classmethod
    def deny(cls, reason: str = "denied", approver: str = "") -> ApprovalDecision:
        return cls(False, reason, approver)


Approver = Callable[[ApprovalRequest], Awaitable[ApprovalDecision]]


class ApprovalLayer:
    """Wraps an approver callable and enforces fail-closed behaviour."""

    def __init__(self, approver: Approver | None = None) -> None:
        self._approver = approver

    @property
    def configured(self) -> bool:
        return self._approver is not None

    async def request(self, request: ApprovalRequest) -> ApprovalDecision:
        """Ask the approver. Denies when no approver is configured."""
        if self._approver is None:
            log_warning(
                "approval denied for %s: no approver configured (fail closed)",
                redact(request.tool),
            )
            return ApprovalDecision.deny(
                "no approver configured; risky tools are denied by default",
                approver="none",
            )

        try:
            decision = await self._approver(request)
        except Exception as exc:
            log_warning("approver raised for %s: %s", redact(request.tool), type(exc).__name__)
            return ApprovalDecision.deny(f"approver raised {type(exc).__name__}", approver="error")

        if not isinstance(decision, ApprovalDecision):
            return ApprovalDecision.deny(
                "approver returned an invalid decision", approver="invalid"
            )

        log_info(
            "approval %s for %s by %s",
            "granted" if decision.approved else "denied",
            redact(request.tool),
            redact(decision.approver or "unknown"),
        )
        return decision


def allow_all_approver() -> Approver:
    """Approve every request. Tests and trusted local runs only."""

    async def _approve(request: ApprovalRequest) -> ApprovalDecision:
        return ApprovalDecision.allow("allow_all approver", approver="allow_all")

    return _approve


def deny_all_approver() -> Approver:
    """Deny every request."""

    async def _deny(request: ApprovalRequest) -> ApprovalDecision:
        return ApprovalDecision.deny("deny_all approver", approver="deny_all")

    return _deny


def static_approver(
    *, allow: tuple[str, ...] | list[str] = (), deny: tuple[str, ...] | list[str] = ()
) -> Approver:
    """Approve tools in *allow*, deny tools in *deny*, deny anything else."""
    allowed = set(allow)
    denied = set(deny)

    async def _decide(request: ApprovalRequest) -> ApprovalDecision:
        if request.tool in allowed:
            return ApprovalDecision.allow("static allowlist", approver="static")
        if request.tool in denied:
            return ApprovalDecision.deny("static denylist", approver="static")
        return ApprovalDecision.deny("tool not in the static allowlist", approver="static")

    return _decide
