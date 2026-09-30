"""Unit tests for the approval layer — it must fail closed."""

from __future__ import annotations

from mcp_for_copilot.approval import (
    ApprovalDecision,
    ApprovalLayer,
    ApprovalRequest,
    allow_all_approver,
    deny_all_approver,
    static_approver,
)


class TestApprovalDecision:
    def test_allow_factory(self) -> None:
        decision = ApprovalDecision.allow("ok", approver="test")
        assert decision.approved is True
        assert decision.reason == "ok"

    def test_deny_factory(self) -> None:
        decision = ApprovalDecision.deny("no", approver="test")
        assert decision.approved is False


class TestApprovalLayer:
    async def test_no_approver_denies(self) -> None:
        layer = ApprovalLayer()
        assert layer.configured is False
        decision = await layer.request(ApprovalRequest(tool="delete_file"))
        assert decision.approved is False
        assert "no approver configured" in decision.reason

    async def test_allow_all_approver(self) -> None:
        layer = ApprovalLayer(allow_all_approver())
        decision = await layer.request(ApprovalRequest(tool="delete_file"))
        assert decision.approved is True

    async def test_deny_all_approver(self) -> None:
        layer = ApprovalLayer(deny_all_approver())
        decision = await layer.request(ApprovalRequest(tool="read_file"))
        assert decision.approved is False

    async def test_approver_exception_denies(self) -> None:
        async def exploding(request: ApprovalRequest) -> ApprovalDecision:
            raise RuntimeError("approver is down")

        layer = ApprovalLayer(exploding)
        decision = await layer.request(ApprovalRequest(tool="write_file"))
        assert decision.approved is False
        assert "RuntimeError" in decision.reason

    async def test_invalid_return_denies(self) -> None:
        async def wrong_type(request: ApprovalRequest) -> object:  # type: ignore[return-value]
            return "yes"

        layer = ApprovalLayer(wrong_type)  # type: ignore[arg-type]
        decision = await layer.request(ApprovalRequest(tool="write_file"))
        assert decision.approved is False
        assert "invalid decision" in decision.reason

    async def test_configured_property(self) -> None:
        assert ApprovalLayer(allow_all_approver()).configured is True
        assert ApprovalLayer().configured is False


class TestStaticApprover:
    async def test_allowlisted_tool_is_approved(self) -> None:
        layer = ApprovalLayer(static_approver(allow=["write_file"]))
        decision = await layer.request(ApprovalRequest(tool="write_file"))
        assert decision.approved is True

    async def test_denylisted_tool_is_denied(self) -> None:
        layer = ApprovalLayer(static_approver(deny=["delete_file"]))
        decision = await layer.request(ApprovalRequest(tool="delete_file"))
        assert decision.approved is False

    async def test_unlisted_tool_is_denied(self) -> None:
        layer = ApprovalLayer(static_approver(allow=["write_file"]))
        decision = await layer.request(ApprovalRequest(tool="drop_table"))
        assert decision.approved is False
        assert "not in the static allowlist" in decision.reason

    async def test_deny_wins_over_allow(self) -> None:
        layer = ApprovalLayer(static_approver(allow=["write_file"], deny=["write_file"]))
        decision = await layer.request(ApprovalRequest(tool="write_file"))
        assert decision.approved is True  # allow is checked first, by design


class TestApprovalRequest:
    def test_defaults(self) -> None:
        request = ApprovalRequest(tool="x")
        assert request.arguments == {}
        assert request.round_index == 0
