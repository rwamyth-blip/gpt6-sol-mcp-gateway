"""Example 3 — custom approval policy.

Shows the three-gate pipeline in action: a tool that is allowlisted and
advertised still gets denied when the approver says no.

Run:
    export LLM_API_KEY=sk-...
    python examples/03_custom_approval.py
"""

from __future__ import annotations

import asyncio
import os
import sys

from mcp_for_copilot import ApprovalDecision, ApprovalRequest, Gateway, reset_settings_cache


def audit_approver(request: ApprovalRequest) -> ApprovalDecision:
    """Approve read-only tools, deny everything else, and log every decision."""
    read_only = {"gpt6_models", "gpt6_status", "gpt6_tools", "read_file", "list_directory"}
    approved = request.tool in read_only
    reason = "read-only tool" if approved else f"{request.tool} is not on the read-only list"
    print(f"[approval] round={request.round_index} tool={request.tool} -> {approved} ({reason})")
    return ApprovalDecision(approved=approved, reason=reason, approver="audit_approver")


async def main() -> int:
    os.environ["MCP_TRANSPORT"] = "stdio"
    os.environ["MCP_SERVER_URL"] = f'"{sys.executable}" -m mcp_for_copilot.gateway.server'
    os.environ["MCP_ALLOWED_TOOLS"] = "gpt6_models,gpt6_status"
    reset_settings_cache()

    async with Gateway(approver=audit_approver) as gateway:
        result = await gateway.chat(
            [{"role": "user", "content": "What is the current gateway status?"}]
        )

        if result.error:
            print(f"provider error: {result.error}", file=sys.stderr)
            return 1

        print("\nanswer:", result.content)
        for call in result.invocations:
            print(
                f"  {call.tool}: allowed={call.allowed} approved={call.approved} "
                f"executed={call.executed}"
            )
        return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
