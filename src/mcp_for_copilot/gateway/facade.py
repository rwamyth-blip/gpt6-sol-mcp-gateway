"""Gateway facade — one object that owns the whole stack.

:class:`Gateway` wires the provider, the MCP client, the router, the approval
layer and the orchestrator together, and manages the MCP connection lifetime.

Typical use::

    async with Gateway() as gateway:
        result = await gateway.chat([{"role": "user", "content": "List the files"}])
        print(result.content)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..approval import ApprovalLayer, Approver
from ..config import Settings, get_settings
from ..debug_marathon import DebugMarathon
from ..logging_utils import log_info, log_warning, redact
from ..mcp_client import (
    MCPClient,
    MCPClientError,
    MCPClientManager,
    default_stdio_command,
)
from ..orchestrator import LLMMCPOrchestrator, OrchestratorResult
from ..provider import LLMProvider
from ..tool_router import ToolRouter


@dataclass
class GatewayResult:
    """A chat completion plus the tool audit trail."""

    content: str = ""
    model: str = ""
    rounds: int = 0
    used_tools: list[str] = field(default_factory=list)
    invocations: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    plan: list[dict[str, str]] = field(default_factory=list)
    plan_progress: dict[str, int] = field(default_factory=dict)
    loop_stopped: bool = False

    @classmethod
    def from_orchestrator(cls, result: OrchestratorResult) -> GatewayResult:
        return cls(
            content=result.content,
            model=result.model,
            rounds=result.rounds,
            used_tools=result.used_tools,
            invocations=[
                {
                    "tool": i.tool,
                    "arguments": i.arguments,
                    "allowed": i.allowed,
                    "approved": i.approved,
                    "executed": i.executed,
                    "is_error": i.is_error,
                    "reason": i.reason,
                }
                for i in result.invocations
            ],
            usage=result.usage,
            error=result.error,
            plan=[step.as_dict() for step in result.plan],
            plan_progress=result.plan_progress,
            loop_stopped=result.loop_stopped,
        )


class Gateway:
    """Owns the provider + MCP client + orchestrator for one process."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        provider: LLMProvider | None = None,
        approver: Approver | None = None,
        connect_mcp: bool = True,
    ) -> None:
        self.settings = settings or get_settings()
        self.provider = provider or LLMProvider()
        self.debug_marathon = DebugMarathon(self.provider)
        self._approver = approver
        self._connect_mcp = connect_mcp

        # ``client`` is either a single :class:`MCPClient` (deprecated
        # ``MCP_SERVER_URL`` path) or an :class:`MCPClientManager` when
        # ``MCP_SERVERS`` lists more than one server. The orchestrator already
        # branches on the type, so both are valid here.
        self.client: MCPClient | MCPClientManager | None = None
        self.router = ToolRouter(
            allowed_tools=self.settings.mcp_allowed_tools or None,
            require_approval=self.settings.mcp_require_approval,
        )
        self.approval = ApprovalLayer(approver)
        self.orchestrator = LLMMCPOrchestrator(
            provider=self.provider,
            client=None,
            router=self.router,
            approval=self.approval,
            settings=self.settings,
        )
        self._connected = False

    # -- lifecycle --------------------------------------------------------
    async def __aenter__(self) -> Gateway:
        await self.start()
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.stop()

    async def start(self) -> None:
        """Connect to the MCP server(s), if any are configured.

        ``MCP_SERVERS`` (a JSON array) takes precedence and yields an
        :class:`MCPClientManager`. When it is unset, the deprecated single
        ``MCP_SERVER_URL`` fields are used exactly as before, so existing
        deployments keep working unchanged.
        """
        if self._connected or not self._connect_mcp:
            return

        if self.settings.mcp_servers_parsed:
            self.client = await self._connect_manager()
        else:
            self.client = await self._connect_single()

        self.orchestrator.client = self.client
        self._connected = True

    async def _connect_manager(self) -> MCPClientManager | None:
        """Connect every server listed in ``MCP_SERVERS``.

        A server that fails to connect is dropped rather than aborting the
        whole gateway: the remaining servers still provide their tools.
        """
        manager = MCPClientManager.from_settings(self.settings)
        connected = 0
        for name, client in list(manager._clients.items()):
            try:
                await client.connect()
                connected += 1
            except MCPClientError as exc:
                log_warning("MCP server %s failed to connect: %s", redact(name), redact(str(exc)))
                manager._clients.pop(name, None)
        if connected == 0:
            log_warning("no MCP server from MCP_SERVERS could be reached; running without tools")
            return None
        log_info("mcp manager connected servers=%d", connected)
        return manager

    async def _connect_single(self) -> MCPClient | None:
        """Connect the single server described by the deprecated fields."""
        if not self.settings.mcp_configured and self.settings.mcp_transport != "stdio":
            log_warning("MCP_SERVER_URL is not set; running without tools")
            return None

        command = self.settings.mcp_server_url or default_stdio_command()
        client = MCPClient(
            url=command,
            transport=self.settings.mcp_transport,
            auth_token=self.settings.mcp_auth_token,
            timeout=float(self.settings.mcp_timeout_seconds),
        )
        try:
            await client.connect()
        except MCPClientError as exc:
            log_warning("MCP connection failed: %s", redact(str(exc)))
            return None
        return client

    async def stop(self) -> None:
        await self.debug_marathon.stop()
        if self.client is not None:
            await self.client.close()
            self.client = None
        self.orchestrator.client = None
        self._connected = False

    # -- public API -------------------------------------------------------
    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        system: str | None = None,
        model_id: str | None = None,
        max_rounds: int | None = None,
    ) -> GatewayResult:
        """Run one chat turn through the full tool loop."""
        if not self._connected:
            await self.start()
        result = await self.orchestrator.run(
            messages, system=system, model_id=model_id, max_rounds=max_rounds
        )
        return GatewayResult.from_orchestrator(result)

    async def list_tools(self) -> list[dict[str, Any]]:
        """Return the allowed tools as OpenAI function schemas."""
        if not self._connected:
            await self.start()
        if self.client is None:
            return []
        try:
            if isinstance(self.client, MCPClientManager):
                listed = await self.client.list_all_tools()
                tools = [tool for tools in listed.values() for tool in tools]
            else:
                tools = await self.client.list_tools()
        except MCPClientError as exc:
            log_warning("could not list tools: %s", redact(str(exc)))
            return []
        self.router.register_all(tools)
        return LLMProvider.to_openai_tools(self.router.available())

    def status(self) -> dict[str, Any]:
        """Non-secret status snapshot, safe to return over HTTP."""
        if isinstance(self.client, MCPClientManager):
            servers = sorted(self.client._clients)
            server_info: dict[str, Any] = {"servers": servers}
            transport = "multi"
        else:
            servers = []
            server_info = self.client.server_info if self.client else {}
            transport = self.settings.mcp_transport
        return {
            "llm": self.provider.describe(),
            "mcp": {
                "transport": transport,
                "configured": self.settings.mcp_configured,
                "connected": self.client is not None,
                "server_info": server_info,
                "servers": servers,
                "allowed_tools": list(self.router.allowed_tools),
                "require_approval": self.settings.mcp_require_approval,
                "approver_configured": self.approval.configured,
            },
        }

    def log_summary(self) -> None:
        info = self.status()
        log_info(
            "gateway ready model=%s mcp_connected=%s tools=%d",
            info["llm"]["model_id"],
            info["mcp"]["connected"],
            len(info["mcp"]["allowed_tools"]),
        )
