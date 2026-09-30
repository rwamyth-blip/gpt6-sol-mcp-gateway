"""mcp-for-copilot — an MCP gateway for GPT-6 Sol and friends.

Two ways to use this package:

**As a library** — wire the layers yourself::

    from mcp_for_copilot import Gateway

    gw = Gateway(api_key="sk-...", model="gpt-6-sol")
    result = await gw.chat("Explain MCP in one paragraph.")

**As a server** — run the OpenAI-compatible FastAPI app::

    mcp-for-copilot serve --port 8000

Layered architecture (each layer is independently testable):

===================  ==================================================
``config``           Configuration — env-driven, no hardcoded secrets
``logging_utils``    Logging — redacts secrets before they are emitted
``provider``         LLM provider adapter — the only place that calls a model
``mcp_client``       MCP client — stdio / http / sse transports
``tool_router``      Tool router — allowlist + argument validation
``approval``         Approval layer — gates risky tools
``orchestrator``     Orchestrator — wires the layers, caps tool rounds
``gateway``          High-level facade + OpenAI-compatible server
===================  ==================================================
"""

from __future__ import annotations

from .approval import (
    ApprovalDecision,
    ApprovalLayer,
    ApprovalRequest,
    allow_all_approver,
    deny_all_approver,
    static_approver,
)
from .config import Settings, get_settings
from .gateway import Gateway, GatewayResult
from .logging_utils import contains_secret, redact, safe_log
from .mcp_client import MCPCallResult, MCPClient, MCPClientError, MCPClientManager, MCPTool
from .orchestrator import LLMMCPOrchestrator, OrchestratorResult, ToolInvocation
from .provider import (
    KNOWN_MODELS,
    MODEL_ALIASES,
    LLMProvider,
    LLMProviderError,
    ProviderResponse,
    ToolCall,
    resolve_model_id,
)
from .tool_router import (
    DEFAULT_ALLOWED_TOOLS,
    RoutedTool,
    RouterDecision,
    ToolRouter,
    ToolRouterError,
    is_risky_tool,
    validate_arguments,
)

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_ALLOWED_TOOLS",
    "KNOWN_MODELS",
    "MODEL_ALIASES",
    "ApprovalDecision",
    "ApprovalLayer",
    "ApprovalRequest",
    "Gateway",
    "GatewayResult",
    "LLMMCPOrchestrator",
    "LLMProvider",
    "LLMProviderError",
    "MCPCallResult",
    "MCPClient",
    "MCPClientError",
    "MCPClientManager",
    "MCPTool",
    "OrchestratorResult",
    "ProviderResponse",
    "RoutedTool",
    "RouterDecision",
    "Settings",
    "ToolCall",
    "ToolInvocation",
    "ToolRouter",
    "ToolRouterError",
    "__version__",
    "allow_all_approver",
    "contains_secret",
    "deny_all_approver",
    "get_settings",
    "is_risky_tool",
    "redact",
    "resolve_model_id",
    "safe_log",
    "static_approver",
    "validate_arguments",
]
