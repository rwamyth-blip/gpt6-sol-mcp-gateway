"""Tool router — the allowlist and argument-validation gate.

Every tool call the model requests passes through here before it can reach the
MCP server. The router answers three questions:

1. **Is this tool allowed at all?** (allowlist)
2. **Is it risky?** (needs an approval decision)
3. **Are the arguments well-formed?** (JSON-schema subset validation)

The router **fails closed**: an unknown tool is denied, and a tool whose name
matches a risky verb is treated as risky even if it is not on the explicit
list.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .logging_utils import log_debug, log_warning, redact

# Verbs that indicate a mutating or destructive operation. Matched as whole
# words against the tool name, so "read_file" is safe but "delete_file" is not.
RISKY_TOOL_PATTERNS: tuple[str, ...] = (
    "write",
    "create",
    "update",
    "delete",
    "remove",
    "drop",
    "truncate",
    "insert",
    "upsert",
    "replace",
    "patch",
    "put",
    "post",
    "send",
    "publish",
    "deploy",
    "push",
    "commit",
    "merge",
    "rebase",
    "reset",
    "revert",
    "execute",
    "exec",
    "run",
    "spawn",
    "kill",
    "stop",
    "start",
    "restart",
    "install",
    "uninstall",
    "upgrade",
    "migrate",
    "grant",
    "revoke",
    "chmod",
    "chown",
    "sudo",
    "shell",
    "eval",
)

# Tools that are risky regardless of their name.
EXPLICIT_RISKY_TOOLS: frozenset[str] = frozenset({"codex_run", "mongo_aggregate"})

# Tools that are safe regardless of their name.
EXPLICIT_SAFE_TOOLS: frozenset[str] = frozenset(
    {
        "read_file",
        "list_files",
        "list_directory",
        "search_files",
        "grep",
        "glob",
        "get_file_info",
        "stat",
        "head",
        "tail",
        "mongo_find",
        "mongo_count",
        "mongo_list_collections",
        "mongo_list_databases",
        "mongo_indexes",
        "mongo_stats",
        "mongo_ping",
    }
)

# Tools that must never be served to the model through this gateway because
# they call the gateway itself: ``gateway_chat`` is the HTTP bridge
# (mcp_bridge.py) and the ``gpt6_*`` tools are the in-process stdio server
# (gateway/server.py). Both are front-ends for this same orchestrator, so
# registering them as back-end tools would create a model -> gateway ->
# gateway -> model loop. Even if an MCP server advertises them (e.g. a
# misconfigured aggregate server), the router refuses to register them.
SELF_LOOP_TOOL_PREFIXES: tuple[str, ...] = ("gpt6_", "gateway_")

# Read-only tools enabled when MCP_ALLOWED_TOOLS is unset.
DEFAULT_ALLOWED_TOOLS: tuple[str, ...] = (
    "read_file",
    "list_files",
    "list_directory",
    "search_files",
    "grep",
    "glob",
    "get_file_info",
    "stat",
    "mongo_find",
    "mongo_count",
    "mongo_list_collections",
    "mongo_stats",
)

_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "object": (dict,),
    "array": (list,),
    "null": (type(None),),
}


class ToolRouterError(RuntimeError):
    """Raised when a tool call is rejected."""


@dataclass
class RoutedTool:
    """A tool that passed the allowlist, with its declared schema."""

    name: str
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)
    risky: bool = False


@dataclass
class RouterDecision:
    """The router's verdict for one requested tool call."""

    allowed: bool
    tool: str
    reason: str = ""
    needs_approval: bool = False
    arguments: dict[str, Any] = field(default_factory=dict)


def is_self_loop_tool(name: str) -> bool:
    """True when *name* is a front-end for this same gateway (never servable).

    ``gateway_chat`` (the HTTP bridge) and the ``gpt6_*`` tools (the bundled
    stdio server) both call back into this orchestrator, so serving them to
    the model would loop: model -> gateway -> gateway -> model.
    """
    lowered = name.lower()
    return any(lowered.startswith(prefix) for prefix in SELF_LOOP_TOOL_PREFIXES)


def is_risky_tool(name: str) -> bool:
    """Return True when *name* looks like a mutating operation.

    Fails closed: an empty or unparseable name is treated as risky.
    """
    if not name:
        return True
    lowered = name.strip().lower()
    if lowered in EXPLICIT_SAFE_TOOLS:
        return False
    if lowered in EXPLICIT_RISKY_TOOLS:
        return True
    words = set(re.split(r"[^a-z0-9]+", lowered))
    return any(pattern in words for pattern in RISKY_TOOL_PATTERNS)


def validate_arguments(schema: dict[str, Any], arguments: dict[str, Any]) -> list[str]:
    """Validate *arguments* against a JSON-schema subset.

    Returns a list of human-readable problems; an empty list means valid.
    Supports ``type``, ``required``, ``properties``, ``enum`` and ``items``.
    Unknown keywords are ignored rather than rejected, so a richer server
    schema still works.
    """
    problems: list[str] = []
    if not isinstance(schema, dict) or not schema:
        return problems
    if not isinstance(arguments, dict):
        return ["arguments must be a JSON object"]

    expected = schema.get("type")
    if expected and expected != "object":
        problems.append(f"schema declares type {expected!r}, expected 'object'")

    for key in schema.get("required") or []:
        if key not in arguments:
            problems.append(f"missing required argument {key!r}")

    properties = schema.get("properties") or {}
    for key, value in arguments.items():
        if key not in properties:
            if schema.get("additionalProperties") is False:
                problems.append(f"unexpected argument {key!r}")
            continue
        problems.extend(_validate_value(key, value, properties[key] or {}))

    return problems


def _validate_value(key: str, value: Any, spec: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    expected = spec.get("type")
    if expected:
        types = expected if isinstance(expected, list) else [expected]
        # bool is a subclass of int in Python; check it explicitly so
        # True does not satisfy {"type": "integer"}.
        if isinstance(value, bool) and "boolean" not in types:
            problems.append(f"{key!r} must be {expected}, got boolean")
            return problems
        allowed: list[type] = []
        for name in types:
            allowed.extend(_JSON_TYPES.get(name, ()))
        if allowed and not isinstance(value, tuple(allowed)):
            problems.append(f"{key!r} must be {expected}, got {type(value).__name__}")
            return problems

    enum = spec.get("enum")
    if enum and value not in enum:
        problems.append(f"{key!r} must be one of {enum}, got {value!r}")

    if isinstance(value, list) and isinstance(spec.get("items"), dict):
        for index, item in enumerate(value):
            problems.extend(_validate_value(f"{key}[{index}]", item, spec["items"]))

    return problems


class ToolRouter:
    """Allowlist + validation gate in front of the MCP server."""

    def __init__(
        self,
        *,
        allowed_tools: tuple[str, ...] | list[str] | None = None,
        require_approval: bool = True,
    ) -> None:
        self.allowed_tools: tuple[str, ...] = tuple(
            DEFAULT_ALLOWED_TOOLS if allowed_tools is None else allowed_tools
        )
        self.require_approval = require_approval
        self._registry: dict[str, RoutedTool] = {}

    # -- registration -----------------------------------------------------
    def register(self, tool: Any) -> RoutedTool | None:
        """Register an MCP tool descriptor (anything with ``name``).

        Returns ``None`` and skips the tool when it is a gateway self-loop
        (``gpt6_*`` / ``gateway_*``): registering those would let the model
        call back into this same orchestrator.
        """
        name = str(getattr(tool, "name", "") or "")
        if not name:
            raise ToolRouterError("Cannot register a tool without a name")
        if is_self_loop_tool(name):
            log_warning("tool %s skipped: gateway self-loop tool is never served", redact(name))
            return None
        routed = RoutedTool(
            name=name,
            description=str(getattr(tool, "description", "") or ""),
            input_schema=getattr(tool, "input_schema", None) or {},
            risky=is_risky_tool(name),
        )
        self._registry[name] = routed
        return routed

    def register_all(self, tools: list[Any]) -> list[RoutedTool]:
        registered: list[RoutedTool] = []
        for tool in tools:
            routed = self.register(tool)
            if routed is not None:
                registered.append(routed)
        return registered

    @property
    def registered(self) -> dict[str, RoutedTool]:
        return dict(self._registry)

    def available(self) -> list[RoutedTool]:
        """Registered tools that are on the allowlist."""
        return [t for t in self._registry.values() if t.name in self.allowed_tools]

    # -- routing ----------------------------------------------------------
    def route(self, name: str, arguments: dict[str, Any] | None = None) -> RouterDecision:
        """Decide whether *name* may run, and whether it needs approval."""
        args = arguments or {}
        if not name:
            return RouterDecision(False, "", "tool name is empty")

        if is_self_loop_tool(name):
            log_warning("tool %s denied: gateway self-loop tool is never served", redact(name))
            return RouterDecision(
                False,
                name,
                f"tool {name!r} calls back into this gateway and is never served",
                arguments=args,
            )

        if name not in self.allowed_tools:
            log_warning("tool %s denied: not in allowlist", redact(name))
            return RouterDecision(
                False,
                name,
                f"tool {name!r} is not in the allowlist ({len(self.allowed_tools)} tools enabled)",
                arguments=args,
            )

        tool = self._registry.get(name)
        if tool is None:
            # Allowed by name but never advertised by the server: still deny,
            # because we cannot validate its arguments.
            return RouterDecision(
                False,
                name,
                f"tool {name!r} was not advertised by the MCP server",
                arguments=args,
            )

        problems = validate_arguments(tool.input_schema, args)
        if problems:
            return RouterDecision(
                False, name, "invalid arguments: " + "; ".join(problems), arguments=args
            )

        risky = tool.risky
        needs_approval = bool(risky and self.require_approval)
        log_debug("tool %s routed risky=%s approval=%s", redact(name), risky, needs_approval)
        return RouterDecision(
            True,
            name,
            "risky tool" if risky else "read-only tool",
            needs_approval=needs_approval,
            arguments=args,
        )

    def needs_approval(self, name: str) -> bool:
        """True when *name* is risky and approval is enabled."""
        return bool(self.require_approval and is_risky_tool(name))
