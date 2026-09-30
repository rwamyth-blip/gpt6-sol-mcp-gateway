"""Unit tests for the tool router — the allowlist and validation gate."""

from __future__ import annotations

import pytest

from mcp_for_copilot.mcp_client import MCPTool
from mcp_for_copilot.tool_router import (
    DEFAULT_ALLOWED_TOOLS,
    ToolRouter,
    is_risky_tool,
    is_self_loop_tool,
    validate_arguments,
)


class TestIsRiskyTool:
    @pytest.mark.parametrize(
        "name",
        ["write_file", "delete_record", "execute_sql", "deploy_app", "mongo_aggregate"],
    )
    def test_mutating_names_are_risky(self, name: str) -> None:
        assert is_risky_tool(name) is True

    @pytest.mark.parametrize(
        "name", ["read_file", "list_files", "mongo_find", "mongo_count", "grep"]
    )
    def test_read_only_names_are_safe(self, name: str) -> None:
        assert is_risky_tool(name) is False

    def test_explicit_safe_beats_risky_verb(self) -> None:
        # "mongo_find" contains no risky verb, but "list_files" would match
        # nothing either; the explicit set must win over a substring match.
        assert is_risky_tool("mongo_list_collections") is False

    def test_empty_name_fails_closed(self) -> None:
        assert is_risky_tool("") is True

    def test_unknown_name_with_risky_verb_is_risky(self) -> None:
        assert is_risky_tool("custom_delete_thing") is True

    def test_verb_must_be_a_whole_word(self) -> None:
        # "runner" contains "run" but not as a word.
        assert is_risky_tool("runner_status") is False


class TestValidateArguments:
    def test_valid_object_passes(self) -> None:
        schema = {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        }
        assert validate_arguments(schema, {"path": "a.txt"}) == []

    def test_missing_required_is_reported(self) -> None:
        schema = {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        }
        problems = validate_arguments(schema, {})
        assert any("missing required" in p for p in problems)

    def test_wrong_type_is_reported(self) -> None:
        schema = {"type": "object", "properties": {"count": {"type": "integer"}}}
        problems = validate_arguments(schema, {"count": "five"})
        assert any("must be" in p for p in problems)

    def test_boolean_is_not_an_integer(self) -> None:
        # bool is a subclass of int in Python; this must still be rejected.
        schema = {"type": "object", "properties": {"count": {"type": "integer"}}}
        problems = validate_arguments(schema, {"count": True})
        assert any("boolean" in p for p in problems)

    def test_boolean_accepted_when_declared(self) -> None:
        schema = {"type": "object", "properties": {"flag": {"type": "boolean"}}}
        assert validate_arguments(schema, {"flag": True}) == []

    def test_enum_is_enforced(self) -> None:
        schema = {"type": "object", "properties": {"mode": {"type": "string", "enum": ["a", "b"]}}}
        assert validate_arguments(schema, {"mode": "c"}) != []
        assert validate_arguments(schema, {"mode": "a"}) == []

    def test_array_items_are_validated(self) -> None:
        schema = {
            "type": "object",
            "properties": {"ids": {"type": "array", "items": {"type": "integer"}}},
        }
        problems = validate_arguments(schema, {"ids": [1, "two"]})
        assert any("ids[1]" in p for p in problems)

    def test_additional_properties_false_rejects_extras(self) -> None:
        schema = {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "additionalProperties": False,
        }
        problems = validate_arguments(schema, {"path": "a", "extra": 1})
        assert any("unexpected argument" in p for p in problems)

    def test_empty_schema_accepts_anything(self) -> None:
        assert validate_arguments({}, {"anything": 1}) == []

    def test_non_dict_arguments_rejected(self) -> None:
        assert validate_arguments({"type": "object"}, "nope") != []


class TestToolRouter:
    def test_default_allowlist_is_read_only(self) -> None:
        router = ToolRouter()
        assert router.allowed_tools == DEFAULT_ALLOWED_TOOLS
        assert "write_file" not in router.allowed_tools

    def test_unregistered_tool_is_denied(self) -> None:
        router = ToolRouter(allowed_tools=["read_file"])
        decision = router.route("read_file", {"path": "a"})
        assert decision.allowed is False
        assert "not advertised" in decision.reason

    def test_tool_outside_allowlist_is_denied(self) -> None:
        router = ToolRouter(allowed_tools=["read_file"])
        router.register(MCPTool(name="write_file", input_schema={}))
        decision = router.route("write_file", {})
        assert decision.allowed is False
        assert "allowlist" in decision.reason

    def test_allowed_read_only_tool_passes_without_approval(self) -> None:
        router = ToolRouter(allowed_tools=["read_file"])
        router.register(
            MCPTool(
                name="read_file",
                input_schema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            )
        )
        decision = router.route("read_file", {"path": "a.txt"})
        assert decision.allowed is True
        assert decision.needs_approval is False

    def test_risky_tool_needs_approval(self) -> None:
        router = ToolRouter(allowed_tools=["write_file"], require_approval=True)
        router.register(MCPTool(name="write_file", input_schema={}))
        decision = router.route("write_file", {})
        assert decision.allowed is True
        assert decision.needs_approval is True

    def test_risky_tool_skips_approval_when_disabled(self) -> None:
        router = ToolRouter(allowed_tools=["write_file"], require_approval=False)
        router.register(MCPTool(name="write_file", input_schema={}))
        decision = router.route("write_file", {})
        assert decision.allowed is True
        assert decision.needs_approval is False

    def test_invalid_arguments_are_denied(self) -> None:
        router = ToolRouter(allowed_tools=["read_file"])
        router.register(
            MCPTool(
                name="read_file",
                input_schema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            )
        )
        decision = router.route("read_file", {})
        assert decision.allowed is False
        assert "invalid arguments" in decision.reason

    def test_empty_tool_name_is_denied(self) -> None:
        assert ToolRouter().route("", {}).allowed is False

    def test_available_filters_by_allowlist(self, fake_tools: list[MCPTool]) -> None:
        router = ToolRouter(allowed_tools=["read_file"])
        router.register_all(fake_tools)
        assert [t.name for t in router.available()] == ["read_file"]

    def test_register_rejects_nameless_tool(self) -> None:
        from mcp_for_copilot.tool_router import ToolRouterError

        with pytest.raises(ToolRouterError):
            ToolRouter().register(MCPTool(name=""))

    def test_needs_approval_helper(self) -> None:
        router = ToolRouter(require_approval=True)
        assert router.needs_approval("delete_file") is True
        assert router.needs_approval("read_file") is False


class TestSelfLoopDenylist:
    """Guardrail F2: gateway front-ends must never be served to the model."""

    @pytest.mark.parametrize(
        "name", ["gpt6_chat", "gpt6_models", "gpt6_status", "gpt6_tools", "gateway_chat"]
    )
    def test_self_loop_names_are_detected(self, name: str) -> None:
        assert is_self_loop_tool(name) is True

    @pytest.mark.parametrize("name", ["read_file", "mongo_find", "ollama_chat"])
    def test_normal_tools_are_not_self_loop(self, name: str) -> None:
        assert is_self_loop_tool(name) is False

    def test_register_skips_self_loop_tool(self) -> None:
        router = ToolRouter(allowed_tools=["gpt6_chat", "read_file"])
        assert router.register(MCPTool(name="gpt6_chat", input_schema={})) is None
        assert "gpt6_chat" not in router.registered

    def test_register_all_drops_self_loop_tools(self) -> None:
        router = ToolRouter(allowed_tools=["gateway_chat", "read_file"])
        registered = router.register_all(
            [
                MCPTool(name="gateway_chat", input_schema={}),
                MCPTool(name="read_file", input_schema={}),
            ]
        )
        assert [t.name for t in registered] == ["read_file"]
        assert "gateway_chat" not in router.registered

    def test_route_denies_self_loop_even_when_allowlisted(self) -> None:
        router = ToolRouter(allowed_tools=["gpt6_chat"])
        decision = router.route("gpt6_chat", {})
        assert decision.allowed is False
        assert "never served" in decision.reason
