"""Unit tests for the LLM provider adapter."""

from __future__ import annotations

import json
from typing import ClassVar

import httpx
import pytest

from mcp_for_copilot.provider import (
    KNOWN_MODELS,
    MODEL_ALIASES,
    LLMProvider,
    LLMProviderError,
    ProviderResponse,
    ToolCall,
    _extract_json_object,
    _salvage_tool_call,
    aliases_for,
    resolve_model_id,
)

from ..conftest import make_completion, make_tool_call, mock_transport


class TestResolveModelId:
    @pytest.mark.parametrize(
        ("alias", "expected"),
        [
            ("gpt-6", "gpt-6-sol"),
            ("gpt6", "gpt-6-sol"),
            ("sol", "gpt-6-sol"),
            ("luna", "gpt-6-luna"),
            ("astra", "gpt-6-astra"),
            ("gpt-5.6", "gpt-5.6-sol"),
        ],
    )
    def test_aliases_resolve(self, alias: str, expected: str) -> None:
        assert resolve_model_id(alias) == expected

    def test_canonical_id_passes_through(self) -> None:
        assert resolve_model_id("gpt-6-luna") == "gpt-6-luna"

    def test_case_insensitive(self) -> None:
        assert resolve_model_id("SOL") == "gpt-6-sol"

    def test_unknown_id_raises(self) -> None:
        with pytest.raises(LLMProviderError, match="Unknown model id"):
            resolve_model_id("gpt-9-imaginary")

    def test_empty_id_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from mcp_for_copilot.config import Settings, reset_settings_cache

        reset_settings_cache()
        monkeypatch.setattr(
            "mcp_for_copilot.provider.get_settings",
            lambda: Settings(llm_model_id=""),
        )
        with pytest.raises(LLMProviderError, match="No model configured"):
            resolve_model_id(None)

    def test_aliases_for_returns_all(self) -> None:
        assert aliases_for("gpt-6-sol") == ["gpt-6", "gpt6", "sol"]


class TestModelCatalog:
    def test_every_alias_targets_a_known_model(self) -> None:
        for alias, target in MODEL_ALIASES.items():
            assert target in KNOWN_MODELS, f"{alias} -> {target} is not in KNOWN_MODELS"

    def test_gpt6_sol_metadata(self) -> None:
        spec = KNOWN_MODELS["gpt-6-sol"]
        assert spec["context_window"] == 1_050_000
        assert spec["max_output"] == 128_000
        assert "none" in spec["reasoning_effort"]

    def test_gpt56_luna_pricing_is_not_gpt6_luna_pricing(self) -> None:
        # A copy-paste bug here would silently misreport cost.
        assert KNOWN_MODELS["gpt-5.6-luna"]["input_price_per_mtok"] == 0.2
        assert KNOWN_MODELS["gpt-6-luna"]["input_price_per_mtok"] == 0.1

    @pytest.mark.parametrize("model_id", ["deepseek-r1:14b", "qwen2.5vl:7b"])
    def test_newly_added_local_models_are_registered(self, model_id: str) -> None:
        """These ids were installed in Ollama but missing from KNOWN_MODELS,
        which made resolve_model_id() reject them and /v1/models omit them."""
        spec = KNOWN_MODELS[model_id]
        assert spec["tier"] == "local"
        assert spec["context_window"] > 0
        assert spec["max_output"] > 0
        # Local inference is never billed by this gateway.
        assert spec["input_price_per_mtok"] == 0.0
        assert spec["output_price_per_mtok"] == 0.0


class TestToolCallParsing:
    def test_parses_json_string_arguments(self) -> None:
        call = ToolCall.from_openai(make_tool_call("read_file", {"path": "a.txt"}))
        assert call.name == "read_file"
        assert call.arguments == {"path": "a.txt"}

    def test_parses_dict_arguments(self) -> None:
        call = ToolCall.from_openai({"id": "c1", "function": {"name": "x", "arguments": {"a": 1}}})
        assert call.arguments == {"a": 1}

    def test_malformed_json_is_marked_not_raised(self) -> None:
        call = ToolCall.from_openai(
            {"id": "c1", "function": {"name": "x", "arguments": "{not json"}}
        )
        assert "__malformed_arguments__" in call.arguments

    def test_non_object_json_is_marked(self) -> None:
        call = ToolCall.from_openai({"id": "c1", "function": {"name": "x", "arguments": "[1,2]"}})
        assert "__malformed_arguments__" in call.arguments

    def test_empty_arguments_becomes_empty_dict(self) -> None:
        call = ToolCall.from_openai({"id": "c1", "function": {"name": "x"}})
        assert call.arguments == {}


class TestProviderResponse:
    def test_wants_tools_true_with_calls(self) -> None:
        response = ProviderResponse(tool_calls=[ToolCall(id="1", name="x")])
        assert response.wants_tools is True

    def test_wants_tools_false_without_calls(self) -> None:
        assert ProviderResponse(content="hi").wants_tools is False


class TestLLMProviderInit:
    def test_rejects_unsupported_provider(self, settings) -> None:
        with pytest.raises(LLMProviderError, match="Unsupported LLM_PROVIDER"):
            LLMProvider(provider="anthropic", api_key="k", model_id="gpt-6-sol")

    def test_configured_requires_key(self, settings) -> None:
        assert LLMProvider(api_key="sk-x", model_id="gpt-6-sol").configured is True
        assert LLMProvider(api_key="", model_id="gpt-6-sol").configured is False

    def test_describe_never_leaks_the_key(self) -> None:
        provider = LLMProvider(api_key="sk-super-secret", model_id="gpt-6-sol")
        described = provider.describe()
        assert described["api_key_present"] is True
        assert "sk-super-secret" not in str(described)

    def test_base_url_trailing_slash_is_stripped(self) -> None:
        provider = LLMProvider(
            api_key="k", base_url="https://api.example.test/v1/", model_id="gpt-6-sol"
        )
        assert provider.base_url == "https://api.example.test/v1"


class TestComplete:
    async def test_missing_key_raises(self) -> None:
        provider = LLMProvider(api_key="", model_id="gpt-6-sol")
        with pytest.raises(LLMProviderError, match="LLM_API_KEY is not configured"):
            await provider.complete([{"role": "user", "content": "hi"}])

    async def test_plain_completion(self) -> None:
        transport = mock_transport(make_completion(content="Hello there"))
        provider = LLMProvider(api_key="sk-test", model_id="gpt-6-sol", transport=transport)
        result = await provider.complete([{"role": "user", "content": "hi"}])
        assert result.content == "Hello there"
        assert result.wants_tools is False
        assert result.usage["total_tokens"] == 15

    async def test_tool_calls_are_parsed(self) -> None:
        payload = make_completion(
            content="",
            tool_calls=[make_tool_call("read_file", {"path": "a.txt"})],
            finish_reason="tool_calls",
        )
        provider = LLMProvider(
            api_key="sk-test", model_id="gpt-6-sol", transport=mock_transport(payload)
        )
        result = await provider.complete([{"role": "user", "content": "read a.txt"}])
        assert result.wants_tools is True
        assert result.tool_calls[0].name == "read_file"

    # -- Ollama /v1 shim: tool calls arrive as JSON text in message.content --
    # Ollama 0.34.4 does not populate message.tool_calls. Without the salvage
    # bridge in _parse the whole MCP loop silently degrades to plain chat, so
    # these cases are the regression net for that.

    @pytest.mark.parametrize(
        "text",
        [
            # exactly what llama3.2:1b returned in the live probe
            '{"type":"function","name":"ollama_status","parameters":{}}',
            # what qwen2.5-coder:7b returned in the live probe
            '{"name": "ollama_status", "arguments": {}}',
            # nested OpenAI shape
            '{"function": {"name": "ollama_status", "arguments": "{}"}}',
            # fenced
            '```json\n{"name": "ollama_status", "arguments": {}}\n```',
            # surrounded by prose
            'Sure, calling it now: {"name": "ollama_status", "arguments": {}}',
        ],
    )
    async def test_ollama_text_tool_call_is_salvaged(self, text: str) -> None:
        payload = make_completion(content=text, tool_calls=[], finish_reason="stop")
        provider = LLMProvider(
            api_key="sk-test", model_id="gpt-6-sol", transport=mock_transport(payload)
        )
        result = await provider.complete([{"role": "user", "content": "go"}])
        assert result.wants_tools is True
        assert result.tool_calls[0].name == "ollama_status"
        assert result.content == ""

    async def test_salvage_keeps_arguments(self) -> None:
        payload = make_completion(
            content='{"name": "read_file", "arguments": {"path": "a.txt"}}',
            tool_calls=[],
        )
        provider = LLMProvider(
            api_key="sk-test", model_id="gpt-6-sol", transport=mock_transport(payload)
        )
        result = await provider.complete([{"role": "user", "content": "go"}])
        assert result.tool_calls[0].arguments == {"path": "a.txt"}

    @pytest.mark.parametrize(
        "text",
        [
            "Here is the answer: the file contains three lines.",
            '{"unrelated": "json that is not a tool call"}',
            '{"name": "", "arguments": {}}',
            "no json at all",
        ],
    )
    async def test_prose_is_never_mistaken_for_a_tool_call(self, text: str) -> None:
        """The salvage must not turn arbitrary text into a tool invocation."""
        payload = make_completion(content=text, tool_calls=[])
        provider = LLMProvider(
            api_key="sk-test", model_id="gpt-6-sol", transport=mock_transport(payload)
        )
        result = await provider.complete([{"role": "user", "content": "go"}])
        assert result.wants_tools is False
        assert result.content == text

    async def test_real_tool_calls_win_over_text_salvage(self) -> None:
        payload = make_completion(
            content='{"name": "wrong_tool", "arguments": {}}',
            tool_calls=[make_tool_call("right_tool", {})],
        )
        provider = LLMProvider(
            api_key="sk-test", model_id="gpt-6-sol", transport=mock_transport(payload)
        )
        result = await provider.complete([{"role": "user", "content": "go"}])
        assert [c.name for c in result.tool_calls] == ["right_tool"]
        assert result.content == '{"name": "wrong_tool", "arguments": {}}'

    def test_salvage_handles_braces_inside_strings(self) -> None:
        call = _salvage_tool_call('{"name": "t", "arguments": {"q": "{not a brace}"}}')
        assert call is not None
        assert call.arguments == {"q": "{not a brace}"}

    def test_salvage_marks_malformed_argument_string(self) -> None:
        call = _salvage_tool_call('{"name": "t", "arguments": "{oops"}')
        assert call is not None
        assert "__malformed_arguments__" in call.arguments

    def test_extract_returns_none_for_unbalanced(self) -> None:
        assert _extract_json_object('{"name": "t"') is None

    async def test_reasoning_effort_forced_to_none_with_tools(self) -> None:
        captured: list[dict] = []
        transport = mock_transport(make_completion(), capture=captured)
        provider = LLMProvider(api_key="sk-test", model_id="gpt-6-sol", transport=transport)
        await provider.complete(
            [{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "x", "parameters": {}}}],
            reasoning_effort="high",
        )
        assert captured[0]["reasoning_effort"] == "none"

    async def test_reasoning_effort_with_tools_uses_lowest_supported(self) -> None:
        # gpt-6-astra rejects "none"; the adapter must send its lowest
        # supported effort ("low") instead of hardcoding "none".
        captured: list[dict] = []
        transport = mock_transport(make_completion(), capture=captured)
        provider = LLMProvider(api_key="sk-test", model_id="gpt-6-astra", transport=transport)
        await provider.complete(
            [{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "x", "parameters": {}}}],
            reasoning_effort="high",
        )
        assert captured[0]["reasoning_effort"] == "low"

    async def test_tools_effort_is_always_supported_by_the_model(self) -> None:
        # Whatever the adapter picks with tools present must be a value the
        # model actually advertises, for every model in the catalog.
        for model_id, spec in KNOWN_MODELS.items():
            captured: list[dict] = []
            transport = mock_transport(make_completion(), capture=captured)
            provider = LLMProvider(api_key="sk-test", model_id=model_id, transport=transport)
            await provider.complete(
                [{"role": "user", "content": "hi"}],
                tools=[{"type": "function", "function": {"name": "x", "parameters": {}}}],
            )
            sent = captured[0]["reasoning_effort"]
            assert sent in spec["reasoning_effort"], (
                f"{model_id} was sent reasoning_effort={sent!r}, "
                f"which it does not support: {spec['reasoning_effort']}"
            )

    async def test_reasoning_effort_forwarded_without_tools(self) -> None:
        captured: list[dict] = []
        transport = mock_transport(make_completion(), capture=captured)
        provider = LLMProvider(api_key="sk-test", model_id="gpt-6-sol", transport=transport)
        await provider.complete([{"role": "user", "content": "hi"}], reasoning_effort="high")
        assert captured[0]["reasoning_effort"] == "high"

    async def test_reasoning_effort_omitted_when_unset(self) -> None:
        captured: list[dict] = []
        transport = mock_transport(make_completion(), capture=captured)
        provider = LLMProvider(api_key="sk-test", model_id="gpt-6-sol", transport=transport)
        await provider.complete([{"role": "user", "content": "hi"}])
        assert "reasoning_effort" not in captured[0]

    async def test_unsupported_effort_raises(self) -> None:
        provider = LLMProvider(
            api_key="sk-test", model_id="gpt-6-sol", transport=mock_transport(make_completion())
        )
        with pytest.raises(LLMProviderError, match="not supported by"):
            await provider.complete([{"role": "user", "content": "hi"}], reasoning_effort="turbo")

    async def test_http_error_is_redacted(self) -> None:
        transport = mock_transport(
            {"error": {"message": "bad key sk-abcdef1234567890abcdef"}}, status_code=401
        )
        provider = LLMProvider(api_key="sk-test", model_id="gpt-6-sol", transport=transport)
        with pytest.raises(LLMProviderError) as excinfo:
            await provider.complete([{"role": "user", "content": "hi"}])
        assert "HTTP 401" in str(excinfo.value)
        assert "sk-abcdef1234567890abcdef" not in str(excinfo.value)

    async def test_no_choices_raises(self) -> None:
        provider = LLMProvider(
            api_key="sk-test",
            model_id="gpt-6-sol",
            transport=mock_transport({"choices": []}),
        )
        with pytest.raises(LLMProviderError, match="no choices"):
            await provider.complete([{"role": "user", "content": "hi"}])

    async def test_timeout_raises_provider_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("too slow", request=request)

        provider = LLMProvider(
            api_key="sk-test",
            model_id="gpt-6-sol",
            transport=httpx.MockTransport(handler),
        )
        with pytest.raises(LLMProviderError, match="timed out"):
            await provider.complete([{"role": "user", "content": "hi"}])

    async def test_max_tokens_uses_max_completion_tokens(self) -> None:
        captured: list[dict] = []
        provider = LLMProvider(
            api_key="sk-test",
            model_id="gpt-6-sol",
            transport=mock_transport(make_completion(), capture=captured),
        )
        await provider.complete([{"role": "user", "content": "hi"}], max_tokens=64)
        assert captured[0]["max_completion_tokens"] == 64


class TestToOpenAITools:
    def test_converts_mcp_tools(self, fake_tools) -> None:
        converted = LLMProvider.to_openai_tools(fake_tools)
        assert len(converted) == 2
        assert converted[0]["type"] == "function"
        assert converted[0]["function"]["name"] == "read_file"
        assert converted[0]["function"]["parameters"]["required"] == ["path"]

    def test_skips_nameless_tools(self) -> None:
        class Nameless:
            name = ""
            description = "x"
            input_schema: ClassVar[dict] = {}

        assert LLMProvider.to_openai_tools([Nameless()]) == []

    def test_missing_schema_defaults_to_empty_object(self) -> None:
        class Bare:
            name = "bare"
            description = ""
            input_schema = None

        converted = LLMProvider.to_openai_tools([Bare()])
        assert converted[0]["function"]["parameters"] == {"type": "object", "properties": {}}


class TestZaiFallback:
    """The Z.ai (GLM) backup used when the primary provider is unreachable."""

    def _provider(self, transport, **kwargs) -> LLMProvider:
        defaults = {
            "api_key": "sk-primary",
            "model_id": "gpt-6-sol",
            "base_url": "https://api.openai.test/v1",
            "transport": transport,
            "fallback_api_key": "zai-key",
            "fallback_base_url": "https://api.z.ai/api/paas/v4",
            "fallback_model_id": "glm-4.5-flash",
            "fallback_enabled": True,
        }
        defaults.update(kwargs)
        return LLMProvider(**defaults)

    def test_glm_models_are_registered(self) -> None:
        for model_id in ("glm-4.5-flash", "glm-5", "glm-5.1", "glm-5.3-flash"):
            assert model_id in KNOWN_MODELS
            assert KNOWN_MODELS[model_id]["reasoning_effort"] == ("none",)

    @pytest.mark.parametrize(
        ("alias", "expected"),
        [
            ("zai", "glm-5"),
            ("glm", "glm-5"),
            ("glm-flash", "glm-4.5-flash"),
            ("zai-flash", "glm-4.5-flash"),
            ("glm51", "glm-5.1"),
        ],
    )
    def test_glm_aliases_resolve(self, alias: str, expected: str) -> None:
        assert resolve_model_id(alias) == expected

    def test_fallback_configured_requires_key(self) -> None:
        assert self._provider(mock_transport(make_completion())).fallback_configured is True
        assert (
            self._provider(mock_transport(make_completion()), fallback_api_key="").fallback_configured
            is False
        )
        assert (
            self._provider(
                mock_transport(make_completion()), fallback_enabled=False
            ).fallback_configured
            is False
        )

    def test_describe_never_leaks_the_fallback_key(self) -> None:
        described = self._provider(mock_transport(make_completion())).describe()
        assert described["fallback_api_key_present"] is True
        assert "zai-key" not in str(described)

    async def test_5xx_triggers_fallback(self) -> None:
        captured: list[dict] = []
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(json.loads(request.content.decode("utf-8")))
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(503, json={"error": {"message": "down"}})
            return httpx.Response(200, json=make_completion(content="from zai"))

        provider = self._provider(httpx.MockTransport(handler))
        result = await provider.complete([{"role": "user", "content": "hi"}])
        assert result.content == "from zai"
        assert len(captured) == 2
        assert captured[1]["model"] == "glm-4.5-flash"

    async def test_transport_error_triggers_fallback(self) -> None:
        captured: list[dict] = []
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(json.loads(request.content.decode("utf-8")))
            calls["n"] += 1
            if calls["n"] == 1:
                raise httpx.ConnectError("no route to host", request=request)
            return httpx.Response(200, json=make_completion(content="zai ok"))

        provider = self._provider(httpx.MockTransport(handler))
        result = await provider.complete([{"role": "user", "content": "hi"}])
        assert result.content == "zai ok"
        assert captured[1]["model"] == "glm-4.5-flash"

    async def test_timeout_triggers_fallback(self) -> None:
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] == 1:
                raise httpx.ReadTimeout("too slow", request=request)
            return httpx.Response(200, json=make_completion(content="zai ok"))

        provider = self._provider(httpx.MockTransport(handler))
        result = await provider.complete([{"role": "user", "content": "hi"}])
        assert result.content == "zai ok"

    async def test_4xx_does_not_trigger_fallback(self) -> None:
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(400, json={"error": {"message": "bad request"}})

        provider = self._provider(httpx.MockTransport(handler))
        with pytest.raises(LLMProviderError, match="HTTP 400"):
            await provider.complete([{"role": "user", "content": "hi"}])
        assert calls["n"] == 1

    async def test_no_fallback_when_not_configured(self) -> None:
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(503, json={"error": {"message": "down"}})

        provider = self._provider(httpx.MockTransport(handler), fallback_api_key="")
        with pytest.raises(LLMProviderError, match="HTTP 503"):
            await provider.complete([{"role": "user", "content": "hi"}])
        assert calls["n"] == 1

    async def test_both_failures_are_reported(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(503, json={"error": {"message": "down"}})

        provider = self._provider(httpx.MockTransport(handler))
        with pytest.raises(LLMProviderError) as excinfo:
            await provider.complete([{"role": "user", "content": "hi"}])
        message = str(excinfo.value)
        assert "primary LLM failed" in message
        assert "Z.ai fallback also failed" in message

    async def test_fallback_never_sends_reasoning_effort(self) -> None:
        captured: list[dict] = []
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(json.loads(request.content.decode("utf-8")))
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(503, json={"error": {"message": "down"}})
            return httpx.Response(200, json=make_completion(content="zai ok"))

        provider = self._provider(httpx.MockTransport(handler))
        await provider.complete([{"role": "user", "content": "hi"}], reasoning_effort="high")
        assert "reasoning_effort" not in captured[1]

    async def test_fallback_preserves_tools(self) -> None:
        captured: list[dict] = []
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(json.loads(request.content.decode("utf-8")))
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(503, json={"error": {"message": "down"}})
            return httpx.Response(200, json=make_completion(content="zai ok"))

        provider = self._provider(httpx.MockTransport(handler))
        tools = [{"type": "function", "function": {"name": "x", "parameters": {}}}]
        await provider.complete([{"role": "user", "content": "hi"}], tools=tools)
        assert captured[1]["tools"] == tools
        assert captured[1]["tool_choice"] == "auto"

    async def test_invalid_fallback_model_is_reported(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(503, json={"error": {"message": "down"}})

        provider = self._provider(
            httpx.MockTransport(handler), fallback_model_id="glm-9-imaginary"
        )
        with pytest.raises(LLMProviderError, match="fallback model is invalid"):
            await provider.complete([{"role": "user", "content": "hi"}])

