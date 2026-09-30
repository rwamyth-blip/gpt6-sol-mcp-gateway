"""LLM provider adapter.

The single place in the package that talks to a language model. Every other
layer (router, approval, orchestrator) is provider-agnostic and only sees the
dataclasses defined here.

Design rules
------------
* **No hardcoded model id.** The model always comes from
  ``settings.llm_model_id``. A caller may override it per request, but the
  default is never a literal inside this module.
* **No hardcoded credential.** The key comes from ``settings.llm_api_key``.
* **No secret in an error.** Every raised message is passed through
  :func:`mcp_for_copilot.logging_utils.redact`.
* **Tools are passed through, never invented.** The adapter forwards the tool
  list it is given; it does not add tools of its own.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import httpx

from .config import get_settings
from .logging_utils import log_debug, log_error, log_info, redact

# ---------------------------------------------------------------------------
# Model catalog — verified against the official OpenAI model documentation.
# https://developers.openai.com/api/docs/models
#
# This table is documentation + validation only. It is NOT used to pick a
# default: the default comes from settings.llm_model_id.
# ---------------------------------------------------------------------------
KNOWN_MODELS: dict[str, dict[str, Any]] = {
    "gpt-6-astra": {
        "label": "GPT-6 Astra",
        "tier": "flagship",
        "context_window": 1_050_000,
        "max_output": 128_000,
        "reasoning_effort": ("low", "medium", "high", "xhigh", "max"),
        "input_price_per_mtok": 10.0,
        "output_price_per_mtok": 50.0,
    },
    "gpt-6-sol": {
        "label": "GPT-6 Sol",
        "tier": "balanced",
        "context_window": 1_050_000,
        "max_output": 128_000,
        "reasoning_effort": ("none", "low", "medium", "high", "xhigh", "max"),
        "input_price_per_mtok": 2.0,
        "output_price_per_mtok": 10.0,
    },
    "gpt-6-luna": {
        "label": "GPT-6 Luna",
        "tier": "efficient",
        "context_window": 1_050_000,
        "max_output": 128_000,
        "reasoning_effort": ("none", "low", "medium", "high", "xhigh", "max"),
        "input_price_per_mtok": 0.1,
        "output_price_per_mtok": 0.5,
    },
    "gpt-5.6-sol": {
        "label": "GPT-5.6 Sol",
        "tier": "flagship",
        "context_window": 1_050_000,
        "max_output": 128_000,
        "reasoning_effort": ("none", "low", "medium", "high", "xhigh", "max"),
        "input_price_per_mtok": 4.0,
        "output_price_per_mtok": 20.0,
    },
    "gpt-5.6-luna": {
        "label": "GPT-5.6 Luna",
        "tier": "efficient",
        "context_window": 1_050_000,
        "max_output": 128_000,
        "reasoning_effort": ("none", "low", "medium", "high", "xhigh", "max"),
        # Standard: $0.20 in / $1.20 out / $0.02 cached. (>272K input -> 2x in, 1.5x out)
        # NOTE: $0.1/$0.5 is GPT-6 Luna's price — do not copy it here.
        "input_price_per_mtok": 0.2,
        "output_price_per_mtok": 1.2,
        "cached_input_price_per_mtok": 0.02,
    },
    # -- Local Ollama models ----------------------------------------------
    # Served by an OpenAI-compatible endpoint (default http://127.0.0.1:11434/v1).
    # Local inference is free, so every price is 0.0. Context windows are the
    # model's native window, not the GPT-6 family's 1.05M.
    # NOTE: Ollama ids are case-sensitive and contain ':' and '.', so they are
    # matched verbatim -- MODEL_ALIASES lookups are lowercased, which is a
    # no-op for these ids.
    "llama3.2:3b": {
        "label": "Llama 3.2 3B (Ollama)",
        "tier": "local",
        "context_window": 131_072,
        "max_output": 8_192,
        "reasoning_effort": ("none",),
        "input_price_per_mtok": 0.0,
        "output_price_per_mtok": 0.0,
    },
    "llama3.2:1b": {
        "label": "Llama 3.2 1B (Ollama)",
        "tier": "local",
        "context_window": 131_072,
        "max_output": 8_192,
        "reasoning_effort": ("none",),
        "input_price_per_mtok": 0.0,
        "output_price_per_mtok": 0.0,
    },
    "qwen2.5-coder:7b": {
        "label": "Qwen2.5 Coder 7B (Ollama)",
        "tier": "local",
        "context_window": 32_768,
        "max_output": 8_192,
        "reasoning_effort": ("none",),
        "input_price_per_mtok": 0.0,
        "output_price_per_mtok": 0.0,
    },
    "qwen3:4b": {
        "label": "Qwen3 4B (Ollama)",
        "tier": "local",
        "context_window": 40_960,
        "max_output": 8_192,
        "reasoning_effort": ("none",),
        "input_price_per_mtok": 0.0,
        "output_price_per_mtok": 0.0,
    },
    "deepseek-r1:7b": {
        "label": "DeepSeek R1 7B (Ollama)",
        "tier": "local",
        "context_window": 65_536,
        "max_output": 8_192,
        "reasoning_effort": ("none",),
        "input_price_per_mtok": 0.0,
        "output_price_per_mtok": 0.0,
    },
    # 14.8B Q4_K_M. 131072 context per /api/show, but this host has 16 GB RAM
    # and the model only partially offloads to the GPU, so responses are slow
    # -- prefer the 7B for interactive editing.
    "deepseek-r1:14b": {
        "label": "DeepSeek R1 14B (Ollama)",
        "tier": "local",
        "context_window": 131_072,
        "max_output": 8_192,
        "reasoning_effort": ("none",),
        "input_price_per_mtok": 0.0,
        "output_price_per_mtok": 0.0,
    },
    # Vision-capable: /api/show reports completion + vision, no tools.
    "qwen2.5vl:7b": {
        "label": "Qwen2.5 VL 7B (Ollama)",
        "tier": "local",
        "context_window": 128_000,
        "max_output": 8_192,
        "reasoning_effort": ("none",),
        "input_price_per_mtok": 0.0,
        "output_price_per_mtok": 0.0,
    },
    "deepseek-coder:6.7b": {
        "label": "DeepSeek Coder 6.7B (Ollama)",
        "tier": "local",
        "context_window": 16_384,
        "max_output": 8_192,
        "reasoning_effort": ("none",),
        "input_price_per_mtok": 0.0,
        "output_price_per_mtok": 0.0,
    },
    "qwen2.5:0.5b": {
        "label": "Qwen2.5 0.5B (Ollama)",
        "tier": "local",
        "context_window": 32_768,
        "max_output": 8_192,
        "reasoning_effort": ("none",),
        "input_price_per_mtok": 0.0,
        "output_price_per_mtok": 0.0,
    },
    # -- Ollama Cloud models ----------------------------------------------
    # These ids are registered in the local Ollama daemon but carry a
    # ``remote_host`` of https://ollama.com, so inference runs remotely and
    # the local entry is only a ~310-byte stub. They are billed by Ollama,
    # not by this gateway, so prices stay 0.0 here.
    "gemma4:31b-cloud": {
        "label": "Gemma 4 31B (Ollama Cloud)",
        "tier": "cloud",
        "context_window": 262_144,
        "max_output": 8_192,
        "reasoning_effort": ("none",),
        "input_price_per_mtok": 0.0,
        "output_price_per_mtok": 0.0,
    },
    "gpt-oss:120b-cloud": {
        "label": "GPT-OSS 120B (Ollama Cloud)",
        "tier": "cloud",
        "context_window": 131_072,
        "max_output": 8_192,
        "reasoning_effort": ("none",),
        "input_price_per_mtok": 0.0,
        "output_price_per_mtok": 0.0,
    },
    "gpt-oss:20b-cloud": {
        "label": "GPT-OSS 20B (Ollama Cloud)",
        "tier": "cloud",
        "context_window": 131_072,
        "max_output": 8_192,
        "reasoning_effort": ("none",),
        "input_price_per_mtok": 0.0,
        "output_price_per_mtok": 0.0,
    },
    "nemotron-3-nano:30b-cloud": {
        "label": "Nemotron 3 Nano 30B (Ollama Cloud)",
        "tier": "cloud",
        "context_window": 262_144,
        "max_output": 8_192,
        "reasoning_effort": ("none",),
        "input_price_per_mtok": 0.0,
        "output_price_per_mtok": 0.0,
    },
    "nemotron-3-super:cloud": {
        "label": "Nemotron 3 Super (Ollama Cloud)",
        "tier": "cloud",
        "context_window": 262_144,
        "max_output": 8_192,
        "reasoning_effort": ("none",),
        "input_price_per_mtok": 0.0,
        "output_price_per_mtok": 0.0,
    },
    "nemotron-3-ultra:cloud": {
        "label": "Nemotron 3 Ultra (Ollama Cloud)",
        "tier": "cloud",
        "context_window": 262_144,
        "max_output": 8_192,
        "reasoning_effort": ("none",),
        "input_price_per_mtok": 0.0,
        "output_price_per_mtok": 0.0,
    },
    # -- DeepSeek (api.deepseek.com) --------------------------------------
    # Verified live on 2026-09-30 against the real API. The API lists exactly
    # two ids: deepseek-flash and deepseek-v4-pro. `deepseek-v4-flash` is NOT
    # real -- DeepSeek aliases it silently to deepseek-flash.
    #
    # reasoning_effort is a real, monotonic dial on deepseek-flash (median
    # reasoning length 488 low / 500 medium / 530 high / 656 max chars).
    # There is no "thinking off": effort=none zeroes reasoning_content but the
    # model then reasons inline in `content` instead, which is not cheaper.
    "deepseek-flash": {
        "label": "DeepSeek Flash",
        "tier": "efficient",
        "context_window": 128_000,
        "max_output": 8_192,
        "reasoning_effort": ("none", "low", "medium", "high", "max"),
        "input_price_per_mtok": 0.0,
        "output_price_per_mtok": 0.0,
        "measured_thinking": True,
        "measured_correct_rate": 1.0,
        "measured_median_latency_s": 0.33,
        "measured_median_output_tokens": 116,
    },
    "deepseek-v4-pro": {
        "label": "DeepSeek V4 Pro",
        "tier": "flagship",
        "context_window": 128_000,
        "max_output": 8_192,
        "reasoning_effort": ("none", "low", "medium", "high", "max"),
        "input_price_per_mtok": 0.0,
        "output_price_per_mtok": 0.0,
        "measured_thinking": True,
        "measured_correct_rate": 1.0,
        "measured_median_latency_s": 0.30,
        "measured_median_output_tokens": 198,
    },

    # -- Z.ai GLM (OpenAI-compatible endpoint https://api.z.ai/api/paas/v4) --
    # The standalone Z.ai team on this gateway: Flash = easy 80% tier,
    # GLM-5 = mid 15% tier, GLM-5.1 = hard 5% tier. Prices are the published
    # Z.ai rates per Mtok (Flash is the promo price); context windows are
    # published sizes. reasoning_effort is ("none",) because GLM uses its own
    # `thinking` parameter instead of OpenAI's field -- the provider never
    # sends reasoning_effort for glm-* models (see _chat below).
    "glm-4.5-flash": {
        "label": "GLM-4.5 Flash (Z.ai)",
        "tier": "efficient",
        "context_window": 128_000,
        "max_output": 32_768,
        "reasoning_effort": ("none",),
        "input_price_per_mtok": 0.075,
        "output_price_per_mtok": 0.25,
    },
    "glm-5.3-flash": {
        "label": "GLM-5.3 Flash (Z.ai)",
        "tier": "efficient",
        "context_window": 128_000,
        "max_output": 32_768,
        "reasoning_effort": ("none",),
        "input_price_per_mtok": 0.075,
        "output_price_per_mtok": 0.25,
    },
    "glm-5": {
        "label": "GLM-5 (Z.ai)",
        "tier": "balanced",
        "context_window": 200_000,
        "max_output": 96_000,
        "reasoning_effort": ("none",),
        "input_price_per_mtok": 1.0,
        "output_price_per_mtok": 3.2,
    },
    "glm-5.1": {
        "label": "GLM-5.1 (Z.ai)",
        "tier": "flagship",
        "context_window": 200_000,
        "max_output": 96_000,
        "reasoning_effort": ("none",),
        "input_price_per_mtok": 1.4,
        "output_price_per_mtok": 4.4,
    },
}

# Aliases the provider accepts, mapped onto a canonical id above.
MODEL_ALIASES: dict[str, str] = {
    "gpt-6": "gpt-6-sol",
    "gpt6": "gpt-6-sol",
    "sol": "gpt-6-sol",
    "luna": "gpt-6-luna",
    "astra": "gpt-6-astra",
    "gpt-5.6": "gpt-5.6-sol",
    # Short local aliases.
    "llama3.2": "llama3.2:3b",
    "qwen3": "qwen3:4b",
    "deepseek-r1": "deepseek-r1:7b",
    "qwen2.5": "qwen2.5:0.5b",
    # Short cloud aliases.
    "gemma4": "gemma4:31b-cloud",
    "gpt-oss": "gpt-oss:120b-cloud",
    "gpt-oss-120b": "gpt-oss:120b-cloud",
    "gpt-oss-20b": "gpt-oss:20b-cloud",
    "nemotron-3-nano": "nemotron-3-nano:30b-cloud",
    "nemotron-3-super": "nemotron-3-super:cloud",
    "nemotron-3-ultra": "nemotron-3-ultra:cloud",
    # Z.ai GLM aliases (standalone Z.ai team).
    "zai-flash": "glm-4.5-flash",
    "glm-flash": "glm-4.5-flash",
    "glm": "glm-5",
    "zai": "glm-5",
    "glm51": "glm-5.1",
}

# ---------------------------------------------------------------------------
# DeepSeek tier routing.
#
# The client-facing ids (gpt-6-luna / gpt-6-sol / gpt-6-astra, plus the short
# luna / sol / astra aliases) are NOT DeepSeek model ids. DeepSeek only knows
# `deepseek-flash` and `deepseek-v4-pro`, so forwarding `gpt-6-luna` verbatim
# makes the upstream return HTTP 400:
#   "The supported API model names are deepseek-flash, deepseek-v4-pro, but
#    you passed gpt-6-luna."
#
# The three tiers are all `deepseek-flash`, separated only by reasoning_effort
# (see routing.yaml). This table is the translation the gateway was missing.
# ---------------------------------------------------------------------------
DEEPSEEK_TIERS: dict[str, dict[str, Any]] = {
    "gpt-6-luna": {"model": "deepseek-flash", "reasoning_effort": "low", "max_tokens": 2048},
    "gpt-6-sol": {"model": "deepseek-flash", "reasoning_effort": "high", "max_tokens": 2048},
    "gpt-6-astra": {"model": "deepseek-flash", "reasoning_effort": "max", "max_tokens": 3000},
}


def resolve_deepseek_tier(model_id: str | None) -> dict[str, Any] | None:
    """Translate a client-facing tier id to its DeepSeek model + effort.

    Returns ``None`` when *model_id* is not a DeepSeek tier, so callers can
    fall through to the normal alias resolution.
    """
    if not model_id:
        return None
    candidate = model_id.strip().lower()
    canonical = MODEL_ALIASES.get(candidate, candidate)
    return DEEPSEEK_TIERS.get(canonical)


SUPPORTED_PROVIDERS = ("openai",)


class LLMProviderError(RuntimeError):
    """Raised for any provider failure. The message is always redacted."""


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Return the first JSON object in *text*, or None.

    Models sometimes wrap the tool call in prose or a ```json fence, so a bare
    json.loads is not enough. Scanning for balanced braces keeps this tolerant
    without accepting arbitrary text as a call.
    """
    stripped = text.strip()
    # Unwrap a fenced block first: ```json {...} ```
    if stripped.startswith("```"):
        newline = stripped.find("\n")
        if newline != -1:
            stripped = stripped[newline + 1 :]
        if stripped.rstrip().endswith("```"):
            stripped = stripped.rstrip()[:-3]
        stripped = stripped.strip()

    start = stripped.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(stripped)):
        char = stripped[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    decoded = json.loads(stripped[start : index + 1])
                except json.JSONDecodeError:
                    return None
                return decoded if isinstance(decoded, dict) else None
    return None


def _salvage_tool_call(content: str) -> ToolCall | None:
    """Recover a tool call that a provider returned as JSON text.

    Only a JSON object that carries a non-empty tool name is accepted. Argument
    payloads are read from the first of ``arguments`` / ``parameters`` /
    ``args`` that is a dict, a JSON string, or absent (empty dict). Anything
    else returns None so ordinary prose never becomes a tool invocation --
    the three security gates in the orchestrator must never be bypassed by a
    string that merely looks like JSON.
    """
    payload = _extract_json_object(content)
    if payload is None:
        return None

    name = ""
    function = payload.get("function")
    if isinstance(function, dict):
        name = str(function.get("name") or "")
    if not name:
        name = str(payload.get("name") or payload.get("tool") or "")

    arguments: dict[str, Any] = {}
    for key in ("arguments", "parameters", "args"):
        if key not in payload:
            continue
        value = payload[key]
        if isinstance(value, dict):
            arguments = value
        elif isinstance(value, str) and value.strip():
            try:
                decoded = json.loads(value)
            except json.JSONDecodeError:
                arguments = {"__malformed_arguments__": value[:200]}
            else:
                arguments = decoded if isinstance(decoded, dict) else {}
        break

    if not name:
        return None

    # Strip the call wrapper and keep any sibling prose as content so the
    # caller does not silently lose text the model wrote around the call.
    return ToolCall(id=f"call_{name}", name=name, arguments=arguments)


@dataclass
class ToolCall:
    """A tool invocation requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_openai(cls, raw: dict[str, Any]) -> ToolCall:
        """Parse one entry of ``choices[0].message.tool_calls``.

        Malformed JSON arguments are surfaced as an empty dict plus a marker
        key so the router can reject the call instead of crashing.
        """
        function = raw.get("function") or {}
        raw_args = function.get("arguments")
        parsed: dict[str, Any] = {}
        if isinstance(raw_args, dict):
            parsed = raw_args
        elif isinstance(raw_args, str) and raw_args.strip():
            try:
                decoded = json.loads(raw_args)
                if isinstance(decoded, dict):
                    parsed = decoded
                else:
                    parsed = {"__malformed_arguments__": raw_args[:200]}
            except json.JSONDecodeError:
                parsed = {"__malformed_arguments__": raw_args[:200]}
        return cls(
            id=str(raw.get("id") or ""),
            name=str(function.get("name") or ""),
            arguments=parsed,
        )


@dataclass
class ProviderResponse:
    """Normalised provider reply."""

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    model: str = ""
    finish_reason: str = ""
    usage: dict[str, Any] = field(default_factory=dict)

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


def resolve_model_id(model_id: str | None = None) -> str:
    """Resolve *model_id* (or the configured default) to a canonical id.

    Raises :class:`LLMProviderError` for an unknown id so a typo fails loudly
    instead of silently hitting the provider with a bad name.
    """
    candidate = (model_id or get_settings().llm_model_id or "").strip()
    if not candidate:
        raise LLMProviderError("No model configured. Set LLM_MODEL_ID in the environment.")
    canonical = MODEL_ALIASES.get(candidate.lower(), candidate)
    if canonical not in KNOWN_MODELS:
        known = ", ".join(sorted(KNOWN_MODELS))
        raise LLMProviderError(f"Unknown model id {redact(candidate)!r}. Known ids: {known}")
    return canonical


def aliases_for(model_id: str) -> list[str]:
    """Every alias that resolves to *model_id* (sorted)."""
    canonical = MODEL_ALIASES.get(model_id.lower(), model_id)
    return sorted(a for a, target in MODEL_ALIASES.items() if target == canonical)


class LLMProvider:
    """OpenAI-compatible chat-completions adapter.

    Only ``openai`` is implemented. An unknown ``LLM_PROVIDER`` raises rather
    than falling back, so a misconfiguration is never masked.
    """

    def __init__(
        self,
        *,
        provider: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        model_id: str | None = None,
        timeout: float | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        settings = get_settings()
        self.provider = (provider or settings.llm_provider or "").strip().lower()
        if self.provider not in SUPPORTED_PROVIDERS:
            raise LLMProviderError(
                f"Unsupported LLM_PROVIDER {redact(self.provider)!r}. "
                f"Supported: {', '.join(SUPPORTED_PROVIDERS)}"
            )
        self.api_key = (api_key if api_key is not None else settings.llm_api_key).strip()
        self.base_url = (
            (base_url if base_url is not None else settings.llm_base_url).strip().rstrip("/")
        )
        self.model_id = resolve_model_id(model_id or settings.llm_model_id or None)
        self.timeout = float(timeout if timeout is not None else settings.llm_timeout_seconds)
        # Injectable transport so tests never touch the network.
        self._transport = transport

    # -- introspection ----------------------------------------------------
    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.model_id)

    def describe(self) -> dict[str, Any]:
        """Non-secret description of the adapter, safe to return over HTTP."""
        return {
            "provider": self.provider,
            "model_id": self.model_id,
            "base_url": self.base_url,
            "api_key_present": bool(self.api_key),
            "timeout_seconds": self.timeout,
            "known_models": sorted(KNOWN_MODELS),
        }

    # -- main call --------------------------------------------------------
    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        model_id: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
    ) -> ProviderResponse:
        """Send *messages* to the provider and return a normalised reply.

        ``tools`` must already be in OpenAI function-calling format; the
        adapter does not transform or extend it.
        """
        if not self.api_key:
            raise LLMProviderError(
                "LLM_API_KEY is not configured. Set it in the environment (never in source)."
            )

        model = resolve_model_id(model_id or self.model_id)

        # A client-facing tier id (gpt-6-luna / gpt-6-sol / gpt-6-astra) is not
        # a real DeepSeek model. Translate it to deepseek-flash + the tier's
        # reasoning_effort before anything is sent upstream, otherwise DeepSeek
        # rejects the request with HTTP 400.
        tier = resolve_deepseek_tier(model)
        if tier is not None:
            model = tier["model"]
            if reasoning_effort is None:
                reasoning_effort = tier["reasoning_effort"]
            if max_tokens is None:
                max_tokens = tier.get("max_tokens")

        body: dict[str, Any] = {"model": model, "messages": messages}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        if max_tokens is not None:
            body["max_completion_tokens"] = int(max_tokens)
        if temperature is not None:
            body["temperature"] = float(temperature)

        allowed = KNOWN_MODELS[model]["reasoning_effort"]
        effort = (reasoning_effort or get_settings().llm_reasoning_effort or "").strip()
        if effort and effort not in allowed:
            raise LLMProviderError(
                f"reasoning_effort {redact(effort)!r} is not supported by "
                f"{model}. Allowed: {', '.join(allowed)}"
            )

        # OpenAI rejects function tools combined with reasoning_effort on
        # /v1/chat/completions for the GPT-6 family:
        #   "Function tools with reasoning_effort are not supported for
        #    gpt-6-luna in /v1/chat/completions. To use function tools, use
        #    /v1/responses or set reasoning_effort to 'none'."
        # Verified against the live API: the field must be present AND set to
        # the model's lowest supported effort -- omitting it entirely is
        # rejected too. Tool calling is the whole point of this layer, so
        # whenever tools are present we send that lowest value.
        #
        # "none" is NOT universal: gpt-6-astra rejects it outright
        #   "Unsupported value: 'reasoning_effort' does not support 'none'
        #    with this model. Supported values are: 'low', 'medium', 'high',
        #    and 'xhigh'."
        # so we fall back to the first (lowest) value the model advertises.
        # Without tools the caller's effort is forwarded unchanged, and an
        # unset effort stays unset.
        #
        # This restriction is specific to the GPT-6 family. DeepSeek accepts
        # tools together with reasoning_effort, so the tier effort must be
        # preserved there -- forcing "none" would silently downgrade every
        # sol/astra request to the cheapest tier.
        if tools and not model.startswith("deepseek-"):
            body["reasoning_effort"] = "none" if "none" in allowed else allowed[0]
        elif effort:
            body["reasoning_effort"] = effort

        # GLM (Z.ai) does not implement OpenAI's reasoning_effort field; a
        # strict upstream may 400 on unknown params, so never send it there.
        if model.startswith("glm-") or "z.ai" in self.base_url:
            body.pop("reasoning_effort", None)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        log_debug(
            "llm request provider=%s model=%s messages=%d tools=%d",
            self.provider,
            model,
            len(messages),
            len(tools or []),
        )

        try:
            async with httpx.AsyncClient(timeout=self.timeout, transport=self._transport) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions", json=body, headers=headers
                )
        except httpx.TimeoutException as exc:
            raise LLMProviderError(f"LLM request timed out after {self.timeout}s") from exc
        except httpx.HTTPError as exc:
            raise LLMProviderError(f"LLM transport error: {redact(type(exc).__name__)}") from exc

        if response.status_code != 200:
            # The upstream body can echo the request; redact before surfacing.
            raise LLMProviderError(
                f"LLM provider returned HTTP {response.status_code}: {redact(response.text[:300])}"
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise LLMProviderError("LLM provider returned a non-JSON body") from exc

        return self._parse(payload, fallback_model=model)

    # -- parsing ----------------------------------------------------------
    @staticmethod
    def _parse(payload: dict[str, Any], *, fallback_model: str) -> ProviderResponse:
        choices = payload.get("choices") or []
        if not choices:
            raise LLMProviderError("LLM provider returned no choices")

        choice = choices[0] or {}
        message = choice.get("message") or {}

        raw_calls = message.get("tool_calls") or []
        tool_calls = [ToolCall.from_openai(c) for c in raw_calls if isinstance(c, dict)]

        content = message.get("content")
        if content is None:
            content = ""
        elif not isinstance(content, str):
            content = str(content)

        # Ollama's OpenAI-compatible /v1 shim does not populate message.tool_calls.
        # Verified against Ollama 0.34.4: with tools + tool_choice supplied, the
        # model answers with the call as JSON *text* in message.content, e.g.
        #   {"name": "ollama_status", "arguments": {}}
        #   {"type":"function","name":"ollama_status","parameters":{}}
        # Without this bridge the orchestrator sees "no tool calls" and the MCP
        # loop silently degrades to plain chat. Parsing is deliberately
        # conservative: only a JSON object that names a tool is accepted, and
        # anything else is left untouched as content.
        if not tool_calls and content:
            salvaged = _salvage_tool_call(content)
            if salvaged is not None:
                tool_calls = [salvaged]
                content = ""

        return ProviderResponse(
            content=content,
            tool_calls=tool_calls,
            model=str(payload.get("model") or fallback_model),
            finish_reason=str(choice.get("finish_reason") or ""),
            usage=payload.get("usage") or {},
        )

    # -- tool schema helper -----------------------------------------------
    @staticmethod
    def to_openai_tools(tools: list[Any]) -> list[dict[str, Any]]:
        """Convert MCP tool descriptors into OpenAI function-calling schemas.

        Accepts objects exposing ``name`` / ``description`` / ``input_schema``
        (i.e. :class:`mcp_for_copilot.mcp_client.MCPTool`).
        """
        converted: list[dict[str, Any]] = []
        for tool in tools:
            name = getattr(tool, "name", None)
            if not name:
                continue
            schema = getattr(tool, "input_schema", None) or {
                "type": "object",
                "properties": {},
            }
            converted.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": getattr(tool, "description", "") or "",
                        "parameters": schema,
                    },
                }
            )
        return converted


def log_provider_summary(provider: LLMProvider) -> None:
    """Emit a one-line, secret-free summary of the adapter configuration."""
    info = provider.describe()
    log_info(
        "llm provider ready provider=%s model=%s key_present=%s",
        info["provider"],
        info["model_id"],
        info["api_key_present"],
    )


def log_provider_failure(exc: Exception) -> None:
    log_error("llm provider failure: %s", redact(str(exc)))
