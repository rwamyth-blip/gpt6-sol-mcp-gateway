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
from .logging_utils import log_debug, log_error, log_info, log_warning, redact

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
    "glm-4.7": {
        "label": "GLM-4.7 (Z.ai)",
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
    "glm-4.7-flash": "glm-4.7",
    "glm47": "glm-4.7",
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


SUPPORTED_PROVIDERS = ("openai", "deepseek")


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
        fallback_api_key: str | None = None,
        fallback_base_url: str | None = None,
        fallback_model_id: str | None = None,
        fallback_enabled: bool | None = None,
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

        # -- Z.ai (GLM) backup provider -----------------------------------
        # Used only when the primary request fails with a transport error,
        # timeout, or 5xx. A 4xx is a caller/request problem, not an outage,
        # so it is never retried against the fallback.
        self.fallback_enabled = (
            settings.llm_fallback_enabled if fallback_enabled is None else fallback_enabled
        )
        self.fallback_api_key = (
            fallback_api_key if fallback_api_key is not None else settings.llm_fallback_api_key
        ).strip()
        self.fallback_base_url = (
            fallback_base_url
            if fallback_base_url is not None
            else settings.llm_fallback_base_url
        ).strip().rstrip("/")
        self.fallback_model_id = (
            fallback_model_id
            if fallback_model_id is not None
            else settings.llm_fallback_model_id
        ).strip()
        self.fallback_timeout = float(settings.llm_fallback_timeout_seconds)

    # -- introspection ----------------------------------------------------
    def _uses_deepseek(self) -> bool:
        """True when the primary upstream is DeepSeek (or a DeepSeek host).

        The tier table in :data:`DEEPSEEK_TIERS` maps the client-facing
        gpt-6-* ids onto ``deepseek-flash``. That is only correct when the
        request is actually going to DeepSeek; for OpenAI the same ids are
        real models and must be forwarded verbatim.
        """
        if self.provider == "deepseek":
            return True
        return "deepseek" in self.base_url.lower()

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.model_id)

    @property
    def fallback_configured(self) -> bool:
        """True when the Z.ai backup is usable (enabled + key + model)."""
        return bool(self.fallback_enabled and self.fallback_api_key and self.fallback_model_id)

    @property
    def _uses_zai(self) -> bool:
        """True when the primary upstream already IS Z.ai (or a Z.ai host)."""
        if self.provider == "zai":
            return True
        return "z.ai" in self.base_url.lower()

    def _glm_route(self) -> tuple[str, str, str] | None:
        """Return ``(base_url, api_key, timeout)`` for GLM model ids.

        GLM (glm-*) models only exist on Z.ai. When the primary upstream is
        NOT Z.ai (e.g. this deployment is OpenAI-primary), sending glm-* there
        is a guaranteed 404 "model does not exist" -- and 4xx never triggers
        the fallback. Instead the request is routed straight to the Z.ai
        credentials already configured for the fallback, keeping one key per
        upstream. Returns None when the primary already serves glm-* itself.
        """
        if self._uses_zai:
            return None
        if not (self.fallback_api_key and self.fallback_base_url):
            return None
        return (self.fallback_base_url, self.fallback_api_key, self.fallback_timeout)

    def describe(self) -> dict[str, Any]:
        """Non-secret description of the adapter, safe to return over HTTP."""
        return {
            "provider": self.provider,
            "model_id": self.model_id,
            "base_url": self.base_url,
            "api_key_present": bool(self.api_key),
            "timeout_seconds": self.timeout,
            "fallback_enabled": self.fallback_enabled,
            "fallback_configured": self.fallback_configured,
            "fallback_base_url": self.fallback_base_url,
            "fallback_model_id": self.fallback_model_id,
            "fallback_api_key_present": bool(self.fallback_api_key),
            "known_models": sorted(KNOWN_MODELS),
        }

    # -- request building -------------------------------------------------
    @staticmethod
    def _build_body(
        model: str,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None,
        max_tokens: int | None,
        temperature: float | None,
        reasoning_effort: str | None,
    ) -> dict[str, Any]:
        """Assemble the chat-completions body for *model*.

        Shared by the primary and fallback paths so both send an identically
        shaped request. ``model`` must already be a canonical id present in
        :data:`KNOWN_MODELS`.
        """
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
        #
        # gpt-6-astra is the one model that supports NEITHER combination:
        # it rejects 'none' outright, and it rejects any other effort while
        # tools are present. There is no valid (tools, effort) pair for it on
        # /v1/chat/completions, so when tools are requested we drop them and
        # keep the model's lowest real effort. The caller still gets a valid
        # answer; only the tool-calling loop is unavailable on this tier.
        if tools and not model.startswith("deepseek-"):
            if "none" in allowed:
                body["reasoning_effort"] = "none"
            else:
                body.pop("tools", None)
                body.pop("tool_choice", None)
                body["reasoning_effort"] = allowed[0]
        elif effort:
            body["reasoning_effort"] = effort

        # GLM (Z.ai) does not implement OpenAI's reasoning_effort field; a
        # strict upstream may 400 on unknown params, so never send it there.
        # NOTE: this is a @staticmethod, so it cannot inspect self.base_url --
        # the model id is the only signal available here.
        if model.startswith("glm-"):
            body.pop("reasoning_effort", None)

        return body

    async def _post(
        self,
        *,
        base_url: str,
        api_key: str,
        body: dict[str, Any],
        timeout: float,
    ) -> ProviderResponse:
        """POST *body* to ``<base_url>/chat/completions`` and parse the reply.

        Raises :class:`LLMProviderError` on any transport failure, timeout, or
        non-200 response. The caller decides whether that is fatal or should
        trigger the fallback.
        """
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        try:
            async with httpx.AsyncClient(timeout=timeout, transport=self._transport) as client:
                response = await client.post(
                    f"{base_url}/chat/completions", json=body, headers=headers
                )
        except httpx.TimeoutException as exc:
            raise LLMProviderError(f"LLM request timed out after {timeout}s") from exc
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

        return self._parse(payload, fallback_model=str(body.get("model") or ""))

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

        When the primary provider fails with a transport error, timeout, or
        5xx and a Z.ai fallback is configured, the same request is retried
        once against the fallback endpoint. A 4xx is never retried: it is a
        request problem, not an outage.
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
        #
        # This translation is ONLY valid when the upstream actually IS
        # DeepSeek. gpt-6-luna / gpt-6-sol / gpt-6-astra are also real OpenAI
        # model ids, so applying the table unconditionally rewrites them to
        # `deepseek-flash` and sends that to OpenAI, which answers
        #   404 "The model `deepseek-flash` does not exist"
        # and the gateway surfaces a 502. Gate on the provider (and on the
        # base_url as a belt-and-braces check for a DeepSeek-compatible host).
        if self._uses_deepseek():
            tier = resolve_deepseek_tier(model)
            if tier is not None:
                model = tier["model"]
                if reasoning_effort is None:
                    reasoning_effort = tier["reasoning_effort"]
                if max_tokens is None:
                    max_tokens = tier.get("max_tokens")

        body = self._build_body(
            model,
            messages,
            tools=tools,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
        )

        # Route glm-* to Z.ai when the primary upstream is not Z.ai -- the
        # primary would 404 on every glm id and 4xx never hits the fallback.
        glm_route = self._glm_route() if model.startswith("glm-") else None
        if glm_route is not None:
            route_url, route_key, route_timeout = glm_route
            log_debug(
                "llm request provider=zai-primary-override model=%s messages=%d tools=%d",
                model,
                len(messages),
                len(tools or []),
            )
            return await self._post(
                base_url=route_url,
                api_key=route_key,
                body=body,
                timeout=route_timeout,
            )

        log_debug(
            "llm request provider=%s model=%s messages=%d tools=%d",
            self.provider,
            model,
            len(messages),
            len(tools or []),
        )

        try:
            return await self._post(
                base_url=self.base_url,
                api_key=self.api_key,
                body=body,
                timeout=self.timeout,
            )
        except LLMProviderError as primary_error:
            if not self.fallback_configured or not self._is_retryable(primary_error):
                raise
            log_warning(
                "primary LLM failed (%s); retrying on Z.ai fallback model=%s",
                redact(str(primary_error)),
                self.fallback_model_id,
            )
            return await self._complete_fallback(
                messages,
                tools=tools,
                max_tokens=max_tokens,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                primary_error=primary_error,
            )

    @staticmethod
    def _is_retryable(error: LLMProviderError) -> bool:
        """True when *error* looks like an outage rather than a bad request.

        Transport errors and timeouts are always retryable. HTTP 5xx is
        retryable; HTTP 4xx is not -- the fallback would fail the same way and
        the caller needs to see the real validation error.
        """
        message = str(error)
        if "timed out" in message or "transport error" in message:
            return True
        return "HTTP 5" in message

    async def _complete_fallback(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None,
        max_tokens: int | None,
        temperature: float | None,
        reasoning_effort: str | None,
        primary_error: LLMProviderError,
    ) -> ProviderResponse:
        """Retry the request against the Z.ai fallback endpoint.

        The fallback model is resolved from the catalog, so an unknown
        ``LLM_FALLBACK_MODEL_ID`` fails loudly instead of being forwarded
        verbatim. If the fallback also fails, the raised error names both
        failures so the operator can see the whole story.
        """
        try:
            fallback_model = resolve_model_id(self.fallback_model_id)
        except LLMProviderError as exc:
            raise LLMProviderError(
                f"primary LLM failed ({redact(str(primary_error))}) and the "
                f"fallback model is invalid: {redact(str(exc))}"
            ) from primary_error

        # GLM does not support reasoning_effort; drop it so the fallback body
        # is valid even when the caller asked for a specific effort.
        fallback_effort = None if fallback_model.startswith("glm-") else reasoning_effort
        body = self._build_body(
            fallback_model,
            messages,
            tools=tools,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=fallback_effort,
        )

        try:
            return await self._post(
                base_url=self.fallback_base_url,
                api_key=self.fallback_api_key,
                body=body,
                timeout=self.fallback_timeout,
            )
        except LLMProviderError as fallback_error:
            raise LLMProviderError(
                f"primary LLM failed ({redact(str(primary_error))}); "
                f"Z.ai fallback also failed ({redact(str(fallback_error))})"
            ) from fallback_error

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
