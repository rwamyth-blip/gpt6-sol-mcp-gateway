"""FastAPI app — an OpenAI-compatible gateway in front of GPT-6 Sol + MCP.

Endpoints
---------
``GET  /health``
    Liveness probe. Always 200 when the process is up.
``GET  /v1/models``
    OpenAI-compatible model list.
``POST /v1/chat/completions``
    OpenAI-compatible chat completions, with optional SSE streaming.
``GET  /v1/status``
    Non-secret configuration snapshot.

Authentication
--------------
When ``GATEWAY_API_KEY`` is set, every ``/v1/*`` request must carry
``Authorization: Bearer <key>``. When it is empty the gateway accepts any key,
which is intended for local development only.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import APIRouter, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..config import Settings, get_settings
from ..logging_utils import log_info, log_warning, redact
from ..provider import KNOWN_MODELS, MODEL_ALIASES, LLMProviderError, resolve_model_id
from .facade import Gateway

# Rough token estimate. Thai script packs ~2.5 characters per token, latin
# text ~4. Used only for the usage block when the provider omits one.
_THAI_RANGE = (0x0E00, 0x0E7F)


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    thai = sum(1 for ch in text if _THAI_RANGE[0] <= ord(ch) <= _THAI_RANGE[1])
    latin = len(text) - thai
    return max(1, int(thai / 2.5 + latin / 4))


def _usage(prompt: str, completion: str, raw: dict[str, Any] | None = None) -> dict[str, Any]:
    if raw:
        return raw
    prompt_tokens = _estimate_tokens(prompt)
    completion_tokens = _estimate_tokens(completion)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _messages_to_text(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(str(block.get("text") or ""))
    return "\n".join(parts)


class DebugMarathonTaskRequest(BaseModel):
    title: str | None = Field(default=None, max_length=200)
    question: str = Field(min_length=1, max_length=10000)
    priority: int = Field(default=3, ge=1, le=5)
    difficulty: int = Field(default=3, ge=1, le=5)
    complexity: int = Field(default=3, ge=1, le=5)


class DebugMarathonRequest(BaseModel):
    tasks: list[DebugMarathonTaskRequest] = Field(min_length=1, max_length=10)


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def create_app(
    *,
    settings: Settings | None = None,
    gateway: Gateway | None = None,
) -> FastAPI:
    """Build the FastAPI app. *gateway* is injectable for tests."""
    resolved = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.gateway = gateway or Gateway(settings=resolved)
        await app.state.gateway.start()
        app.state.gateway.log_summary()
        try:
            yield
        finally:
            await app.state.gateway.stop()

    app = FastAPI(
        title="MCP for Copilot",
        version="0.1.0",
        description=(
            "OpenAI-compatible gateway that gives GPT-6 Sol tool access "
            "through the Model Context Protocol."
        ),
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=resolved.gateway_cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    router = APIRouter()

    def _check_auth(authorization: str | None) -> None:
        expected = resolved.gateway_api_key.strip()
        if not expected:
            return
        token = ""
        if authorization and authorization.lower().startswith("bearer "):
            token = authorization[7:].strip()
        if token != expected:
            raise HTTPException(status_code=401, detail="Invalid API key")

    def _get_gateway(request: Request) -> Gateway:
        current: Gateway | None = getattr(request.app.state, "gateway", None)
        if current is None:
            raise HTTPException(status_code=503, detail="Gateway is not ready")
        return current

    @router.post("/debug/marathon", status_code=202)
    async def submit_debug_marathon(
        payload: DebugMarathonRequest,
        request: Request,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        _check_auth(authorization)
        try:
            return _get_gateway(request).debug_marathon.submit(
                [task.model_dump() for task in payload.tasks]
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @router.get("/debug/marathon/{job_id}")
    async def get_debug_marathon(
        job_id: str,
        request: Request,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        _check_auth(authorization)
        job = _get_gateway(request).debug_marathon.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Debug marathon job not found")
        return job

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "service": "mcp-for-copilot"}

    @router.get("/models")
    async def list_models(
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        _check_auth(authorization)
        created = int(time.time())
        data = [
            {
                "id": model_id,
                "object": "model",
                "created": created,
                "owned_by": "openai",
                "metadata": {
                    "label": spec["label"],
                    "tier": spec["tier"],
                    "context_window": spec["context_window"],
                    "max_output": spec["max_output"],
                    "reasoning_effort": list(spec["reasoning_effort"]),
                    "aliases": sorted(
                        a for a, target in MODEL_ALIASES.items() if target == model_id
                    ),
                },
            }
            for model_id, spec in sorted(KNOWN_MODELS.items())
        ]
        return {"object": "list", "data": data}

    @router.get("/status")
    async def status(
        request: Request,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        _check_auth(authorization)
        return _get_gateway(request).status()

    @router.post("/chat/completions")
    async def chat_completions(
        request: Request,
        authorization: str | None = Header(default=None),
    ) -> Any:
        _check_auth(authorization)
        try:
            body = await request.json()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Body must be JSON") from exc
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="Body must be a JSON object")

        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise HTTPException(status_code=400, detail="'messages' must be a non-empty list")

        try:
            model_id = resolve_model_id(body.get("model"))
        except LLMProviderError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        gateway_instance = _get_gateway(request)
        system = None
        conversation: list[dict[str, Any]] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            if message.get("role") == "system":
                system = str(message.get("content") or "")
            else:
                conversation.append(message)

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        if body.get("stream"):
            return StreamingResponse(
                _stream(
                    gateway_instance,
                    conversation,
                    system=system,
                    model_id=model_id,
                    completion_id=completion_id,
                    created=created,
                ),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        try:
            result = await gateway_instance.chat(conversation, system=system, model_id=model_id)
        except LLMProviderError as exc:
            log_warning("chat failed: %s", redact(str(exc)))
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        if result.error:
            log_warning("chat failed: %s", redact(result.error))
            raise HTTPException(status_code=502, detail=result.error)

        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": result.model or model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result.content},
                    "finish_reason": "stop",
                }
            ],
            "usage": _usage(_messages_to_text(conversation), result.content, result.usage),
            "x_gateway": {
                "rounds": result.rounds,
                "used_tools": result.used_tools,
                "invocations": result.invocations,
                "plan": result.plan,
                "plan_progress": result.plan_progress,
                "loop_stopped": result.loop_stopped,
            },
        }

    app.include_router(router, prefix="/v1")
    return app


async def _stream(
    gateway: Gateway,
    messages: list[dict[str, Any]],
    *,
    system: str | None,
    model_id: str,
    completion_id: str,
    created: int,
) -> AsyncIterator[str]:
    """Yield an OpenAI-compatible SSE stream.

    The orchestrator is not incremental, so the answer is produced first and
    then chunked. Chunking keeps the wire format identical to OpenAI's, which
    is what client libraries expect.
    """

    def _chunk(delta: dict[str, Any], finish: str | None = None) -> str:
        return _sse(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_id,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
        )

    yield _chunk({"role": "assistant", "content": ""})

    try:
        result = await gateway.chat(messages, system=system, model_id=model_id)
    except LLMProviderError as exc:
        log_warning("stream failed: %s", redact(str(exc)))
        yield _sse({"error": {"message": str(exc), "type": "provider_error"}})
        yield "data: [DONE]\n\n"
        return

    if result.error:
        log_warning("stream failed: %s", redact(result.error))
        yield _sse({"error": {"message": result.error, "type": "provider_error"}})
        yield "data: [DONE]\n\n"
        return

    chunk_chars = 220
    text = result.content or ""
    for start in range(0, len(text), chunk_chars):
        yield _chunk({"content": text[start : start + chunk_chars]})

    yield _chunk({}, finish="stop")
    yield "data: [DONE]\n\n"
    log_info(
        "stream complete model=%s chars=%d tools=%d",
        result.model or model_id,
        len(text),
        len(result.used_tools),
    )


app = create_app()
