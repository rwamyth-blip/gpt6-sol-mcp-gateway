"""Integration tests for the FastAPI gateway routes."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from mcp_for_copilot.config import Settings
from mcp_for_copilot.gateway.app import create_app
from mcp_for_copilot.gateway.facade import Gateway
from mcp_for_copilot.provider import LLMProvider

from ..conftest import FakeMCPClient, make_completion, make_tool_call, mock_transport


def _client(
    responses,
    *,
    settings: Settings | None = None,
    fake_client: FakeMCPClient | None = None,
    capture: list[dict] | None = None,
) -> TestClient:
    resolved = settings or Settings(
        llm_api_key="sk-test",
        llm_model_id="gpt-6-sol",
        llm_base_url="https://api.example.test/v1",
        mcp_server_url="",
        gateway_api_key="",
    )
    provider = LLMProvider(
        api_key=resolved.llm_api_key,
        base_url=resolved.llm_base_url,
        model_id=resolved.llm_model_id,
        transport=mock_transport(responses, capture=capture),
    )
    gateway = Gateway(settings=resolved, provider=provider, connect_mcp=False)
    gateway.client = fake_client
    gateway.orchestrator.client = fake_client
    gateway._connected = True
    app = create_app(settings=resolved, gateway=gateway)
    return TestClient(app)


class TestHealth:
    def test_health_is_unauthenticated(self) -> None:
        with _client(make_completion()) as client:
            response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


class TestModels:
    def test_lists_every_known_model(self) -> None:
        with _client(make_completion()) as client:
            response = client.get("/v1/models")
        assert response.status_code == 200
        ids = [m["id"] for m in response.json()["data"]]
        assert "gpt-6-sol" in ids
        assert "gpt-6-luna" in ids

    def test_model_metadata_includes_aliases(self) -> None:
        with _client(make_completion()) as client:
            data = client.get("/v1/models").json()["data"]
        sol = next(m for m in data if m["id"] == "gpt-6-sol")
        assert "sol" in sol["metadata"]["aliases"]
        assert sol["metadata"]["context_window"] == 1_050_000

    def test_requires_key_when_configured(self) -> None:
        settings = Settings(
            llm_api_key="sk-test", llm_model_id="gpt-6-sol", gateway_api_key="gw-secret"
        )
        with _client(make_completion(), settings=settings) as client:
            assert client.get("/v1/models").status_code == 401
            ok = client.get("/v1/models", headers={"Authorization": "Bearer gw-secret"})
            assert ok.status_code == 200


class TestStatus:
    def test_status_reports_no_secrets(self) -> None:
        with _client(make_completion()) as client:
            payload = client.get("/v1/status").json()
        assert payload["llm"]["api_key_present"] is True
        assert "sk-test" not in json.dumps(payload)


class TestDebugMarathon:
    def test_submit_and_get_job_status(self, monkeypatch) -> None:
        with _client(make_completion()) as client:
            marathon = client.app.state.gateway.debug_marathon
            monkeypatch.setattr(
                marathon,
                "submit",
                lambda tasks: {
                    "job_id": "job-1",
                    "status": "queued",
                    "total": len(tasks),
                },
            )
            monkeypatch.setattr(
                marathon,
                "get",
                lambda job_id: {"job_id": job_id, "status": "running"},
            )
            submitted = client.post(
                "/v1/debug/marathon",
                json={"tasks": [{"question": "Find the crash", "priority": 5}]},
            )
            status = client.get("/v1/debug/marathon/job-1")

        assert submitted.status_code == 202
        assert submitted.json() == {"job_id": "job-1", "status": "queued", "total": 1}
        assert status.status_code == 200
        assert status.json() == {"job_id": "job-1", "status": "running"}

    def test_debug_marathon_is_authenticated_and_validates_scores(self) -> None:
        settings = Settings(
            llm_api_key="sk-test",
            llm_model_id="gpt-6-sol",
            gateway_api_key="gw-secret",
        )
        with _client(make_completion(), settings=settings) as client:
            url = "/v1/debug/marathon"
            assert client.post(url, json={"tasks": [{"question": "x"}]}).status_code == 401
            response = client.post(
                url,
                headers={"Authorization": "Bearer gw-secret"},
                json={"tasks": [{"question": "x", "difficulty": 6}]},
            )

        assert response.status_code == 422


class TestChatCompletions:
    def test_plain_completion(self) -> None:
        with _client(make_completion(content="Hi there")) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-6-sol",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )
        assert response.status_code == 200
        body = response.json()
        assert body["object"] == "chat.completion"
        assert body["choices"][0]["message"]["content"] == "Hi there"
        assert body["usage"]["total_tokens"] > 0

    def test_alias_is_accepted(self) -> None:
        with _client(make_completion()) as client:
            response = client.post(
                "/v1/chat/completions",
                json={"model": "sol", "messages": [{"role": "user", "content": "hi"}]},
            )
        assert response.status_code == 200

    def test_unknown_model_is_400(self) -> None:
        with _client(make_completion()) as client:
            response = client.post(
                "/v1/chat/completions",
                json={"model": "gpt-9-nope", "messages": [{"role": "user", "content": "hi"}]},
            )
        assert response.status_code == 400
        assert "Unknown model id" in response.json()["detail"]

    def test_missing_messages_is_400(self) -> None:
        with _client(make_completion()) as client:
            response = client.post("/v1/chat/completions", json={"model": "gpt-6-sol"})
        assert response.status_code == 400

    def test_empty_messages_is_400(self) -> None:
        with _client(make_completion()) as client:
            response = client.post(
                "/v1/chat/completions", json={"model": "gpt-6-sol", "messages": []}
            )
        assert response.status_code == 400

    def test_non_json_body_is_400(self) -> None:
        with _client(make_completion()) as client:
            response = client.post(
                "/v1/chat/completions",
                content=b"not json",
                headers={"Content-Type": "application/json"},
            )
        assert response.status_code == 400

    def test_system_message_is_extracted(self) -> None:
        captured: list[dict] = []
        with _client(make_completion(), capture=captured) as client:
            client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-6-sol",
                    "messages": [
                        {"role": "system", "content": "Be terse."},
                        {"role": "user", "content": "hi"},
                    ],
                },
            )
        assert "Be terse." in captured[0]["messages"][0]["content"]

    def test_tool_audit_trail_is_returned(self, fake_client: FakeMCPClient) -> None:
        responses = [
            make_completion(
                content="",
                tool_calls=[make_tool_call("read_file", {"path": "a.txt"})],
                finish_reason="tool_calls",
            ),
            make_completion(content="The file is empty."),
        ]
        with _client(responses, fake_client=fake_client) as client:
            body = client.post(
                "/v1/chat/completions",
                json={"model": "gpt-6-sol", "messages": [{"role": "user", "content": "read"}]},
            ).json()
        assert body["x_gateway"]["used_tools"] == ["read_file"]
        assert body["x_gateway"]["rounds"] == 2

    def test_provider_error_is_502(self) -> None:
        import httpx

        settings = Settings(
            llm_api_key="sk-test",
            llm_model_id="gpt-6-sol",
            llm_base_url="https://api.example.test/v1",
            gateway_api_key="",
        )
        provider = LLMProvider(
            api_key="sk-test",
            base_url=settings.llm_base_url,
            model_id="gpt-6-sol",
            transport=httpx.MockTransport(
                lambda request: httpx.Response(500, text="upstream down")
            ),
        )
        gateway = Gateway(settings=settings, provider=provider, connect_mcp=False)
        gateway._connected = True
        app = create_app(settings=settings, gateway=gateway)
        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={"model": "gpt-6-sol", "messages": [{"role": "user", "content": "hi"}]},
            )
        assert response.status_code == 502


class TestStreaming:
    def test_stream_emits_openai_chunks(self) -> None:
        with (
            _client(make_completion(content="Hello world")) as client,
            client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": "gpt-6-sol",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            ) as response,
        ):
            assert response.status_code == 200
            assert "text/event-stream" in response.headers["content-type"]
            body = "".join(response.iter_text())

        assert body.startswith("data: ")
        assert body.rstrip().endswith("data: [DONE]")
        chunks = [
            json.loads(line[6:])
            for line in body.splitlines()
            if line.startswith("data: ") and line[6:] != "[DONE]"
        ]
        assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
        text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
        assert text == "Hello world"

    def test_stream_reports_provider_error(self) -> None:
        import httpx

        settings = Settings(
            llm_api_key="sk-test",
            llm_model_id="gpt-6-sol",
            llm_base_url="https://api.example.test/v1",
            gateway_api_key="",
        )
        provider = LLMProvider(
            api_key="sk-test",
            base_url=settings.llm_base_url,
            model_id="gpt-6-sol",
            transport=httpx.MockTransport(
                lambda request: httpx.Response(500, text="upstream down")
            ),
        )
        gateway = Gateway(settings=settings, provider=provider, connect_mcp=False)
        gateway._connected = True
        app = create_app(settings=settings, gateway=gateway)
        with (
            TestClient(app) as client,
            client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": "gpt-6-sol",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            ) as response,
        ):
            body = "".join(response.iter_text())
        assert "provider_error" in body
        assert body.rstrip().endswith("data: [DONE]")


class TestTokenEstimation:
    def test_thai_text_uses_a_denser_ratio(self) -> None:
        from mcp_for_copilot.gateway.app import _estimate_tokens

        thai = _estimate_tokens("สวัสดีครับ")
        latin = _estimate_tokens("hello there")
        assert thai > 0
        assert latin > 0

    def test_empty_text_is_zero(self) -> None:
        from mcp_for_copilot.gateway.app import _estimate_tokens

        assert _estimate_tokens("") == 0

    def test_usage_prefers_provider_values(self) -> None:
        from mcp_for_copilot.gateway.app import _usage

        raw = {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
        assert _usage("a", "b", raw) == raw
