# Server Usage

The gateway ships an OpenAI-compatible HTTP API, so any OpenAI SDK client works against it unchanged.

## Start the server

```bash
mcp-for-copilot serve --host 0.0.0.0 --port 8080
```

Or programmatically:

```python
import uvicorn
from mcp_for_copilot.gateway.app import create_app

app = create_app()
uvicorn.run(app, host="0.0.0.0", port=8080)
```

## Routes

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Liveness probe |
| `GET` | `/v1/models` | OpenAI-shaped model list |
| `GET` | `/v1/status` | Configuration summary (booleans only) |
| `POST` | `/v1/chat/completions` | Chat completions, streaming and non-streaming |
| `POST` | `/v1/debug/marathon` | Queue up to 10 prioritized debug tasks (202) |
| `GET` | `/v1/debug/marathon/{job_id}` | Read queue, progress, and results |

### `GET /health`

```json
{"status": "ok", "service": "mcp-for-copilot"}
```

### `GET /v1/models`

```json
{
  "object": "list",
  "data": [
    {"id": "gpt-6-astra", "object": "model", "owned_by": "openai"},
    {"id": "gpt-6-sol", "object": "model", "owned_by": "openai"},
    {"id": "gpt-6-luna", "object": "model", "owned_by": "openai"},
    {"id": "gpt-5.6-sol", "object": "model", "owned_by": "openai"},
    {"id": "gpt-5.6-luna", "object": "model", "owned_by": "openai"}
  ]
}
```

### `POST /v1/chat/completions`

```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "gpt-6-sol",
        "messages": [{"role": "user", "content": "Hello!"}],
        "stream": false
      }'
```

Streaming:

```bash
curl -N http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "gpt-6-sol",
        "messages": [{"role": "user", "content": "Count to five."}],
        "stream": true
      }'
```

### `x_gateway` metadata

Non-streaming responses carry an `x_gateway` object describing what the orchestrator did:

| Field | Type | Meaning |
| --- | --- | --- |
| `rounds` | `int` | Number of model round-trips |
| `used_tools` | `bool` | Whether any tool was executed |
| `invocations` | `list` | Every tool call attempted, with allow/approve/execute flags |
| `plan` | `list` | The model's working plan, empty unless `MCP_PLAN_MODE` is on |
| `plan_progress` | `dict` | Count of plan steps per status |

## Debug marathon

Submit up to 10 tasks. Each score is an integer from 1 to 5. Tasks are ordered by priority,
difficulty, then complexity, highest first. The response includes a `job_id`; poll the status route
to follow the sequential run.

```bash
curl -s http://localhost:8080/v1/debug/marathon \
  -H "Content-Type: application/json" \
  -d '{
        "tasks": [
          {"title": "Simple regression", "question": "Find the null handling bug", "priority": 3, "difficulty": 2, "complexity": 2},
          {"title": "Race condition", "question": "Diagnose the intermittent write race", "priority": 5, "difficulty": 5, "complexity": 5}
        ]
      }'

curl -s http://localhost:8080/v1/debug/marathon/<job_id>
```

The target task mix is 80% Luna-only (difficulty/complexity 1-2), 15% Luna then Sol (3-4), and 5%
the full Luna -> Sol -> Astra ladder (5). These are target proportions, not enforced quotas; uncertain
outputs and provider failures escalate to the next model. At equal token volume, catalog prices are
approximately 1:20:100 for Luna:Sol:Astra, so this target mix is about 10 Luna-cost units per task
versus 121 units when every task uses all three stages. Actual spend depends on prompt and answer
lengths and the incoming task mix.

`completed` means the final model self-reported `SOLVED`; the gateway does not execute code or tests.
The status response reports `processed`, `completed`, `model_reported_success_rate`, token usage, and
catalog-based `estimated_cost_usd`. Measure accepted fixes or passing tests on a representative
benchmark before claiming an actual success rate above 80%. Job state is in process memory and is
lost when the gateway restarts.

Each task and the job itself also carry a `verification` field. It is `not_executed` unless a stage was
actually handed execution output, in which case it is `executed`. The default is deliberately
pessimistic: a model claiming success does **not** set `executed`. Tasks additionally carry a `plan`
list parsed from the model's `MARATHON_PLAN` line.

## Status codes

| Code | When |
| --- | --- |
| `200` | Success |
| `202` | Debug marathon accepted and queued |
| `400` | `messages` missing, empty, or not a list; unknown model |
| `401` | Gateway API key is missing or invalid |
| `404` | Debug marathon job ID is unknown or no longer retained |
| `422` | Invalid debug marathon payload or score |
| `502` | The upstream provider failed (see `GatewayResult.error`) |

A provider failure is reported as `502` rather than `500` because the gateway itself is healthy — the
upstream is not.

## Docker

### Build

```bash
docker build -t mcp-for-copilot:latest .
```

The image is multi-stage: a builder stage compiles the wheel, and a slim runtime stage installs it as a
non-root user (uid `10001`) with a `HEALTHCHECK` against `/health`.

### Run

```bash
docker run --rm -p 8080:8080 \
  -e LLM_API_KEY="sk-..." \
  -e LLM_MODEL_ID="gpt-6-sol" \
  mcp-for-copilot:latest
```

### Compose

```bash
docker compose up --build
```

`docker-compose.yml` defines two services:

| Service | Role |
| --- | --- |
| `gateway` | The FastAPI server on port 8080 |
| `mcp` | A stdio MCP server the gateway connects to |

Both declare healthchecks, and `gateway` waits for `mcp` to become healthy before starting.

## Reverse proxy notes

- Streaming uses `text/event-stream`. Disable response buffering in nginx
  (`proxy_buffering off;`) or the stream will arrive in one burst.
- Set generous read timeouts. A tool-calling round trip can take tens of seconds.
- The gateway does not terminate TLS. Put it behind a proxy that does.

## Production checklist

- [ ] `LLM_API_KEY` injected from a secret manager, never baked into the image
- [ ] `MCP_ALLOWED_TOOLS` restricted to the minimum set the workload needs
- [ ] An approver configured, or risky tools removed from the allowlist entirely
- [ ] `MCP_MAX_TOOL_ROUNDS` set low enough to bound cost
- [ ] Logs shipped to a collector that respects the built-in redaction
- [ ] `/health` wired to your orchestrator's liveness probe
