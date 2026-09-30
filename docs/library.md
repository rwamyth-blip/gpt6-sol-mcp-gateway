# Library Usage

The library surface is the `Gateway` facade plus the lower-level building blocks, all re-exported from
the top-level package.

```python
from mcp_for_copilot import Gateway, LLMMCPOrchestrator, ToolRouter, ApprovalLayer
```

## The `Gateway` facade

`Gateway` owns the whole stack: settings, provider, MCP client, router, approval layer, and
orchestrator. Use it as an async context manager so the MCP connection is closed deterministically.

```python
import asyncio
from mcp_for_copilot import Gateway

async def main() -> None:
    async with Gateway(connect_mcp=False) as gateway:
        result = await gateway.chat([{"role": "user", "content": "Hello!"}])
        print(result.content)

asyncio.run(main())
```

### Constructor arguments

| Argument | Type | Default | Meaning |
| --- | --- | --- | --- |
| `settings` | `Settings \| None` | `None` | Explicit settings; otherwise loaded from env |
| `provider` | `LLMProvider \| None` | `None` | Inject a provider (used in tests) |
| `approver` | `Approver \| None` | `None` | Approval callback; `None` means deny all risky calls |
| `connect_mcp` | `bool` | `True` | Set `False` for pure-LLM use |

The MCP endpoint, transport, and allowlist come from `Settings` (`MCP_SERVER_URL`,
`MCP_TRANSPORT`, `MCP_ALLOWED_TOOLS`), not from constructor keywords.

### Methods

| Method | Returns | Notes |
| --- | --- | --- |
| `await gateway.chat(messages, *, system=None, model_id=None, max_rounds=None)` | `GatewayResult` | The main entry point |
| `await gateway.start()` | `None` | Called automatically by `__aenter__` |
| `await gateway.stop()` | `None` | Called automatically by `__aexit__` |
| `await gateway.list_tools()` | `list[dict]` | OpenAI-shaped tool schemas |
| `gateway.status()` | `dict` | Booleans only, never credential values |
| `gateway.log_summary()` | `None` | Logs the effective configuration |

### `GatewayResult`

| Field | Type | Meaning |
| --- | --- | --- |
| `content` | `str` | The model's final answer |
| `model` | `str` | Resolved model id |
| `rounds` | `int` | Number of model round-trips |
| `used_tools` | `bool` | Whether any tool was executed |
| `invocations` | `list[ToolInvocation]` | Every tool call attempted |
| `plan` | `list[dict]` | The model's working plan (`{"description", "status"}`), empty unless plan mode is on |
| `plan_progress` | `dict` | Count of steps per status (`pending`/`in_progress`/`done`/`blocked`) |
| `usage` | `dict` | Token accounting when the provider reports it |
| `error` | `str` | Non-empty when the provider failed |

### Plan mode

Set `MCP_PLAN_MODE=true` (or pass `settings` with `mcp_plan_mode=True`) to let the model keep a short
working plan. The orchestrator then advertises a synthetic `update_plan` tool and asks for at most six
steps, each `pending`, `in_progress`, `done`, or `blocked`, with exactly one step `in_progress`.

The plan is **advisory only**. `update_plan` is intercepted inside the orchestrator: it never reaches
the allowlist, the approval layer, or a back-end MCP server, so turning plan mode on cannot widen what
the model may execute. The plan is reset at the start of every `chat()` call, so one turn never
inherits another's plan.

```python
from mcp_for_copilot import Gateway, get_settings

settings = get_settings().model_copy(update={"mcp_plan_mode": True})
async with Gateway(settings=settings) as gateway:
    result = await gateway.chat([{"role": "user", "content": "Refactor the parser"}])

for step in result.plan:
    print(step["status"], step["description"])
print(result.plan_progress)  # {'pending': 2, 'in_progress': 1, 'done': 3, 'blocked': 0}
```

### `ToolInvocation`

| Field | Type | Meaning |
| --- | --- | --- |
| `tool` | `str` | Tool name |
| `arguments` | `dict` | Arguments the model supplied |
| `allowed` | `bool` | Passed the allowlist and risk gates |
| `approved` | `bool` | Passed the approval gate |
| `executed` | `bool` | Actually invoked on the MCP server |
| `is_error` | `bool` | The MCP server reported an error |
| `result` | `str` | Tool output, or the denial reason |
| `reason` | `str` | Why the call was allowed or denied |

## Custom approvers

An approver is any callable that takes an `ApprovalRequest` and returns an `ApprovalDecision`.

```python
from mcp_for_copilot import ApprovalDecision, ApprovalRequest

def my_approver(request: ApprovalRequest) -> ApprovalDecision:
    print(f"Tool: {request.tool}")
    print(f"Args: {request.arguments}")
    print(f"Why:  {request.reason}")
    answer = input("Allow? [y/N] ").strip().lower()
    return ApprovalDecision(approved=answer == "y", reason="operator decision")
```

`ApprovalRequest` carries `tool`, `arguments`, `reason`, and `round_index`.

Three factories ship with the package:

| Factory | Behaviour |
| --- | --- |
| `allow_all_approver()` | Approves everything. Use only in trusted automation. |
| `deny_all_approver()` | Denies everything. Useful for dry runs. |
| `static_approver(allow=(...), deny=(...))` | Keyword-only allow/deny lists by tool name. |

```python
from mcp_for_copilot import static_approver

approver = static_approver(allow=("read_file",), deny=("write_file",))
```

!!! danger "Fail-closed by design"
    If the approver raises, or returns a non-`ApprovalDecision`, the call is **denied**. There is no
    path where an exception in your approver silently grants access.

## Tool routing

`ToolRouter` is usable on its own if you want to reuse the safety logic elsewhere.

```python
from mcp_for_copilot import ToolRouter, is_risky_tool

print(is_risky_tool("delete_file"))   # True
print(is_risky_tool("read_file"))     # False

router = ToolRouter(allowed_tools=("read_file", "list_directory"))
decision = router.route(tool_call, advertised_tools)
print(decision.allowed, decision.reason)
```

Passing an **empty tuple** means "allow nothing". Passing `None` means "use the built-in read-only
defaults". This distinction matters when you want a locked-down deployment.

## Using the orchestrator directly

```python
from mcp_for_copilot import LLMMCPOrchestrator, LLMProvider, MCPClient

provider = LLMProvider()
client = MCPClient(url="python -m my_mcp_server", transport="stdio")
orchestrator = LLMMCPOrchestrator(provider=provider, mcp_client=client)

result = await orchestrator.run([{"role": "user", "content": "List the files."}])
print(result.content, result.error)
```

## Model resolution

```python
from mcp_for_copilot import resolve_model_id, aliases_for, KNOWN_MODELS

print(resolve_model_id("sol"))        # gpt-6-sol
print(resolve_model_id("gpt6"))       # gpt-6-sol
print(aliases_for("gpt-6-sol"))       # ('gpt-6', 'gpt6', 'sol', ...)
print(sorted(KNOWN_MODELS))           # all five model ids
```

## Testing your integration

`Gateway(connect_mcp=False)` plus a hand-assigned fake client is the supported injection pattern:

```python
gateway = Gateway(connect_mcp=False)
gateway.client = FakeMCPClient(tools=[...])
```

This avoids spawning a subprocess in unit tests while still exercising the real orchestrator.
