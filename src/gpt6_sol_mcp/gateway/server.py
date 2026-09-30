"""stdio MCP server — exposes the gateway itself as MCP tools.

Run it directly::

    python -m gpt6_sol_mcp.gateway.server

or let a client spawn it::

    gpt6-mcp-gateway mcp

Tools
-----
``gpt6_chat``
    Send a prompt to GPT-6 Sol and return the answer, plus the working plan
    the model kept (when plan mode is enabled).
``gpt6_models``
    List the models this gateway knows about.
``gpt6_status``
    Report the non-secret gateway configuration.
``gpt6_tools``
    List the MCP tools the gateway is allowed to call.
``gpt6_plan``
    Return the working plan from the most recent ``gpt6_chat`` call.
``gpt6_debug_marathon``
    Queue prioritized debug tasks through GPT-6 Luna, Sol, and Astra.
``gpt6_debug_marathon_status``
    Get a queued debug marathon's progress and results.
``copilot_install``
    Verify (and optionally install) the GitHub Copilot CLI.
``hubspot_track``
    Record a lead through VihokAI's own HubSpot route.
``vihokai_deploy``
    Report on, and optionally push, the VihokAI repository.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import CallToolResult, ListToolsResult, TextContent, Tool

from ..config import get_settings
from ..logging_utils import log_error, log_info, redact
from ..provider import KNOWN_MODELS, MODEL_ALIASES
from .facade import Gateway

SERVER_NAME = "gpt6-sol-mcp-gateway"

_CHAT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "prompt": {
            "type": "string",
            "description": "The user message to send to the model.",
        },
        "system": {
            "type": "string",
            "description": "Optional system instruction.",
        },
        "model": {
            "type": "string",
            "description": "Model id or alias. Defaults to the configured model.",
        },
    },
    "required": ["prompt"],
    "additionalProperties": False,
}

_EMPTY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {},
    "additionalProperties": False,
}

_DEBUG_MARATHON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "tasks": {
            "type": "array",
            "minItems": 1,
            "maxItems": 10,
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "question": {"type": "string"},
                    "priority": {"type": "integer", "minimum": 1, "maximum": 5},
                    "difficulty": {"type": "integer", "minimum": 1, "maximum": 5},
                    "complexity": {"type": "integer", "minimum": 1, "maximum": 5},
                },
                "required": ["question"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["tasks"],
    "additionalProperties": False,
}

_DEBUG_MARATHON_STATUS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"job_id": {"type": "string"}},
    "required": ["job_id"],
    "additionalProperties": False,
}

_COPILOT_INSTALL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "install": {
            "type": "boolean",
            "description": "Run `npm install -g @github/copilot`. Default false (verify only).",
        },
    },
    "additionalProperties": False,
}

_HUBSPOT_TRACK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "message": {"type": "string", "description": "What the lead asked for."},
        "intent": {
            "type": "string",
            "enum": ["pitch", "invest", "api", "support"],
            "description": "pitch / invest / api / support.",
        },
        "email": {"type": "string", "description": "Optional lead email."},
        "site": {"type": "string", "description": "Base URL. Default https://www.vihokai.com"},
    },
    "required": ["message", "intent"],
    "additionalProperties": False,
}

_VIHOKAI_DEPLOY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "push": {
            "type": "boolean",
            "description": "Actually push. Default false — report status only.",
        },
        "allow_main": {
            "type": "boolean",
            "description": "Permit acting on main. Default false.",
        },
    },
    "additionalProperties": False,
}

# Repo root for the VihokAI checkout. Kept as a constant so it is one edit, not
# a search-and-replace across the file.
REPO_ROOT = r"D:\PYWWW\vihokai_com_complete\ai_super_platform"
COPILOT_PACKAGE = "@github/copilot"
COPILOT_JS_ENTRY = REPO_ROOT + r"\github-copilot-1.0.89-win32-x64\package\app.js"


async def _run_cmd(*argv: str, timeout: float = 60.0) -> tuple[int, str, str]:
    """Run a command with no shell and a hard timeout.

    An argument tuple means a message can never be re-parsed as shell syntax.
    Returns ``(returncode, stdout, stderr)``; never raises.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            # stdin MUST be DEVNULL. Without it the child inherits this server's
            # own stdin pipe, so a CLI that reads stdin swallows the remaining
            # JSON-RPC frames and the MCP session dies with "Connection closed".
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=REPO_ROOT,
            # No console window on Windows; the client has no interactive UI
            # to attach to anyway.
            creationflags=getattr(__import__("subprocess"), "CREATE_NO_WINDOW", 0),
        )
    except OSError as exc:
        return -1, "", str(exc)
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout)
    except asyncio.TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        return -1, "", "timed out"
    return proc.returncode or 0, out.decode("utf-8", "replace"), err.decode("utf-8", "replace")


def _env_present(name: str) -> str:
    """Report whether an env var is set — never its value."""
    import os

    return f"{name}: set" if os.environ.get(name) else f"{name}: not set"


async def _copilot_version() -> tuple[str | None, str]:
    """Return ``(version, how)`` for the Copilot CLI, or ``(None, why)``.

    Prefers the extracted package's JS entry over the PATH ``copilot``: on this
    machine the PATH shim is the VS Code bootstrapper, which prints an
    "Install? (y/N)" prompt and blocks a non-interactive caller forever.
    """
    import os

    if os.path.isfile(COPILOT_JS_ENTRY):
        # node, not sys.executable: the entry point is JavaScript.
        rc, out, err = await _run_cmd("node", COPILOT_JS_ENTRY, "--version", timeout=30)
        if rc == 0:
            first = next((ln.strip() for ln in (out or err).splitlines() if ln.strip()), "")
            return (first or None), f"node {COPILOT_JS_ENTRY}"

    rc, out, err = await _run_cmd("copilot", "--version", timeout=30)
    if rc == 0:
        first = next((ln.strip() for ln in (out or err).splitlines() if ln.strip()), "")
        return (first or None), "copilot (PATH)"

    blob = f"{out}\n{err}"
    if "Install?" in blob or "Cannot find GitHub Copilot CLI" in blob:
        return None, (
            "`copilot` on PATH is the VS Code bootstrapper shim and blocks on an "
            f"interactive prompt. Run `npm install -g {COPILOT_PACKAGE}`, or make "
            f"{COPILOT_JS_ENTRY} exist."
        )
    return None, f"copilot not runnable: {(out or err).strip()[:200]}"


async def _copilot_install(args: dict[str, Any]) -> dict[str, Any]:
    do_install = bool(args.get("install"))
    steps: list[str] = []

    if do_install:
        rc, out, err = await _run_cmd("npm", "install", "-g", COPILOT_PACKAGE, timeout=300)
        steps.append(f"npm install -g {COPILOT_PACKAGE} -> rc={rc}")

    version, how = await _copilot_version()
    return {
        "ok": version is not None,
        "action": "install" if do_install else "verify",
        "version": version,
        "via": how,
        "githubToken": _env_present("GITHUB_TOKEN"),
        "steps": steps,
    }


async def _hubspot_track(args: dict[str, Any]) -> dict[str, Any]:
    """Post a lead to VihokAI's own route, which holds the HubSpot token.

    The token is never read or sent here: the app's route handler owns it, so
    this is as safe to call as an ordinary fetch from the site.
    """
    import base64

    message = str(args.get("message") or "").strip()
    if not message:
        return {"ok": False, "error": "'message' is required"}
    intent = str(args.get("intent") or "support")
    if intent not in ("pitch", "invest", "api", "support"):
        return {"ok": False, "error": f"unknown intent {intent!r}"}

    site = str(args.get("site") or "https://www.vihokai.com").rstrip("/")
    target = f"{site}/api/hubspot/conversation"

    body = json.dumps(
        {
            "message": message,
            "intent": intent,
            "email": args.get("email"),
            "page": "/copilot/hubspot-chat-widget",
            "locale": "th",
        },
        ensure_ascii=False,
    ).encode("utf-8")
    body_b64 = base64.b64encode(body).decode("ascii")

    # base64, not a PowerShell here-string: a here-string needs a literal
    # newline after @' and a Thai message does not fit that shape.
    script = (
        "$ProgressPreference='SilentlyContinue';"
        f"$b=[System.Text.Encoding]::UTF8.GetString([System.Convert]::FromBase64String('{body_b64}'));"
        f"try {{ $r=Invoke-WebRequest -Uri '{target}' -Method POST -ContentType 'application/json' "
        "-Body $b -UseBasicParsing -TimeoutSec 20;"
        'Write-Output ("STATUS:" + $r.StatusCode); Write-Output $r.Content } '
        "catch { $c=$_.Exception.Response; if($c){ Write-Output ('STATUS:' + [int]$c.StatusCode); "
        "$s=New-Object System.IO.StreamReader($c.GetResponseStream()); Write-Output $s.ReadToEnd() } "
        "else { Write-Output 'STATUS:000' } }"
    )

    rc, out, err = await _run_cmd(
        "powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script, timeout=60
    )
    text = f"{out}\n{err}"
    status: int | None = None
    for line in text.splitlines():
        if line.startswith("STATUS:"):
            with contextlib.suppress(ValueError):
                status = int(line.split(":", 1)[1])
            break
    payload = text.replace(f"STATUS:{status}", "", 1).strip() if status is not None else text.strip()

    configured = False
    mode: str | None = None
    with contextlib.suppress(json.JSONDecodeError, AttributeError):
        parsed = json.loads(payload)
        configured = bool(parsed.get("configured"))
        mode = parsed.get("mode")

    return {
        "ok": status == 200 and '"ok":false' not in payload.replace(" ", ""),
        "target": target,
        "httpStatus": status,
        "routeConfigured": configured,
        "mode": mode,
        "response": payload[:1000],
        "apiKey": _env_present("HUBSPOT_API_KEY"),
        "rc": rc,
    }


async def _vihokai_deploy(args: dict[str, Any]) -> dict[str, Any]:
    """Report repo state, and push only when explicitly asked and allowed."""
    rc, out, _err = await _run_cmd("git", "rev-parse", "--abbrev-ref", "HEAD", timeout=30)
    branch = out.strip() if rc == 0 else None
    if not branch:
        return {"ok": False, "action": "none", "error": "not a git repository"}

    rc, sha, _ = await _run_cmd("git", "rev-parse", "--short", "HEAD", timeout=30)
    head = sha.strip() if rc == 0 else None

    rc, dirty_out, _ = await _run_cmd("git", "status", "--porcelain", "--untracked-files=no", timeout=30)
    dirty = bool(dirty_out.strip()) if rc == 0 else None

    rc, ahead_out, _ = await _run_cmd(
        "git", "rev-list", "--count", f"origin/{branch}..HEAD", timeout=30
    )
    ahead = int(ahead_out.strip()) if rc == 0 and ahead_out.strip().isdigit() else None

    base = {
        "branch": branch,
        "head": head,
        "ahead": ahead,
        "dirty": dirty,
    }

    steps: list[str] = []
    if not args.get("push"):
        return {**base, "ok": True, "action": "status-only",
                "steps": ["skipped push (push not requested)"]}

    if branch == "main" and not args.get("allow_main"):
        return {**base, "ok": False, "action": "refused", "steps": [],
                "error": "on main — retry with allow_main=true to deploy from main"}

    if dirty and not args.get("allow_main"):
        return {**base, "ok": False, "action": "refused", "steps": [],
                "error": "tracked files have uncommitted changes — commit them first"}

    rc, push_out, push_err = await _run_cmd("git", "push", "origin", branch, timeout=180)
    steps.append(f"git push origin {branch} -> rc={rc}")
    return {
        **base,
        "ok": rc == 0,
        "action": "push",
        "steps": steps,
        "error": None if rc == 0 else (push_out + push_err).strip()[:500],
    }


def build_server(gateway: Gateway | None = None) -> Server:
    """Create the MCP server. *gateway* is injectable for tests."""
    state: dict[str, Gateway | None] = {"gateway": gateway}
    # Last plan seen from ``gpt6_chat``, so ``gpt6_plan`` can report it without
    # re-running the model. Read-only: it never influences routing or approval.
    last_plan: dict[str, Any] = {"plan": [], "plan_progress": {}}

    async def _gateway() -> Gateway:
        current = state["gateway"]
        if current is None:
            current = Gateway(settings=get_settings())
            await current.start()
            state["gateway"] = current
        return current

    async def list_tools(_ctx: Any, _params: Any = None) -> ListToolsResult:
        return ListToolsResult(
            tools=[
                Tool(
                    name="gpt6_chat",
                    description=(
                        "Ask GPT-6 Sol a question. The model may call MCP tools "
                        "through the gateway to answer."
                    ),
                    input_schema=_CHAT_SCHEMA,
                ),
                Tool(
                    name="gpt6_models",
                    description="List the models this gateway knows about, with aliases.",
                    input_schema=_EMPTY_SCHEMA,
                ),
                Tool(
                    name="gpt6_status",
                    description="Report the gateway configuration (no secrets).",
                    input_schema=_EMPTY_SCHEMA,
                ),
                Tool(
                    name="gpt6_tools",
                    description="List the MCP tools the gateway is allowed to call.",
                    input_schema=_EMPTY_SCHEMA,
                ),
                Tool(
                    name="gpt6_plan",
                    description=(
                        "Return the working plan from the most recent gpt6_chat call. "
                        "Empty unless plan mode is enabled. Advisory only."
                    ),
                    input_schema=_EMPTY_SCHEMA,
                ),
                Tool(
                    name="gpt6_debug_marathon",
                    description=(
                        "Queue 1-10 debug tasks. Tasks are ordered by priority, "
                        "difficulty, then complexity (highest first). Each task "
                        "runs through GPT-6 Luna, then Sol, then Astra. Returns a job_id."
                    ),
                    input_schema=_DEBUG_MARATHON_SCHEMA,
                ),
                Tool(
                    name="gpt6_debug_marathon_status",
                    description="Get progress and results for a debug marathon job_id.",
                    input_schema=_DEBUG_MARATHON_STATUS_SCHEMA,
                ),
                Tool(
                    name="copilot_install",
                    description=(
                        "Verify the GitHub Copilot CLI, and optionally install it with "
                        "`npm install -g @github/copilot`. Verifies only unless install=true."
                    ),
                    input_schema=_COPILOT_INSTALL_SCHEMA,
                ),
                Tool(
                    name="hubspot_track",
                    description=(
                        "Record a lead through VihokAI's /api/hubspot/conversation route, "
                        "which holds the HubSpot credential server-side. Reports whether the "
                        "app has HubSpot configured."
                    ),
                    input_schema=_HUBSPOT_TRACK_SCHEMA,
                ),
                Tool(
                    name="vihokai_deploy",
                    description=(
                        "Report the VihokAI repo branch, HEAD, commits ahead of origin, and "
                        "dirty state. Pushes only when push=true; refuses on main without "
                        "allow_main=true."
                    ),
                    input_schema=_VIHOKAI_DEPLOY_SCHEMA,
                ),
            ]
        )

    async def call_tool(_ctx: Any, params: Any) -> CallToolResult:
        name = str(getattr(params, "name", "") or "")
        args = dict(getattr(params, "arguments", None) or {})
        blocks: list[Any] = list(await _dispatch(name, args))
        return CallToolResult(content=blocks)

    async def _dispatch(name: str, args: dict[str, Any]) -> list[TextContent]:
        try:
            if name == "gpt6_chat":
                prompt = str(args.get("prompt") or "").strip()
                if not prompt:
                    return [TextContent(type="text", text="Error: 'prompt' is required")]
                current = await _gateway()
                result = await current.chat(
                    [{"role": "user", "content": prompt}],
                    system=args.get("system"),
                    model_id=args.get("model"),
                )
                payload = {
                    "content": result.content,
                    "model": result.model,
                    "rounds": result.rounds,
                    "used_tools": result.used_tools,
                    "plan": result.plan,
                    "plan_progress": result.plan_progress,
                    "loop_stopped": result.loop_stopped,
                }
                last_plan["plan"] = result.plan
                last_plan["plan_progress"] = result.plan_progress
                return [TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))]

            if name == "gpt6_models":
                payload = {
                    model_id: {
                        "label": spec["label"],
                        "tier": spec["tier"],
                        "context_window": spec["context_window"],
                        "max_output": spec["max_output"],
                        "aliases": sorted(
                            a for a, target in MODEL_ALIASES.items() if target == model_id
                        ),
                    }
                    for model_id, spec in sorted(KNOWN_MODELS.items())
                }
                return [TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))]

            if name == "gpt6_status":
                current = await _gateway()
                return [
                    TextContent(
                        type="text",
                        text=json.dumps(current.status(), ensure_ascii=False),
                    )
                ]

            if name == "gpt6_tools":
                current = await _gateway()
                tools = await current.list_tools()
                return [
                    TextContent(
                        type="text",
                        text=json.dumps([t["function"]["name"] for t in tools], ensure_ascii=False),
                    )
                ]

            if name == "gpt6_plan":
                return [
                    TextContent(
                        type="text",
                        text=json.dumps(last_plan, ensure_ascii=False),
                    )
                ]

            if name == "gpt6_debug_marathon":
                current = await _gateway()
                submitted_job = current.debug_marathon.submit(args.get("tasks") or [])
                return [
                    TextContent(type="text", text=json.dumps(submitted_job, ensure_ascii=False))
                ]

            if name == "gpt6_debug_marathon_status":
                current = await _gateway()
                status_payload = current.debug_marathon.get(str(args.get("job_id") or ""))
                if status_payload is None:
                    return [TextContent(type="text", text="Error: debug marathon job not found")]
                return [
                    TextContent(type="text", text=json.dumps(status_payload, ensure_ascii=False))
                ]

            if name == "copilot_install":
                return [TextContent(type="text", text=json.dumps(await _copilot_install(args), ensure_ascii=False))]

            if name == "hubspot_track":
                return [TextContent(type="text", text=json.dumps(await _hubspot_track(args), ensure_ascii=False))]

            if name == "vihokai_deploy":
                return [TextContent(type="text", text=json.dumps(await _vihokai_deploy(args), ensure_ascii=False))]

            return [TextContent(type="text", text=f"Error: unknown tool {name!r}")]
        except Exception as exc:
            log_error("mcp tool %s failed: %s", redact(name), redact(str(exc)))
            return [TextContent(type="text", text=f"Error: {exc}")]

    return Server(
        SERVER_NAME,
        on_list_tools=list_tools,
        on_call_tool=call_tool,
    )


async def _run() -> None:
    server = build_server()
    log_info("starting stdio MCP server %s", SERVER_NAME)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main() -> None:
    """Console entry point for ``python -m gpt6_sol_mcp.gateway.server``."""
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(_run())


if __name__ == "__main__":
    main()
