"""Command-line interface for the gateway.

Subcommands
-----------
``serve``
    Run the FastAPI gateway with uvicorn.
``mcp``
    Run the stdio MCP server.
``chat``
    Send one prompt and print the answer.
``marathon``
    Queue debug tasks and poll the job until it finishes.
``models``
    List the known models.
``status``
    Print the non-secret configuration.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any

from .config import get_settings
from .logging_utils import log_error, redact
from .provider import KNOWN_MODELS, MODEL_ALIASES


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mcp-for-copilot",
        description="MCP for Copilot — OpenAI-compatible gateway with MCP tools.",
    )
    parser.add_argument("--version", action="version", version="mcp-for-copilot 0.1.0")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="Run the FastAPI gateway (uvicorn).")
    serve.add_argument("--host", default=None, help="Bind host (default: GATEWAY_HOST).")
    serve.add_argument("--port", type=int, default=None, help="Bind port (default: GATEWAY_PORT).")
    serve.add_argument("--reload", action="store_true", help="Auto-reload on code changes.")

    sub.add_parser("mcp", help="Run the stdio MCP server.")

    chat = sub.add_parser("chat", help="Send one prompt and print the answer.")
    chat.add_argument("prompt", help="The prompt to send.")
    chat.add_argument("--system", default=None, help="Optional system instruction.")
    chat.add_argument("--model", default=None, help="Model id or alias.")
    chat.add_argument("--json", action="store_true", help="Print the full JSON result.")
    chat.add_argument(
        "--plan",
        action="store_true",
        help="Enable plan mode for this call only (does not change the environment).",
    )

    marathon = sub.add_parser(
        "marathon", help="Queue debug tasks and poll the job until it finishes."
    )
    marathon.add_argument(
        "questions",
        nargs="+",
        help="One or more debug questions (1-10).",
    )
    marathon.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Seconds between polls (default: 2.0).",
    )
    marathon.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="Give up after this many seconds (default: 600).",
    )
    marathon.add_argument("--json", action="store_true", help="Print the final job JSON.")

    sub.add_parser("models", help="List the known models.")
    sub.add_parser("status", help="Print the non-secret configuration.")

    return parser


def _cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        print(
            "uvicorn is not installed. Install the server extras:\n"
            "  pip install 'mcp-for-copilot[server]'",
            file=sys.stderr,
        )
        return 1

    settings = get_settings()
    host = args.host or settings.gateway_host
    port = args.port or settings.gateway_port
    print(f"Starting gateway on http://{host}:{port}")
    uvicorn.run(
        "mcp_for_copilot.gateway.app:app",
        host=host,
        port=port,
        reload=bool(args.reload),
    )
    return 0


def _cmd_mcp(_args: argparse.Namespace) -> int:
    from .gateway.server import main as server_main

    server_main()
    return 0


def _cmd_chat(args: argparse.Namespace) -> int:
    from .gateway.facade import Gateway

    settings = get_settings()
    if args.plan:
        # Per-invocation override only; the process environment is untouched.
        settings = settings.model_copy(update={"mcp_plan_mode": True})

    async def _run() -> int:
        async with Gateway(settings=settings) as gateway:
            result = await gateway.chat(
                [{"role": "user", "content": args.prompt}],
                system=args.system,
                model_id=args.model,
            )
        if args.json:
            print(
                json.dumps(
                    {
                        "content": result.content,
                        "model": result.model,
                        "rounds": result.rounds,
                        "used_tools": result.used_tools,
                        "usage": result.usage,
                        "plan": result.plan,
                        "plan_progress": result.plan_progress,
                        "loop_stopped": result.loop_stopped,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            print(result.content)
            if result.loop_stopped:
                print("\n[loop] stopped: the model repeated the same tool call.")
            if result.plan:
                print("\nPlan:")
                for step in result.plan:
                    print(f"  [{step['status']}] {step['description']}")
        return 0

    try:
        return asyncio.run(_run())
    except Exception as exc:
        log_error("chat failed: %s", redact(str(exc)))
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def _cmd_marathon(args: argparse.Namespace) -> int:
    from .gateway.facade import Gateway

    questions = [q for q in args.questions if q.strip()][:10]
    if not questions:
        print("Error: at least one non-empty question is required", file=sys.stderr)
        return 1

    async def _run() -> int:
        async with Gateway() as gateway:
            job = gateway.debug_marathon.submit([{"question": q} for q in questions])
            job_id = str(job.get("job_id") or "")
            print(f"job_id={job_id}", file=sys.stderr)
            waited = 0.0
            while waited < args.timeout:
                await asyncio.sleep(args.interval)
                waited += args.interval
                current = gateway.debug_marathon.get(job_id)
                if current is None:
                    print("Error: job disappeared", file=sys.stderr)
                    return 1
                if current.get("status") in {"completed", "failed"}:
                    if args.json:
                        print(json.dumps(current, ensure_ascii=False, indent=2))
                    else:
                        print(f"status={current.get('status')}")
                        print(f"verification={current.get('verification')}")
                        for result in current.get("results") or []:
                            print(f"- {result.get('title')}: {result.get('status')}")
                    return 0
            print(f"Error: timed out after {args.timeout}s", file=sys.stderr)
            return 1

    try:
        return asyncio.run(_run())
    except Exception as exc:
        log_error("marathon failed: %s", redact(str(exc)))
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def _cmd_models(_args: argparse.Namespace) -> int:
    payload: dict[str, Any] = {
        model_id: {
            "label": spec["label"],
            "tier": spec["tier"],
            "context_window": spec["context_window"],
            "max_output": spec["max_output"],
            "aliases": sorted(a for a, t in MODEL_ALIASES.items() if t == model_id),
        }
        for model_id, spec in sorted(KNOWN_MODELS.items())
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _cmd_status(_args: argparse.Namespace) -> int:
    print(json.dumps(get_settings().describe(), ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``mcp-for-copilot`` console script."""
    args = _build_parser().parse_args(argv)
    handlers = {
        "serve": _cmd_serve,
        "mcp": _cmd_mcp,
        "chat": _cmd_chat,
        "marathon": _cmd_marathon,
        "models": _cmd_models,
        "status": _cmd_status,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
