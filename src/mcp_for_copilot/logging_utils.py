"""Secret-safe logging.

Every log line produced by this package goes through :func:`safe_log`, which
runs the message through :func:`redact` first. The redactor is deliberately
aggressive: it removes anything that *looks* like a credential even when the
caller forgot to mark it as one.

Threat model
------------
A tool result or an upstream error body can contain a token that the model
echoed back, or a connection string with an embedded password. If that text
reaches a log file it is a leak, so redaction happens at the boundary rather
than at each call site.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger("mcp_for_copilot")

# ---------------------------------------------------------------------------
# Redaction rules
# ---------------------------------------------------------------------------
# Order matters: the most specific patterns run first so that a JWT is not
# partially eaten by the generic "long token" rule.
_REDACTION_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    # OpenAI / DeepSeek / Moonshot style keys: sk-..., sk-proj-..., sk-b...
    (re.compile(r"\bsk-[A-Za-z0-9_\-]{8,}\b"), "sk-***REDACTED***"),
    # Groq keys
    (re.compile(r"\bgsk_[A-Za-z0-9]{8,}\b"), "gsk_***REDACTED***"),
    # Google API keys
    (re.compile(r"\bAIza[A-Za-z0-9_\-]{20,}\b"), "AIza***REDACTED***"),
    # GitHub tokens
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{16,}\b"), "gh*_***REDACTED***"),
    # Slack tokens
    (re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}\b"), "xox*-***REDACTED***"),
    # AWS access key ids
    (re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"), "AKIA***REDACTED***"),
    # Bearer / Basic authorization headers
    (
        re.compile(r"(?i)\b(bearer|basic)\s+[A-Za-z0-9._\-+/=]{8,}"),
        r"\1 ***REDACTED***",
    ),
    # JWTs (three base64url segments)
    (
        re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\b"),
        "***JWT-REDACTED***",
    ),
    # key=value / key: value where the key name says "secret"
    (
        re.compile(
            r"(?i)\b([A-Za-z0-9_]*(?:api[_-]?key|secret|password|passwd|token|"
            r"credential|auth)[A-Za-z0-9_]*)\s*[:=]\s*[\"']?([^\s\"',;]{4,})"
        ),
        r"\1=***REDACTED***",
    ),
    # Credentials embedded in a URL: scheme://user:pass@host
    (
        re.compile(r"(?i)\b([a-z][a-z0-9+.\-]*://[^:/\s@]+):([^@/\s]{3,})@"),
        r"\1:***REDACTED***@",
    ),
    # MongoDB connection strings
    (
        re.compile(r"(?i)\bmongodb(?:\+srv)?://[^\s\"']+"),
        "mongodb://***REDACTED***",
    ),
)

# Values that are obviously placeholders and must NOT be redacted, otherwise
# the docs and .env.example become unreadable in logs.
_PLACEHOLDER_RE = re.compile(
    r"(?i)^(?:your[_-]?|my[_-]?|xxx+|placeholder|changeme|redacted|\*+|none|null|"
    r"true|false|example|test|dummy|fake)"
)


def redact(value: Any) -> str:
    """Return *value* as a string with every credential-looking substring removed.

    Safe to call on any object; ``None`` becomes ``""``.
    """
    if value is None:
        return ""
    text = value if isinstance(value, str) else str(value)
    for pattern, replacement in _REDACTION_RULES:
        text = pattern.sub(replacement, text)
    return text


def redact_mapping(data: dict[str, Any]) -> dict[str, Any]:
    """Redact every value in a shallow mapping (keys are preserved)."""
    return {k: redact(v) for k, v in data.items()}


def contains_secret(text: str) -> bool:
    """True when *text* still contains something the redactor would remove.

    Used by the test-suite to assert that no secret reaches a log sink.
    """
    if not text:
        return False
    for pattern, _ in _REDACTION_RULES:
        for match in pattern.finditer(text):
            candidate = match.group(0)
            # Ignore matches that are already redaction markers or placeholders.
            if "REDACTED" in candidate:
                continue
            if _PLACEHOLDER_RE.match(candidate):
                continue
            return True
    return False


def _redact_arg(value: Any) -> Any:
    """Redact a log argument without changing its type.

    ``redact()`` always returns ``str``, which would break ``%d``/``%f``
    format specifiers. Non-string scalars are passed through untouched
    (they cannot carry a credential); everything else is redacted.
    """
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return redact(value)


def safe_log(level: int, message: str, *args: Any, **kwargs: Any) -> None:
    """Log *message* after redaction.

    ``%``-style args are redacted individually so a secret passed as an
    argument cannot slip through the format string. Scalar args keep their
    type so ``%d``/``%f`` specifiers still work.
    """
    safe_args = tuple(_redact_arg(a) for a in args)
    logger.log(level, redact(message), *safe_args, **kwargs)


def log_info(message: str, *args: Any) -> None:
    safe_log(logging.INFO, message, *args)


def log_warning(message: str, *args: Any) -> None:
    safe_log(logging.WARNING, message, *args)


def log_error(message: str, *args: Any) -> None:
    safe_log(logging.ERROR, message, *args)


def log_debug(message: str, *args: Any) -> None:
    safe_log(logging.DEBUG, message, *args)
