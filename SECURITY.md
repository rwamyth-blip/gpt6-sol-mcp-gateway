# Security Policy

## Supported versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | ✅ |

## Reporting a vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Report privately using GitHub's
[private vulnerability reporting](https://github.com/rwamyth-blip/mcp-for-copilot/security/advisories/new),
or email **dev@vihokai.com**.

Please include:

- A description of the issue and its impact
- Steps to reproduce, or a proof-of-concept
- The affected version(s)
- Any suggested mitigation

### What to expect

| Stage | Target |
|-------|--------|
| Acknowledgement | within 48 hours |
| Initial assessment | within 5 business days |
| Fix or mitigation plan | within 30 days |
| Public disclosure | coordinated with you, after a fix ships |

We will credit you in the release notes unless you ask us not to.

---

## Security model

This package sits between a language model and your tools, so its threat model
is worth stating explicitly.

### Trust boundaries

| Component | Trust level |
|-----------|-------------|
| Your application code | Trusted |
| `Settings` / environment | Trusted |
| The LLM provider | Semi-trusted — it can request any tool call |
| The MCP server | Semi-trusted — it can return arbitrary content |
| **Tool output** | **Untrusted** — may contain injected instructions |
| **Model-requested arguments** | **Untrusted** — validated before dispatch |

### Controls

**1. Allowlist (fail closed).**
Only tools in `MCP_ALLOWED_TOOLS` (or the read-only default set) can run. A tool
that is allowed by name but was never advertised by the server is still denied,
because its arguments cannot be validated.

**2. Approval (fail closed).**
Tools matching a risky verb (`write`, `delete`, `exec`, `deploy`, …) require an
explicit approval decision. When no approver is configured, every risky request
is **denied**. An approver that raises or returns an invalid object also denies.

**3. Argument validation.**
Arguments are validated against the tool's declared JSON schema before dispatch.
`bool` is checked explicitly, because it is a subclass of `int` in Python and
would otherwise satisfy `{"type": "integer"}`.

**4. Prompt-injection guard.**
Tool output is wrapped in a labelled block that tells the model to treat it as
data only and never follow instructions found inside it. This reduces, but does
not eliminate, injection risk — treat it as defence in depth.

**5. Secret redaction.**
Eleven regex rules scrub API keys, bearer tokens, JWTs, connection strings, and
private keys from every log line and error message. `Settings.describe()` and
`Gateway.status()` return only booleans for credential presence, never values.

**6. Gateway authentication.**
When `GATEWAY_API_KEY` is set, every `/v1/*` request must present it as a bearer
token. When it is empty the gateway accepts any key — **this is for local
development only** and must never be used on a public interface.

### Known limitations

- **No rate limiting.** Put the gateway behind a reverse proxy or API gateway
  that enforces quotas.
- **No per-user authorisation.** The gateway has one identity. If you need
  multi-tenancy, run one instance per tenant.
- **Prompt injection is mitigated, not solved.** A sufficiently adversarial tool
  result may still influence the model. Keep the allowlist tight and require
  approval for anything that mutates state.
- **`GATEWAY_API_KEY` is a shared secret**, not a user account system. Rotate it
  like any other credential.
- **Tool output is not size-limited.** A tool that returns a very large payload
  can exhaust the model's context window. Bound it in your MCP server.

### Deployment checklist

- [ ] `LLM_API_KEY` is set from a secret manager, never from a committed file
- [ ] `GATEWAY_API_KEY` is set to a strong random value
- [ ] `GATEWAY_CORS_ORIGINS` is an explicit list, not `*`
- [ ] `MCP_ALLOWED_TOOLS` is as narrow as your use case allows
- [ ] `MCP_REQUIRE_APPROVAL=true` for any tool that mutates state
- [ ] The gateway is not exposed directly to the public internet
- [ ] Logs are shipped somewhere with access control
- [ ] `.env` is gitignored and absent from the image
