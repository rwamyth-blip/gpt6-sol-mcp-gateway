# Contributing to MCP for Copilot

Thanks for taking the time to contribute! 🎉

This document explains how to set up the project, what we expect from a pull
request, and how to get it merged quickly.

---

## Code of Conduct

By participating you agree to abide by our
[Code of Conduct](CODE_OF_CONDUCT.md). Please report unacceptable behaviour to
`dev@vihokai.com`.

---

## Ways to contribute

- 🐛 **Report a bug** — open an issue using the bug template
- 💡 **Request a feature** — open an issue using the feature template
- 📖 **Improve docs** — typos, unclear sections, and missing examples all count
- 🔧 **Send a pull request** — see the workflow below
- ⭐ **Star the repo** — it genuinely helps

---

## Development setup

### Requirements

- Python **3.10+**
- Git
- (optional) Docker, for testing the container build

### Steps

```bash
# 1. Fork the repo on GitHub, then clone your fork
git clone https://github.com/<your-username>/mcp-for-copilot.git
cd mcp-for-copilot

# 2. Create a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. Install with all extras plus dev tooling
pip install -e ".[all,dev]"

# 4. Install the git hooks
pre-commit install

# 5. Verify everything works
make check
```

### Useful commands

| Command | What it does |
|---------|--------------|
| `make test` | Run pytest with coverage |
| `make lint` | Run `ruff check` and `ruff format --check` |
| `make format` | Auto-fix lint issues and format |
| `make typecheck` | Run `mypy --strict` |
| `make check` | lint + typecheck + test |
| `make docs` | Serve the docs locally at http://127.0.0.1:8000 |
| `make build` | Build the sdist and wheel |
| `make clean` | Remove caches and build artifacts |

---

## Project layout

```
src/mcp_for_copilot/
├── __init__.py          # public API surface
├── config.py            # env-driven Settings
├── logging_utils.py     # secret redaction + logging helpers
├── provider.py          # OpenAI-compatible LLM adapter
├── mcp_client.py        # MCP client (stdio / http / sse)
├── tool_router.py       # allowlist + argument validation
├── approval.py          # human-in-the-loop approval layer
├── orchestrator.py      # the model → tool → model loop
├── cli.py               # mcp-for-copilot entry point
└── gateway/
    ├── facade.py        # Gateway — owns the whole stack
    ├── app.py           # FastAPI OpenAI-compatible app
    └── server.py        # stdio MCP server
```

---

## Coding standards

- **Formatting** — `ruff format` (line length 100, double quotes). Do not
  hand-format; run `make format`.
- **Linting** — `ruff check` must pass with the configured rule set.
- **Types** — full annotations. `mypy --strict` must pass on `src/`.
- **Docstrings** — every public module, class, and function. Explain *why*, not
  just *what*.
- **Comments** — only where the code is not self-evident. Do not narrate.
- **Errors** — raise a specific exception type. Never `except Exception: pass`.
- **Secrets** — never log a credential. Route every message through
  `logging_utils.redact()`.

### Design rules this project holds to

1. **Fail closed.** An unknown tool, a missing approver, or an unparseable
   argument is a *denial*, never an implicit allow.
2. **No hardcoded model ids or credentials.** Everything comes from `Settings`.
3. **No secret in an error message.** Redact at the boundary.
4. **Tool output is untrusted.** It is wrapped and labelled before it re-enters
   the model context.
5. **Tests never touch the network.** Inject an `httpx` transport or a fake MCP
   client.

---

## Testing

Tests live in three layers. Put your test in the lowest layer that can cover it.

| Layer | Path | Use for |
|-------|------|---------|
| Unit | `tests/unit/` | One module in isolation, no I/O |
| Integration | `tests/integration/` | FastAPI routes, MCP server tools |
| End-to-end | `tests/e2e/` | The full loop with a fake MCP server |

```bash
pytest tests/unit -q                       # fast feedback
pytest tests/unit/test_tool_router.py -q   # one file
pytest --cov --cov-report=html             # coverage report
```

Guidelines:

- Every bug fix needs a regression test that fails before the fix.
- Every new public function needs at least one test.
- Use `pytest.mark.asyncio` (or rely on `asyncio_mode = "auto"`).
- Prefer a real object over a mock. Mock only at the network boundary.

---

## Pull request workflow

1. **Open an issue first** for anything larger than a typo, so we can agree on
   the approach before you write code.
2. **Branch from `main`** using a descriptive name:
   - `feat/add-sse-reconnect`
   - `fix/router-bool-as-int`
   - `docs/clarify-approval`
3. **Keep commits focused.** One logical change per commit.
4. **Write a conventional commit message:**

   ```
   feat(router): reject boolean values for integer arguments

   bool is a subclass of int in Python, so True satisfied {"type": "integer"}.
   Check booleans explicitly before the isinstance test.

   Closes #42
   ```

   Types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `build`, `ci`,
   `chore`, `revert`.
5. **Run `make check`** and make sure it passes.
6. **Update the docs** if you changed behaviour or configuration.
7. **Add a `CHANGELOG.md` entry** under `## [Unreleased]`.
8. **Open the PR** and fill in the template. Link the issue with `Closes #N`.

### Review criteria

A reviewer will check:

- [ ] Does it solve the stated problem, completely?
- [ ] Does it fail closed on the error paths?
- [ ] Are there tests, and do they fail without the change?
- [ ] Is `make check` green?
- [ ] Are docs and `CHANGELOG.md` updated?
- [ ] Is there any secret, hardcoded path, or hardcoded model id?

### After merge

The maintainer will squash-merge and, for user-visible changes, cut a release.
Releases are published to PyPI automatically by
[`.github/workflows/release.yml`](.github/workflows/release.yml) using PyPI
Trusted Publishing — no API token is stored in the repo.

---

## Reporting a security issue

**Do not open a public issue.** Follow [SECURITY.md](SECURITY.md).

---

## Questions?

Open a [discussion](https://github.com/rwamyth-blip/mcp-for-copilot/discussions)
or an issue. We are happy to help.
