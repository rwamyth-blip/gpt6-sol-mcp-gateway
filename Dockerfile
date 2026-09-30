# syntax=docker/dockerfile:1.7
# ---------------------------------------------------------------------------
# Multi-stage build. One image serves both modes:
#   docker run ... <image>                 -> FastAPI gateway (default)
#   docker run ... <image> mcp             -> stdio MCP server
# ---------------------------------------------------------------------------

# --- Stage 1: builder ------------------------------------------------------
FROM python:3.12-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

# Build tooling for any dependency that needs to compile.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only what the build needs, so the dependency layer caches well.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src

# Install into an isolated prefix we can copy wholesale into the runtime stage.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip setuptools wheel \
    && pip install ".[all]"

# --- Stage 2: runtime ------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH="/opt/venv/bin:$PATH" \
    GATEWAY_HOST=0.0.0.0 \
    GATEWAY_PORT=8000

# Non-root user. A fixed uid keeps volume permissions predictable.
RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --create-home --shell /usr/sbin/nologin app

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=app:app pyproject.toml README.md LICENSE ./
COPY --chown=app:app src ./src

USER app

EXPOSE 8000

# /health is unauthenticated by design, so the probe needs no credentials.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import os,urllib.request,sys; \
port=os.environ.get('GATEWAY_PORT','8000'); \
sys.exit(0 if urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=4).status==200 else 1)"

# Default: the FastAPI gateway. Override with `mcp` for the stdio server.
ENTRYPOINT ["mcp-for-copilot"]
CMD ["serve"]
