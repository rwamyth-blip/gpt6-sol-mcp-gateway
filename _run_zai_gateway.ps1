# ---------------------------------------------------------------------------
# mcp5-Zai — standalone Z.ai (GLM) gateway launcher
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File _run_zai_gateway.ps1
#   powershell -NoProfile -ExecutionPolicy Bypass -File _run_zai_gateway.ps1 -Port 7424
#
# Starts the FastAPI gateway on port 7424 using ONLY this folder's .venv and
# .env. It never touches ports 7420/7421/7422/7423 or their processes.
# ---------------------------------------------------------------------------
param(
    [int]$Port = 0  # 0 = use GATEWAY_PORT from .env (default 7424)
)

$ErrorActionPreference = "Stop"

$m5 = $PSScriptRoot
$venvPython = Join-Path $m5 ".venv\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    throw "mcp5 venv python not found: $venvPython (run: python -m venv .venv && pip install -e `".[all]`")"
}

# Load GATEWAY_PORT from .env when -Port was not given explicitly.
if ($Port -eq 0) {
    $envLine = Get-Content (Join-Path $m5 ".env") | Where-Object { $_ -match '^GATEWAY_PORT\s*=' } | Select-Object -First 1
    $Port = if ($envLine) { [int]($envLine.Split('=')[1].Trim()) } else { 7424 }
}

if (Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue) {
    Write-Host "Port $Port is already listening - is the Zai gateway already up?"
    exit 0
}

# Ensure OPENAI_API_KEY-style fallback is never needed: the Z.ai key lives in
# .env as LLM_API_KEY, and pydantic-settings reads .env from the package root.
Set-Location $m5

Write-Host "Starting mcp5-Zai gateway on http://127.0.0.1:$Port (standalone: $venvPython)"
& $venvPython -m uvicorn gpt6_sol_mcp.gateway.app:app --host 127.0.0.1 --port $Port
