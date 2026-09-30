# Starts the MCP for Copilot gateway as a hidden background process.
#
# Used by the Startup-folder shortcut so the gateway survives logoff/reboot.
# Safe to run repeatedly: an existing listener on the port is left alone.
#
# Port: read from GATEWAY_PORT in .env. The default is 7423, NOT 7420 --
# :7420 is held by vihokai-codex-agent (its _run_api.ps1 hardcodes 7420).
# :7421 = MCP2 clone, :7422 = MCP3 Codex API.

$ErrorActionPreference = "Stop"

$gwDir = Split-Path -Parent $PSScriptRoot
$python = Join-Path $gwDir "venv\Scripts\pythonw.exe"
$logDir = Join-Path $gwDir "logs"
$logFile = Join-Path $logDir "gateway.log"

if (-not (Test-Path $python)) {
    Write-Error "Interpreter not found: $python"
    exit 1
}

# Read the port from .env so this script never hardcodes it.
$port = 7423
$envFile = Join-Path $gwDir ".env"
if (Test-Path $envFile) {
    $match = Select-String -Path $envFile -Pattern '^\s*GATEWAY_PORT\s*=\s*(\d+)' |
        Select-Object -First 1
    if ($match) { $port = [int]$match.Matches[0].Groups[1].Value }
}

# Already listening? Only a no-op when the listener is *our* gateway. A foreign
# service on the port is a hard error: silently exiting 0 was why a restart
# looked successful while the old service kept serving /health.
$existing = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
if ($existing) {
    $ownerPid = $existing[0].OwningProcess
    $cmd = (Get-CimInstance Win32_Process -Filter "ProcessId=$ownerPid" -ErrorAction SilentlyContinue).CommandLine
    if ($cmd -and $cmd -like "*mcp_for_copilot*") {
        Write-Output "Gateway already listening on port $port (PID $ownerPid)."
        exit 0
    }
    Write-Error ("Port $port is held by PID $ownerPid, which is not this gateway. " +
        "Set GATEWAY_PORT in .env to a free port (e.g. 7423). " +
        "Do NOT stop that process here.")
    exit 1
}

New-Item -ItemType Directory -Force -Path $logDir | Out-Null

# Rotate the log so it cannot grow without bound.
if (Test-Path $logFile) {
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    Move-Item $logFile (Join-Path $logDir "gateway-$stamp.log") -Force
    Get-ChildItem $logDir -Filter "gateway-*.log" |
        Sort-Object LastWriteTime -Descending |
        Select-Object -Skip 5 |
        Remove-Item -Force -ErrorAction SilentlyContinue
}

$env:PYTHONPATH = Join-Path $gwDir "src"

$proc = Start-Process -FilePath $python `
    -ArgumentList @(
        "-m", "uvicorn", "mcp_for_copilot.gateway.app:app",
        "--host", "127.0.0.1", "--port", "$port"
    ) `
    -WorkingDirectory $gwDir `
    -WindowStyle Hidden `
    -RedirectStandardOutput $logFile `
    -RedirectStandardError (Join-Path $logDir "gateway.err.log") `
    -PassThru

# Wait briefly so a crash-on-startup is visible in the exit code.
Start-Sleep -Seconds 4
if ($proc.HasExited) {
    Write-Error "Gateway exited immediately (code $($proc.ExitCode)). See $logDir"
    exit 1
}

Write-Output "Gateway started on port $port (PID $($proc.Id)). Log: $logFile"
