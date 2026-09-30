# Stops the MCP for Copilot gateway listening on the configured port.
#
# Only terminates the process that owns the listening socket AND whose command
# line points at this package, so an unrelated service (e.g. vihokai-codex-agent
# on :7420) is never killed by accident.

$ErrorActionPreference = "Stop"

$gwDir = Split-Path -Parent $PSScriptRoot

$port = 7423
$envFile = Join-Path $gwDir ".env"
if (Test-Path $envFile) {
    $match = Select-String -Path $envFile -Pattern '^\s*GATEWAY_PORT\s*=\s*(\d+)' |
        Select-Object -First 1
    if ($match) { $port = [int]$match.Matches[0].Groups[1].Value }
}

$listeners = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
if (-not $listeners) {
    Write-Output "Nothing listening on port $port."
    exit 0
}

foreach ($ownerPid in ($listeners.OwningProcess | Select-Object -Unique)) {
    $cmd = (Get-CimInstance Win32_Process -Filter "ProcessId=$ownerPid" -ErrorAction SilentlyContinue).CommandLine
    if (-not ($cmd -and $cmd -like "*mcp_for_copilot*")) {
        Write-Error ("Port $port is held by PID $ownerPid, which is not this gateway: $cmd. Refusing to stop it.")
        exit 1
    }
    Stop-Process -Id $ownerPid -Force -ErrorAction SilentlyContinue
    Write-Output "Stopped gateway PID $ownerPid."
}

Start-Sleep -Seconds 2
if (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue) {
    Write-Error "Port $port is still in use."
    exit 1
}
Write-Output "Port $port is free."
