<# .SYNOPSIS
    Start the A-Share Agent LangGraph server.

.DESCRIPTION
    Supports two modes:
      -Dev       Local development server (pickle-based persistence, default)
      -Prod      Docker-based server with PostgreSQL persistence

.EXAMPLE
    .\scripts\start-server.ps1           # Dev mode (default)
    .\scripts\start-server.ps1 -Prod     # Production mode with Docker
#>

param(
    [switch]$Prod,
    [int]$Port = 2024,
    [switch]$NoReload,
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
Push-Location $PSScriptRoot\..

try {
    if ($Prod) {
        Write-Host "`n=== Starting A-Share Agent (Production - Docker + PostgreSQL) ===" -ForegroundColor Green
        Write-Host "Threads and checkpoints will be persisted in PostgreSQL." -ForegroundColor Cyan
        Write-Host ""

        # Verify Docker is available
        try {
            docker version | Out-Null
        } catch {
            Write-Host "ERROR: Docker is not running or not installed." -ForegroundColor Red
            Write-Host "Please install Docker Desktop: https://www.docker.com/products/docker-desktop/" -ForegroundColor Yellow
            exit 1
        }

        # Check for required env vars
        if (-not (Test-Path ".env")) {
            Write-Host "ERROR: .env file not found." -ForegroundColor Red
            Write-Host "Create .env with at least DEEPSEEK_API_KEY." -ForegroundColor Yellow
            exit 1
        }

        $envContent = Get-Content ".env" -Raw
        if ($envContent -notmatch "LANGSMITH_API_KEY|LANGGRAPH_CLOUD_LICENSE_KEY") {
            Write-Host "WARNING: Neither LANGSMITH_API_KEY nor LANGGRAPH_CLOUD_LICENSE_KEY found in .env" -ForegroundColor Yellow
            Write-Host "You may need one of these for the Docker deployment to work." -ForegroundColor Yellow
            Write-Host ""
        }

        Write-Host "Running: langgraph up --port $Port" -ForegroundColor Gray
        uv run langgraph up --port $Port
    }
    else {
        Write-Host "`n=== Starting A-Share Agent (Dev - Local Server) ===" -ForegroundColor Green
        Write-Host "Using pickle-based persistence (.langgraph_api/ directory)." -ForegroundColor Cyan
        Write-Host "Per-user thread isolation is enforced by the chat-ui proxy." -ForegroundColor Cyan
        Write-Host ""

        $args = @("run", "langgraph", "dev", "--port", "$Port")
        if ($NoReload) { $args += "--no-reload" }
        if ($NoBrowser) { $args += "--no-browser" }

        Write-Host "Running: uv $($args -join ' ')" -ForegroundColor Gray
        & uv @args
    }
}
finally {
    Pop-Location
}
