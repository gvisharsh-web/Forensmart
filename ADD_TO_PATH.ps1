# Add PostgreSQL and Redis to System PATH
# Run this script as Administrator

# Check if running as Administrator
if (-NOT ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
    Write-Host "[ERROR] This script must be run as Administrator" -ForegroundColor Red
    Write-Host "Please right-click PowerShell and select 'Run as Administrator'" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "ADD POSTGRESQL AND REDIS TO SYSTEM PATH" -ForegroundColor Cyan
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host ""

# Get current PATH
$currentPath = [Environment]::GetEnvironmentVariable("Path", "Machine")

# PostgreSQL path
$postgresPath = "C:\Program Files\PostgreSQL\15\bin"

# Redis path
$redisPath = "C:\Program Files\Redis"

# Check if paths already exist
$postgresExists = $currentPath -like "*PostgreSQL*"
$redisExists = $currentPath -like "*Redis*"

Write-Host "[INFO] Current PATH contains:" -ForegroundColor Yellow
Write-Host "  PostgreSQL: $postgresExists" -ForegroundColor White
Write-Host "  Redis: $redisExists" -ForegroundColor White
Write-Host ""

# Add PostgreSQL to PATH if not already there
if (-not $postgresExists) {
    Write-Host "[ACTION] Adding PostgreSQL to PATH..." -ForegroundColor Yellow
    $newPath = $currentPath + ";" + $postgresPath
    [Environment]::SetEnvironmentVariable("Path", $newPath, "Machine")
    Write-Host "[OK] PostgreSQL added to PATH" -ForegroundColor Green
} else {
    Write-Host "[OK] PostgreSQL already in PATH" -ForegroundColor Green
}

Write-Host ""

# Add Redis to PATH if not already there
if (-not $redisExists) {
    Write-Host "[ACTION] Adding Redis to PATH..." -ForegroundColor Yellow
    $currentPath = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $newPath = $currentPath + ";" + $redisPath
    [Environment]::SetEnvironmentVariable("Path", $newPath, "Machine")
    Write-Host "[OK] Redis added to PATH" -ForegroundColor Green
} else {
    Write-Host "[OK] Redis already in PATH" -ForegroundColor Green
}

Write-Host ""
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "PATH UPDATE COMPLETE" -ForegroundColor Cyan
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "IMPORTANT: You must restart PowerShell/Command Prompt for changes to take effect!" -ForegroundColor Yellow
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Close this PowerShell window" -ForegroundColor White
Write-Host "2. Open a NEW PowerShell or Command Prompt window" -ForegroundColor White
Write-Host "3. Run: psql --version" -ForegroundColor White
Write-Host "4. Run: redis-cli --version" -ForegroundColor White
Write-Host ""

Read-Host "Press Enter to exit"
