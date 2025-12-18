# ============================================================================
# FORENSMART DATABASE SETUP SCRIPT (PowerShell)
# Setup PostgreSQL and Redis as Windows Services
# ============================================================================

# Requires Administrator privileges
if (-NOT ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
    Write-Host "[ERROR] This script must be run as Administrator" -ForegroundColor Red
    Write-Host "Please right-click PowerShell and select 'Run as Administrator'" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "FORENSMART DATABASE SETUP" -ForegroundColor Cyan
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host ""

# ============================================================================
# POSTGRESQL SETUP
# ============================================================================

Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "POSTGRESQL SETUP" -ForegroundColor Cyan
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host ""

# Check if PostgreSQL is installed
$psqlPath = Get-Command psql -ErrorAction SilentlyContinue
if ($psqlPath) {
    Write-Host "[OK] PostgreSQL is installed" -ForegroundColor Green
    
    # Check if service exists
    $postgresService = Get-Service -Name "PostgreSQL" -ErrorAction SilentlyContinue
    if ($postgresService) {
        Write-Host "[OK] PostgreSQL service already exists" -ForegroundColor Green
        
        # Check if service is running
        if ($postgresService.Status -eq "Running") {
            Write-Host "[OK] PostgreSQL service is RUNNING" -ForegroundColor Green
        } else {
            Write-Host "[WARNING] PostgreSQL service is not running" -ForegroundColor Yellow
            Write-Host "[ACTION] Starting PostgreSQL service..." -ForegroundColor Yellow
            Start-Service -Name "PostgreSQL"
            Write-Host "[OK] PostgreSQL service started" -ForegroundColor Green
        }
    } else {
        Write-Host "[WARNING] PostgreSQL service not found" -ForegroundColor Yellow
        Write-Host "[ACTION] Attempting to register PostgreSQL as service..." -ForegroundColor Yellow
        
        # Try to find PostgreSQL installation
        $pgPaths = @(
            "C:\Program Files\PostgreSQL\15\bin\pg_ctl.exe",
            "C:\Program Files (x86)\PostgreSQL\15\bin\pg_ctl.exe",
            "C:\Program Files\PostgreSQL\14\bin\pg_ctl.exe",
            "C:\Program Files (x86)\PostgreSQL\14\bin\pg_ctl.exe"
        )
        
        $pgFound = $false
        foreach ($pgPath in $pgPaths) {
            if (Test-Path $pgPath) {
                $pgDir = Split-Path -Parent $pgPath
                $dataDir = $pgDir -replace "\\bin$", "\data"
                
                Write-Host "[ACTION] Found PostgreSQL at: $pgDir" -ForegroundColor Yellow
                
                Push-Location $pgDir
                & .\pg_ctl.exe register -N "PostgreSQL" -D $dataDir
                Pop-Location
                
                if ($LASTEXITCODE -eq 0) {
                    Write-Host "[OK] PostgreSQL service registered" -ForegroundColor Green
                    Start-Service -Name "PostgreSQL"
                    Write-Host "[OK] PostgreSQL service started" -ForegroundColor Green
                    $pgFound = $true
                    break
                }
            }
        }
        
        if (-not $pgFound) {
            Write-Host "[ERROR] PostgreSQL installation not found" -ForegroundColor Red
            Write-Host "[ACTION] Please install PostgreSQL from https://www.postgresql.org/download/" -ForegroundColor Yellow
        }
    }
} else {
    Write-Host "[ERROR] PostgreSQL is not installed or not in PATH" -ForegroundColor Red
    Write-Host "[ACTION] Please install PostgreSQL from https://www.postgresql.org/download/" -ForegroundColor Yellow
}

Write-Host ""

# ============================================================================
# REDIS SETUP
# ============================================================================

Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "REDIS SETUP" -ForegroundColor Cyan
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host ""

# Check if Redis is installed
$redisPath = Get-Command redis-cli -ErrorAction SilentlyContinue
if ($redisPath) {
    Write-Host "[OK] Redis is installed" -ForegroundColor Green
    
    # Check if service exists
    $redisService = Get-Service -Name "Redis" -ErrorAction SilentlyContinue
    if ($redisService) {
        Write-Host "[OK] Redis service already exists" -ForegroundColor Green
        
        # Check if service is running
        if ($redisService.Status -eq "Running") {
            Write-Host "[OK] Redis service is RUNNING" -ForegroundColor Green
        } else {
            Write-Host "[WARNING] Redis service is not running" -ForegroundColor Yellow
            Write-Host "[ACTION] Starting Redis service..." -ForegroundColor Yellow
            Start-Service -Name "Redis"
            Write-Host "[OK] Redis service started" -ForegroundColor Green
        }
    } else {
        Write-Host "[WARNING] Redis service not found" -ForegroundColor Yellow
        Write-Host "[ACTION] Attempting to register Redis as service..." -ForegroundColor Yellow
        
        # Try to find Redis installation
        $redisPaths = @(
            "C:\Program Files\Redis\redis-server.exe",
            "C:\Program Files (x86)\Redis\redis-server.exe",
            "C:\Redis\redis-server.exe"
        )
        
        $redisFound = $false
        foreach ($redisPath in $redisPaths) {
            if (Test-Path $redisPath) {
                $redisDir = Split-Path -Parent $redisPath
                
                Write-Host "[ACTION] Found Redis at: $redisDir" -ForegroundColor Yellow
                
                Push-Location $redisDir
                & .\redis-server.exe --service-install redis.windows.conf --service-name Redis
                Pop-Location
                
                if ($LASTEXITCODE -eq 0) {
                    Write-Host "[OK] Redis service registered" -ForegroundColor Green
                    & redis-server --service-start
                    Write-Host "[OK] Redis service started" -ForegroundColor Green
                    $redisFound = $true
                    break
                }
            }
        }
        
        if (-not $redisFound) {
            Write-Host "[ERROR] Redis installation not found" -ForegroundColor Red
            Write-Host "[ACTION] Please install Redis from https://redis.io/download" -ForegroundColor Yellow
        }
    }
} else {
    Write-Host "[ERROR] Redis is not installed or not in PATH" -ForegroundColor Red
    Write-Host "[ACTION] Please install Redis from https://redis.io/download" -ForegroundColor Yellow
}

Write-Host ""

# ============================================================================
# VERIFICATION
# ============================================================================

Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "VERIFICATION" -ForegroundColor Cyan
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "[ACTION] Testing PostgreSQL connection..." -ForegroundColor Yellow
$pgTest = psql -U postgres -h localhost -c "SELECT version();" 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] PostgreSQL connection successful" -ForegroundColor Green
} else {
    Write-Host "[ERROR] PostgreSQL connection failed" -ForegroundColor Red
}

Write-Host ""

Write-Host "[ACTION] Testing Redis connection..." -ForegroundColor Yellow
$redisTest = redis-cli ping 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] Redis connection successful" -ForegroundColor Green
} else {
    Write-Host "[ERROR] Redis connection failed" -ForegroundColor Red
}

Write-Host ""

# ============================================================================
# SUMMARY
# ============================================================================

Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "SETUP COMPLETE" -ForegroundColor Cyan
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Yellow
Write-Host "1. Update .env file with database credentials" -ForegroundColor White
Write-Host "2. Create forensmart database and user in PostgreSQL" -ForegroundColor White
Write-Host "3. Run: streamlit run app.py" -ForegroundColor White
Write-Host ""

Write-Host "PostgreSQL Service Status:" -ForegroundColor Yellow
Get-Service -Name "PostgreSQL" -ErrorAction SilentlyContinue | Format-Table -AutoSize

Write-Host "Redis Service Status:" -ForegroundColor Yellow
Get-Service -Name "Redis" -ErrorAction SilentlyContinue | Format-Table -AutoSize

Write-Host ""
Read-Host "Press Enter to exit"
