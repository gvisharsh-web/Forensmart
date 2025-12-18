@echo off
REM ============================================================================
REM FORENSMART DATABASE SETUP SCRIPT
REM Setup PostgreSQL and Redis as Windows Services
REM ============================================================================

setlocal enabledelayedexpansion

echo.
echo ============================================================================
echo FORENSMART DATABASE SETUP
echo ============================================================================
echo.

REM Check if running as Administrator
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo [ERROR] This script must be run as Administrator
    echo Please right-click and select "Run as Administrator"
    pause
    exit /b 1
)

echo [INFO] Running as Administrator
echo.

REM ============================================================================
REM POSTGRESQL SETUP
REM ============================================================================

echo ============================================================================
echo POSTGRESQL SETUP
echo ============================================================================
echo.

REM Check if PostgreSQL is installed
where psql >nul 2>&1
if %errorLevel% equ 0 (
    echo [OK] PostgreSQL is installed
    
    REM Check if service exists
    sc query PostgreSQL >nul 2>&1
    if %errorLevel% equ 0 (
        echo [OK] PostgreSQL service already exists
        
        REM Check if service is running
        for /f "tokens=3" %%a in ('sc query PostgreSQL ^| findstr STATE') do (
            if "%%a"=="RUNNING" (
                echo [OK] PostgreSQL service is RUNNING
            ) else (
                echo [WARNING] PostgreSQL service is not running
                echo [ACTION] Starting PostgreSQL service...
                net start PostgreSQL
                if %errorLevel% equ 0 (
                    echo [OK] PostgreSQL service started successfully
                ) else (
                    echo [ERROR] Failed to start PostgreSQL service
                )
            )
        )
    ) else (
        echo [WARNING] PostgreSQL service not found
        echo [ACTION] Attempting to register PostgreSQL as service...
        
        REM Try to find PostgreSQL installation
        if exist "C:\Program Files\PostgreSQL\15\bin\pg_ctl.exe" (
            cd /d "C:\Program Files\PostgreSQL\15\bin"
            pg_ctl register -N "PostgreSQL" -D "C:\Program Files\PostgreSQL\15\data"
            if %errorLevel% equ 0 (
                echo [OK] PostgreSQL service registered
                net start PostgreSQL
                echo [OK] PostgreSQL service started
            ) else (
                echo [ERROR] Failed to register PostgreSQL service
            )
        ) else if exist "C:\Program Files (x86)\PostgreSQL\15\bin\pg_ctl.exe" (
            cd /d "C:\Program Files (x86)\PostgreSQL\15\bin"
            pg_ctl register -N "PostgreSQL" -D "C:\Program Files (x86)\PostgreSQL\15\data"
            if %errorLevel% equ 0 (
                echo [OK] PostgreSQL service registered
                net start PostgreSQL
                echo [OK] PostgreSQL service started
            ) else (
                echo [ERROR] Failed to register PostgreSQL service
            )
        ) else (
            echo [ERROR] PostgreSQL installation not found in standard locations
            echo [ACTION] Please install PostgreSQL from https://www.postgresql.org/download/
        )
    )
) else (
    echo [ERROR] PostgreSQL is not installed or not in PATH
    echo [ACTION] Please install PostgreSQL from https://www.postgresql.org/download/
)

echo.

REM ============================================================================
REM REDIS SETUP
REM ============================================================================

echo ============================================================================
echo REDIS SETUP
echo ============================================================================
echo.

REM Check if Redis is installed
where redis-cli >nul 2>&1
if %errorLevel% equ 0 (
    echo [OK] Redis is installed
    
    REM Check if service exists
    sc query Redis >nul 2>&1
    if %errorLevel% equ 0 (
        echo [OK] Redis service already exists
        
        REM Check if service is running
        for /f "tokens=3" %%a in ('sc query Redis ^| findstr STATE') do (
            if "%%a"=="RUNNING" (
                echo [OK] Redis service is RUNNING
            ) else (
                echo [WARNING] Redis service is not running
                echo [ACTION] Starting Redis service...
                net start Redis
                if %errorLevel% equ 0 (
                    echo [OK] Redis service started successfully
                ) else (
                    echo [ERROR] Failed to start Redis service
                )
            )
        )
    ) else (
        echo [WARNING] Redis service not found
        echo [ACTION] Attempting to register Redis as service...
        
        REM Try to find Redis installation
        if exist "C:\Program Files\Redis\redis-server.exe" (
            cd /d "C:\Program Files\Redis"
            redis-server --service-install redis.windows.conf --service-name Redis
            if %errorLevel% equ 0 (
                echo [OK] Redis service registered
                redis-server --service-start
                echo [OK] Redis service started
            ) else (
                echo [ERROR] Failed to register Redis service
            )
        ) else if exist "C:\Program Files (x86)\Redis\redis-server.exe" (
            cd /d "C:\Program Files (x86)\Redis"
            redis-server --service-install redis.windows.conf --service-name Redis
            if %errorLevel% equ 0 (
                echo [OK] Redis service registered
                redis-server --service-start
                echo [OK] Redis service started
            ) else (
                echo [ERROR] Failed to register Redis service
            )
        ) else (
            echo [ERROR] Redis installation not found in standard locations
            echo [ACTION] Please install Redis from https://redis.io/download
        )
    )
) else (
    echo [ERROR] Redis is not installed or not in PATH
    echo [ACTION] Please install Redis from https://redis.io/download
)

echo.

REM ============================================================================
REM VERIFICATION
REM ============================================================================

echo ============================================================================
echo VERIFICATION
echo ============================================================================
echo.

echo [ACTION] Testing PostgreSQL connection...
psql -U postgres -h localhost -c "SELECT version();" >nul 2>&1
if %errorLevel% equ 0 (
    echo [OK] PostgreSQL connection successful
) else (
    echo [ERROR] PostgreSQL connection failed
)

echo.

echo [ACTION] Testing Redis connection...
redis-cli ping >nul 2>&1
if %errorLevel% equ 0 (
    echo [OK] Redis connection successful
) else (
    echo [ERROR] Redis connection failed
)

echo.

REM ============================================================================
REM SUMMARY
REM ============================================================================

echo ============================================================================
echo SETUP COMPLETE
echo ============================================================================
echo.
echo Next Steps:
echo 1. Update .env file with database credentials
echo 2. Create forensmart database and user in PostgreSQL
echo 3. Run: streamlit run app.py
echo.
echo PostgreSQL Service Status:
sc query PostgreSQL | findstr STATE
echo.
echo Redis Service Status:
sc query Redis | findstr STATE
echo.

pause
