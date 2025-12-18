@echo off
REM Setup ForenSmart Database
REM This script creates the forensmart database and user

echo.
echo ============================================================================
echo FORENSMART DATABASE SETUP
echo ============================================================================
echo.

REM Set PostgreSQL password
set PGPASSWORD=Viszz290

echo [INFO] Connecting to PostgreSQL as postgres user...
echo.

REM Create database and user
"C:\Program Files\PostgreSQL\18\bin\psql.exe" -U postgres -h localhost -c "CREATE DATABASE forensmart;" 2>nul
"C:\Program Files\PostgreSQL\18\bin\psql.exe" -U postgres -h localhost -c "CREATE USER forensmart_user WITH PASSWORD 'Viszz290';" 2>nul
"C:\Program Files\PostgreSQL\18\bin\psql.exe" -U postgres -h localhost -c "GRANT ALL PRIVILEGES ON DATABASE forensmart TO forensmart_user;" 2>nul

echo [OK] Database setup complete!
echo.

REM Verify connection
echo [INFO] Testing connection to forensmart database...
"C:\Program Files\PostgreSQL\18\bin\psql.exe" -U forensmart_user -h localhost -d forensmart -c "SELECT 1;" >nul 2>&1

if %errorlevel% equ 0 (
    echo [OK] Successfully connected to forensmart database!
    echo.
    echo ============================================================================
    echo DATABASE SETUP SUCCESSFUL
    echo ============================================================================
    echo.
    echo You can now start ForenSmart with:
    echo   cd c:\Forensmart
    echo   .\venv\Scripts\Activate.ps1
    echo   streamlit run app.py
    echo.
) else (
    echo [ERROR] Could not connect to forensmart database
    echo [ERROR] Please check your password
    echo.
)

pause
