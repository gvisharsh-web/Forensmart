@echo off
REM Fix forensmart_user password

echo.
echo ============================================================================
echo FIXING FORENSMART USER PASSWORD
echo ============================================================================
echo.

set PGPASSWORD=Viszz290

echo [INFO] Connecting to PostgreSQL as postgres user...
echo.

REM Drop existing user if exists
"C:\Program Files\PostgreSQL\18\bin\psql.exe" -U postgres -h localhost -c "DROP USER IF EXISTS forensmart_user;" 2>nul

REM Create new user with correct password
"C:\Program Files\PostgreSQL\18\bin\psql.exe" -U postgres -h localhost -c "CREATE USER forensmart_user WITH PASSWORD 'Viszz290';" 2>nul

REM Grant privileges
"C:\Program Files\PostgreSQL\18\bin\psql.exe" -U postgres -h localhost -c "GRANT ALL PRIVILEGES ON DATABASE forensmart TO forensmart_user;" 2>nul

echo [OK] User password fixed!
echo.

REM Test connection
echo [INFO] Testing connection...
"C:\Program Files\PostgreSQL\18\bin\psql.exe" -U forensmart_user -h localhost -d forensmart -c "SELECT 1;" >nul 2>&1

if %errorlevel% equ 0 (
    echo [OK] Connection successful!
    echo.
) else (
    echo [ERROR] Connection failed
    echo.
)

pause
