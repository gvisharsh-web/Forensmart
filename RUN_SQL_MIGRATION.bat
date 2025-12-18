@echo off
REM Run SQL migration to create approval tables

echo.
echo ============================================================================
echo CREATING APPROVAL TABLES
echo ============================================================================
echo.

set PGPASSWORD=Viszz290

echo [INFO] Running SQL migration...
echo.

"C:\Program Files\PostgreSQL\18\bin\psql.exe" -U forensmart_user -h localhost -d forensmart -f c:\Forensmart\CREATE_APPROVAL_TABLES.sql

echo.
echo [OK] Migration complete!
echo.

REM Verify tables created
echo [INFO] Verifying tables...
echo.

"C:\Program Files\PostgreSQL\18\bin\psql.exe" -U forensmart_user -h localhost -d forensmart -c "\dt"

echo.
echo ============================================================================
echo APPROVAL TABLES CREATED SUCCESSFULLY
echo ============================================================================
echo.

pause
