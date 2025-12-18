@echo off
REM Grant schema permissions to forensmart_user

echo.
echo ============================================================================
echo GRANTING SCHEMA PERMISSIONS
echo ============================================================================
echo.

set PGPASSWORD=Viszz290

echo [INFO] Granting permissions to forensmart_user...
echo.

"C:\Program Files\PostgreSQL\18\bin\psql.exe" -U postgres -h localhost -d forensmart -c "GRANT ALL ON SCHEMA public TO forensmart_user;"
"C:\Program Files\PostgreSQL\18\bin\psql.exe" -U postgres -h localhost -d forensmart -c "ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO forensmart_user;"
"C:\Program Files\PostgreSQL\18\bin\psql.exe" -U postgres -h localhost -d forensmart -c "ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO forensmart_user;"

echo [OK] Permissions granted!
echo.

pause
