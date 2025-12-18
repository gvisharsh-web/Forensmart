@echo off
REM Clean up test data from approval tables

echo.
echo ============================================================================
echo CLEANING UP TEST DATA
echo ============================================================================
echo.

set PGPASSWORD=Viszz290

echo [INFO] Deleting test data from approval tables...
echo.

"C:\Program Files\PostgreSQL\18\bin\psql.exe" -U forensmart_user -h localhost -d forensmart -c "DELETE FROM approval_history;"
"C:\Program Files\PostgreSQL\18\bin\psql.exe" -U forensmart_user -h localhost -d forensmart -c "DELETE FROM consent_approvals;"
"C:\Program Files\PostgreSQL\18\bin\psql.exe" -U forensmart_user -h localhost -d forensmart -c "DELETE FROM approval_links;"

echo [OK] Test data deleted!
echo.

REM Verify tables are empty
echo [INFO] Verifying tables are empty...
echo.

"C:\Program Files\PostgreSQL\18\bin\psql.exe" -U forensmart_user -h localhost -d forensmart -c "SELECT COUNT(*) as approval_links_count FROM approval_links;"
"C:\Program Files\PostgreSQL\18\bin\psql.exe" -U forensmart_user -h localhost -d forensmart -c "SELECT COUNT(*) as consent_approvals_count FROM consent_approvals;"
"C:\Program Files\PostgreSQL\18\bin\psql.exe" -U forensmart_user -h localhost -d forensmart -c "SELECT COUNT(*) as approval_history_count FROM approval_history;"

echo.
echo ============================================================================
echo DATABASE CLEANED
echo ============================================================================
echo.

pause
