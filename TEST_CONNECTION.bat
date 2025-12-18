@echo off
set PGPASSWORD=Viszz290
"C:\Program Files\PostgreSQL\18\bin\psql.exe" -U forensmart_user -h localhost -d forensmart -c "SELECT 1;"
