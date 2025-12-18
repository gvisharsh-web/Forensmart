# SETUP POSTGRESQL & REDIS AS WINDOWS SERVICES

**Date**: December 1, 2025  
**Status**: ✅ COMPLETE SETUP GUIDE

---

## 🎯 OVERVIEW

You have PostgreSQL and Redis installed but they're not running as services. This guide will help you:
1. Install PostgreSQL as a Windows Service
2. Install Redis as a Windows Service
3. Start and manage the services
4. Verify they're running

---

## 🗄️ POSTGRESQL SETUP AS SERVICE

### Step 1: Check if PostgreSQL is Installed

```bash
# Check if psql command exists
where psql

# If not found, PostgreSQL is not in PATH
# If found, PostgreSQL is installed
```

### Step 2: Find PostgreSQL Installation Directory

**Common Locations**:
- `C:\Program Files\PostgreSQL\15` (or version number)
- `C:\Program Files (x86)\PostgreSQL\15`
- `C:\PostgreSQL\15`

**To Find**:
```bash
# Search for PostgreSQL folder
dir "C:\Program Files" | findstr postgres
```

### Step 3: Install PostgreSQL as Service

**If PostgreSQL is NOT running as service**:

```bash
# Navigate to PostgreSQL bin directory
cd "C:\Program Files\PostgreSQL\15\bin"

# Register PostgreSQL as service
pg_ctl register -N "PostgreSQL" -D "C:\Program Files\PostgreSQL\15\data"

# Start the service
net start PostgreSQL

# Or using PowerShell (as Administrator)
Start-Service -Name PostgreSQL
```

### Step 4: Verify PostgreSQL is Running

```bash
# Check service status
sc query PostgreSQL

# Or using PowerShell
Get-Service PostgreSQL

# Test connection
psql -U postgres -h localhost -c "SELECT version();"
```

### Step 5: Create Database and User

```bash
# Connect to PostgreSQL
psql -U postgres

# Create database
CREATE DATABASE forensmart;

# Create user
CREATE USER forensmart_user WITH PASSWORD 'your_password';

# Grant permissions
GRANT ALL PRIVILEGES ON DATABASE forensmart TO forensmart_user;

# Verify
\l
\du
```

---

## 🔴 REDIS SETUP AS SERVICE

### Step 1: Check if Redis is Installed

```bash
# Check if redis-cli command exists
where redis-cli

# If not found, Redis is not in PATH
# If found, Redis is installed
```

### Step 2: Find Redis Installation Directory

**Common Locations**:
- `C:\Program Files\Redis`
- `C:\Program Files (x86)\Redis`
- `C:\Redis`

**To Find**:
```bash
# Search for Redis folder
dir "C:\Program Files" | findstr redis
```

### Step 3: Install Redis as Service (Windows)

**Option A: Using Redis Windows Service Wrapper**

```bash
# Navigate to Redis directory
cd "C:\Program Files\Redis"

# Install as service
redis-server --service-install redis.windows.conf --service-name Redis

# Start the service
redis-server --service-start

# Or using PowerShell (as Administrator)
Start-Service -Name Redis
```

**Option B: Manual Service Installation**

```bash
# Create a batch file to run Redis
# File: C:\Redis\start-redis.bat
@echo off
"C:\Program Files\Redis\redis-server.exe" "C:\Program Files\Redis\redis.conf"

# Then create a scheduled task or Windows service
```

### Step 4: Verify Redis is Running

```bash
# Test connection
redis-cli ping

# Should return: PONG

# Get Redis info
redis-cli info

# Check memory usage
redis-cli info memory
```

---

## 🚀 QUICK START COMMANDS

### PostgreSQL Service Commands

```bash
# Start PostgreSQL
net start PostgreSQL
# or
Start-Service -Name PostgreSQL

# Stop PostgreSQL
net stop PostgreSQL
# or
Stop-Service -Name PostgreSQL

# Check status
sc query PostgreSQL
# or
Get-Service PostgreSQL

# Restart PostgreSQL
net stop PostgreSQL && net start PostgreSQL
# or
Restart-Service -Name PostgreSQL

# Remove service (if needed)
pg_ctl unregister -N "PostgreSQL"
```

### Redis Service Commands

```bash
# Start Redis
redis-server --service-start
# or
Start-Service -Name Redis

# Stop Redis
redis-server --service-stop
# or
Stop-Service -Name Redis

# Check status
redis-cli ping

# Test connection
redis-cli
> ping
> PONG
> exit
```

---

## 🔧 TROUBLESHOOTING

### PostgreSQL Won't Start

**Problem**: Service fails to start

**Solution**:
```bash
# Check PostgreSQL logs
# Location: C:\Program Files\PostgreSQL\15\data\pg_log\

# Check if port 5432 is in use
netstat -ano | findstr :5432

# If port is in use, kill the process
taskkill /PID [PID] /F

# Try starting again
net start PostgreSQL
```

### Redis Won't Start

**Problem**: Redis service fails to start

**Solution**:
```bash
# Check if port 6379 is in use
netstat -ano | findstr :6379

# If port is in use, kill the process
taskkill /PID [PID] /F

# Try starting again
redis-server --service-start

# Or run Redis manually to see errors
redis-server.exe
```

### Connection Refused

**Problem**: Cannot connect to PostgreSQL or Redis

**Solution**:
```bash
# Verify services are running
Get-Service PostgreSQL
Get-Service Redis

# Check if ports are listening
netstat -ano | findstr :5432
netstat -ano | findstr :6379

# Test connection
psql -U postgres -h localhost
redis-cli ping
```

---

## 📋 VERIFICATION CHECKLIST

### PostgreSQL

- [ ] PostgreSQL installed in `C:\Program Files\PostgreSQL\15`
- [ ] PostgreSQL service registered
- [ ] PostgreSQL service is running
- [ ] Can connect with `psql -U postgres`
- [ ] Database `forensmart` created
- [ ] User `forensmart_user` created
- [ ] Port 5432 is listening

### Redis

- [ ] Redis installed in `C:\Program Files\Redis`
- [ ] Redis service registered
- [ ] Redis service is running
- [ ] Can connect with `redis-cli ping`
- [ ] Port 6379 is listening
- [ ] Redis returns `PONG` on ping

---

## 🔌 TEST CONNECTIONS

### Test PostgreSQL

```bash
# Connect to PostgreSQL
psql -U postgres -h localhost

# In psql prompt:
postgres=# SELECT version();
postgres=# \l
postgres=# \q
```

### Test Redis

```bash
# Connect to Redis
redis-cli

# In redis prompt:
127.0.0.1:6379> ping
PONG
127.0.0.1:6379> SET test "hello"
OK
127.0.0.1:6379> GET test
"hello"
127.0.0.1:6379> exit
```

---

## 📊 SERVICE STATUS COMMANDS

### Check All Services

```bash
# PowerShell (as Administrator)
Get-Service | Where-Object {$_.Name -like "*postgres*" -or $_.Name -like "*redis*"}

# Command Prompt
sc query PostgreSQL
sc query Redis
```

### View Service Details

```bash
# PostgreSQL
Get-Service PostgreSQL | Format-List

# Redis
Get-Service Redis | Format-List
```

---

## 🔐 SECURITY SETUP

### PostgreSQL Security

```bash
# Set PostgreSQL password
psql -U postgres -h localhost

# In psql:
ALTER USER postgres WITH PASSWORD 'new_password';

# Update .env
DATABASE_URL=postgresql://postgres:new_password@localhost:5432/forensmart
```

### Redis Security

```bash
# Set Redis password in redis.conf
# Find: # requirepass foobared
# Change to: requirepass your_password

# Restart Redis
redis-server --service-stop
redis-server --service-start

# Test with password
redis-cli -a your_password ping
```

---

## 📝 FINAL SETUP IN .ENV

```bash
# PostgreSQL
DATABASE_URL=postgresql://postgres:password@localhost:5432/forensmart
DATABASE_POOL_SIZE=10
DATABASE_MAX_OVERFLOW=20

# Redis
REDIS_URL=redis://localhost:6379/0
REDIS_PASSWORD=
REDIS_DB=0
```

---

## ✅ COMPLETE SETUP CHECKLIST

- [ ] PostgreSQL installed as service
- [ ] PostgreSQL service running
- [ ] PostgreSQL database created
- [ ] PostgreSQL user created
- [ ] Redis installed as service
- [ ] Redis service running
- [ ] .env file updated with credentials
- [ ] PostgreSQL connection tested
- [ ] Redis connection tested
- [ ] ForenSmart app can connect to both

---

## 🚀 START USING FORENSMART

Once both services are running:

```bash
# Navigate to ForenSmart
cd c:\Forensmart

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py

# App will be available at
http://localhost:8501
```

---

**Status**: ✅ COMPLETE SETUP GUIDE  
**Ready to Install**: YES

