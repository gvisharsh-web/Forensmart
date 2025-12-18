# INSTALL REDIS ON WINDOWS

**Date**: December 1, 2025  
**Status**: ✅ COMPLETE INSTALLATION GUIDE

---

## 🎯 OVERVIEW

This guide will help you install Redis on Windows and set it up as a service.

---

## 📥 STEP 1: DOWNLOAD REDIS

### Option A: Windows Subsystem for Linux (WSL) - Recommended

Redis doesn't have an official Windows version, but you can use WSL:

1. Install WSL2:
   ```bash
   wsl --install
   ```

2. Install Redis in WSL:
   ```bash
   sudo apt-get update
   sudo apt-get install redis-server
   ```

3. Start Redis:
   ```bash
   redis-server
   ```

### Option B: Redis Windows Port (Community)

Download from: https://github.com/microsoftarchive/redis/releases

**Latest Release**: Redis 3.2.100

Download: `Redis-x64-3.2.100.msi`

---

## 💾 STEP 2: INSTALL REDIS (Option B)

### Installation Steps

1. **Run the Installer**
   - Double-click `Redis-x64-3.2.100.msi`
   - Click "Next" to proceed

2. **Select Installation Directory**
   - Default: `C:\Program Files\Redis`
   - Click "Next"

3. **Configure Redis Port**
   - Default: `6379`
   - Click "Next"

4. **Memory Limit**
   - Default: 512 MB
   - Click "Next"

5. **Service Configuration**
   - ✅ Install as Service
   - Click "Next"

6. **Review Summary**
   - Click "Install"

7. **Installation Complete**
   - Click "Finish"

---

## ✅ STEP 3: VERIFY INSTALLATION

### Test Redis

```bash
# Open Command Prompt and run:
redis-cli --version

# Should show: redis-cli 3.2.100
```

### Test Connection

```bash
# Connect to Redis
redis-cli

# In redis prompt, run:
ping

# Should return: PONG

# Exit:
exit
```

---

## 🔧 STEP 4: CONFIGURE FOR FORENSMART

### Update .env File

Edit `c:\Forensmart\.env`:

```
REDIS_URL=redis://localhost:6379/0
REDIS_PASSWORD=
REDIS_DB=0
```

### Test Connection from Python

```bash
# Navigate to ForenSmart
cd c:\Forensmart

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Test connection
python -c "import redis; r = redis.Redis(host='localhost', port=6379, db=0); print(r.ping()); print('Connected!')"

# Should print: True and Connected!
```

---

## 🔐 STEP 5: SECURE REDIS (Optional)

### Set Redis Password

1. Find Redis configuration file:
   - Location: `C:\Program Files\Redis\redis.windows.conf`

2. Edit the file:
   - Find: `# requirepass foobared`
   - Change to: `requirepass your_secure_password`

3. Restart Redis service:
   ```bash
   net stop Redis
   net start Redis
   ```

4. Update .env:
   ```
   REDIS_URL=redis://:your_secure_password@localhost:6379/0
   ```

---

## 🚀 STEP 6: START/STOP REDIS

### Start Redis Service

```bash
# Using Services (GUI)
1. Press Win + R
2. Type: services.msc
3. Find "Redis"
4. Right-click → Start

# Or using Command Prompt (as Administrator)
net start Redis

# Or using PowerShell (as Administrator)
Start-Service -Name "Redis"
```

### Stop Redis Service

```bash
# Using Command Prompt (as Administrator)
net stop Redis

# Or using PowerShell (as Administrator)
Stop-Service -Name "Redis"
```

### Check Status

```bash
# Using PowerShell
Get-Service Redis

# Should show Status: Running
```

---

## 🐛 TROUBLESHOOTING

### Redis Won't Start

**Problem**: Service fails to start

**Solution**:
```bash
# Check if port 6379 is in use
netstat -ano | findstr :6379

# If in use, kill the process
taskkill /PID [PID] /F

# Try starting again
net start Redis
```

### Can't Connect to Redis

**Problem**: Connection refused

**Solution**:
```bash
# Check if Redis is running
Get-Service Redis

# Verify port is listening
netstat -ano | findstr :6379

# Try connecting
redis-cli ping

# Should return: PONG
```

### Redis Service Not Found

**Problem**: Redis service not registered

**Solution**:
```bash
# Navigate to Redis directory
cd "C:\Program Files\Redis"

# Register as service
redis-server --service-install redis.windows.conf --service-name Redis

# Start service
redis-server --service-start
```

---

## ✅ VERIFICATION CHECKLIST

- [ ] Redis installer downloaded
- [ ] Redis installed successfully
- [ ] Redis service running
- [ ] Can connect with `redis-cli ping`
- [ ] Returns `PONG`
- [ ] .env file updated with credentials
- [ ] Python can connect to Redis
- [ ] Port 6379 is listening

---

## 📊 COMMON ISSUES & SOLUTIONS

| Issue | Solution |
|-------|----------|
| "redis-cli not found" | Add Redis to PATH or use full path |
| "Connection refused" | Redis service not running |
| "Port 6379 already in use" | Kill process using port or change port |
| "WRONGPASS" | Wrong password set in redis.conf |
| "Service not found" | Register service with --service-install |

---

## 🎯 NEXT STEPS

Once Redis is installed and running:

1. ✅ PostgreSQL installed
2. ✅ Redis installed
3. ⏭️ Update .env file with both credentials
4. ⏭️ Run ForenSmart

---

## 📝 ALTERNATIVE: USE DOCKER

If you prefer, you can run Redis in Docker:

```bash
# Install Docker Desktop from https://www.docker.com/products/docker-desktop

# Run Redis container
docker run -d -p 6379:6379 redis:latest

# Verify
docker ps

# Should show redis container running
```

---

**Status**: ✅ COMPLETE INSTALLATION GUIDE  
**Ready to Install**: YES

