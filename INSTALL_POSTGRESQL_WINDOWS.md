# INSTALL POSTGRESQL ON WINDOWS

**Date**: December 1, 2025  
**Status**: ✅ COMPLETE INSTALLATION GUIDE

---

## 🎯 OVERVIEW

This guide will help you install PostgreSQL on Windows and set it up as a service.

---

## 📥 STEP 1: DOWNLOAD POSTGRESQL

### Option A: Official Installer (Recommended)

1. Go to https://www.postgresql.org/download/windows/
2. Click "Download the installer"
3. Choose version 15 or latest
4. Download the `.exe` file

### Option B: Direct Download Links

**PostgreSQL 15** (Recommended):
- https://www.postgresql.org/ftp/source/v15.5/postgresql-15.5.tar.gz
- Or Windows installer: https://sbp.enterprisedb.com/getfile.jsp?fileid=1258453

**PostgreSQL 14**:
- https://sbp.enterprisedb.com/getfile.jsp?fileid=1258452

---

## 💾 STEP 2: INSTALL POSTGRESQL

### Installation Steps

1. **Run the Installer**
   - Double-click the `.exe` file
   - Click "Next" to proceed

2. **Select Installation Directory**
   - Default: `C:\Program Files\PostgreSQL\15`
   - Click "Next"

3. **Select Components**
   - ✅ PostgreSQL Server
   - ✅ pgAdmin 4
   - ✅ Stack Builder
   - ✅ Command Line Tools
   - Click "Next"

4. **Data Directory**
   - Default: `C:\Program Files\PostgreSQL\15\data`
   - Click "Next"

5. **Set Password**
   - Enter password for `postgres` user
   - **Important**: Remember this password!
   - Example: `postgres123`
   - Click "Next"

6. **Port Number**
   - Default: `5432`
   - Click "Next"

7. **Locale**
   - Default: English
   - Click "Next"

8. **Review Summary**
   - Click "Next" to install

9. **Installation Progress**
   - Wait for installation to complete
   - Click "Finish"

---

## ✅ STEP 3: VERIFY INSTALLATION

### Test PostgreSQL

```bash
# Open Command Prompt and run:
psql --version

# Should show: psql (PostgreSQL) 15.x
```

### Test Connection

```bash
# Connect to PostgreSQL
psql -U postgres -h localhost

# Enter password when prompted

# In psql prompt, run:
SELECT version();

# Should show PostgreSQL version

# Exit:
\q
```

---

## 🗄️ STEP 4: CREATE DATABASE & USER

### Create Database

```bash
# Connect to PostgreSQL
psql -U postgres -h localhost

# In psql prompt, run these commands:

-- Create database
CREATE DATABASE forensmart;

-- Create user
CREATE USER forensmart_user WITH PASSWORD 'your_password';

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE forensmart TO forensmart_user;

-- Verify
\l

-- Exit
\q
```

### Verify Database Created

```bash
# List databases
psql -U postgres -h localhost -l

# Should show "forensmart" in the list
```

---

## 🔧 STEP 5: CONFIGURE FOR FORENSMART

### Update .env File

Edit `c:\Forensmart\.env`:

```
DATABASE_URL=postgresql://forensmart_user:your_password@localhost:5432/forensmart
DATABASE_POOL_SIZE=10
DATABASE_MAX_OVERFLOW=20
```

### Test Connection from Python

```bash
# Navigate to ForenSmart
cd c:\Forensmart

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Test connection
python -c "import psycopg2; conn = psycopg2.connect('postgresql://forensmart_user:your_password@localhost:5432/forensmart'); print('Connected!'); conn.close()"

# Should print: Connected!
```

---

## 🔐 STEP 6: SECURE POSTGRESQL

### Change Default Password

```bash
# Connect as postgres user
psql -U postgres -h localhost

# Change password
ALTER USER postgres WITH PASSWORD 'new_secure_password';

# Exit
\q
```

### Create pgpass File (Optional - Auto-login)

Create file: `C:\Users\[YourUsername]\AppData\Roaming\postgresql\pgpass`

Content:
```
localhost:5432:forensmart:forensmart_user:your_password
localhost:5432:*:postgres:postgres_password
```

---

## 🚀 STEP 7: START/STOP POSTGRESQL

### Start PostgreSQL Service

```bash
# Using Services (GUI)
1. Press Win + R
2. Type: services.msc
3. Find "postgresql-x64-15"
4. Right-click → Start

# Or using Command Prompt (as Administrator)
net start postgresql-x64-15

# Or using PowerShell (as Administrator)
Start-Service -Name "postgresql-x64-15"
```

### Stop PostgreSQL Service

```bash
# Using Command Prompt (as Administrator)
net stop postgresql-x64-15

# Or using PowerShell (as Administrator)
Stop-Service -Name "postgresql-x64-15"
```

### Check Status

```bash
# Using PowerShell
Get-Service postgresql-x64-15

# Should show Status: Running
```

---

## 🐛 TROUBLESHOOTING

### PostgreSQL Won't Start

**Problem**: Service fails to start

**Solution**:
```bash
# Check if port 5432 is in use
netstat -ano | findstr :5432

# If in use, kill the process
taskkill /PID [PID] /F

# Try starting again
net start postgresql-x64-15
```

### Can't Connect to PostgreSQL

**Problem**: Connection refused

**Solution**:
```bash
# Check if PostgreSQL is running
Get-Service postgresql-x64-15

# Verify port is listening
netstat -ano | findstr :5432

# Check PostgreSQL logs
# Location: C:\Program Files\PostgreSQL\15\data\pg_log\

# Try connecting with correct credentials
psql -U postgres -h localhost
```

### Forgot Password

**Problem**: Can't remember postgres password

**Solution**:
```bash
# Stop PostgreSQL
net stop postgresql-x64-15

# Edit pg_hba.conf
# Location: C:\Program Files\PostgreSQL\15\data\pg_hba.conf

# Find this line:
# host    all             all             127.0.0.1/32            md5

# Change to:
# host    all             all             127.0.0.1/32            trust

# Start PostgreSQL
net start postgresql-x64-15

# Connect without password
psql -U postgres -h localhost

# Change password
ALTER USER postgres WITH PASSWORD 'new_password';

# Exit
\q

# Revert pg_hba.conf back to md5
# Restart PostgreSQL
```

---

## ✅ VERIFICATION CHECKLIST

- [ ] PostgreSQL installer downloaded
- [ ] PostgreSQL installed successfully
- [ ] PostgreSQL service running
- [ ] Can connect with `psql -U postgres`
- [ ] Database `forensmart` created
- [ ] User `forensmart_user` created
- [ ] .env file updated with credentials
- [ ] Python can connect to database
- [ ] Port 5432 is listening

---

## 📊 COMMON ISSUES & SOLUTIONS

| Issue | Solution |
|-------|----------|
| "psql not found" | Add PostgreSQL to PATH or use full path |
| "Connection refused" | PostgreSQL service not running |
| "Password authentication failed" | Wrong password or user doesn't exist |
| "Database does not exist" | Create database with CREATE DATABASE command |
| "Port 5432 already in use" | Kill process using port or change port |

---

## 🎯 NEXT STEPS

Once PostgreSQL is installed and running:

1. ✅ PostgreSQL installed
2. ✅ Database created
3. ✅ User created
4. ⏭️ Install Redis (see INSTALL_REDIS_WINDOWS.md)
5. ⏭️ Update .env file
6. ⏭️ Run ForenSmart

---

**Status**: ✅ COMPLETE INSTALLATION GUIDE  
**Ready to Install**: YES

