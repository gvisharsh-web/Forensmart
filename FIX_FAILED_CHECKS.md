# 🔧 FORENSMART - FIX FAILED CHECKS

**Date**: November 26, 2025
**Status**: ✅ FIXES PROVIDED
**Issues**: 3 failed checks

---

## ❌ FAILED CHECKS SUMMARY

```
✗ FAILED - PostgreSQL: 'psql' is not recognized
✗ FAILED - Redis: 'redis-cli' is not recognized
✗ FAILED - frontend/tsconfig.json: Not found
```

---

## 🔧 FIX 1: INSTALL POSTGRESQL

### **Windows**

#### **Option A: Download Installer (Recommended)**
1. Visit: https://www.postgresql.org/download/windows/
2. Download PostgreSQL 15 or latest
3. Run installer
4. Follow setup wizard
5. Remember the password for `postgres` user
6. Keep port as 5432
7. Complete installation

#### **Option B: Using Chocolatey**
```bash
choco install postgresql
```

#### **Option C: Using Windows Package Manager**
```bash
winget install PostgreSQL.PostgreSQL
```

#### **Verify Installation**
After installation, restart CMD and run:
```bash
psql --version
```

**Expected Output**: `psql (PostgreSQL) 15.x` or higher

---

### **macOS**

#### **Using Homebrew (Recommended)**
```bash
brew install postgresql@15
brew services start postgresql@15
```

#### **Verify Installation**
```bash
psql --version
```

---

### **Linux (Ubuntu/Debian)**

```bash
sudo apt update
sudo apt install postgresql postgresql-contrib -y
sudo systemctl start postgresql
sudo systemctl enable postgresql
```

#### **Verify Installation**
```bash
psql --version
```

---

## 🔧 FIX 2: INSTALL REDIS

### **Windows**

#### **Option A: Using WSL (Windows Subsystem for Linux)**
```bash
# Open PowerShell as Administrator
wsl

# Inside WSL
sudo apt update
sudo apt install redis-server -y
redis-server
```

#### **Option B: Using Chocolatey**
```bash
choco install redis-64
```

#### **Option C: Using Windows Package Manager**
```bash
winget install tporadowski.redis
```

#### **Option D: Manual Installation**
1. Download from: https://github.com/microsoftarchive/redis/releases
2. Extract to a folder
3. Run `redis-server.exe`

#### **Verify Installation**
```bash
redis-cli --version
```

**Expected Output**: `Redis server v=6.x.x` or higher

---

### **macOS**

#### **Using Homebrew (Recommended)**
```bash
brew install redis
brew services start redis
```

#### **Verify Installation**
```bash
redis-cli --version
```

---

### **Linux (Ubuntu/Debian)**

```bash
sudo apt update
sudo apt install redis-server -y
sudo systemctl start redis-server
sudo systemctl enable redis-server
```

#### **Verify Installation**
```bash
redis-cli --version
```

---

## 🔧 FIX 3: CREATE MISSING tsconfig.json

### **Step 1: Navigate to Frontend Directory**
```bash
cd frontend
```

### **Step 2: Create tsconfig.json**

Create a new file `frontend/tsconfig.json` with this content:

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "esModuleInterop": true,
    "allowSyntheticDefaultImports": true,

    /* Bundler mode */
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",

    /* Linting */
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true
  },
  "include": ["src"],
  "references": [{ "path": "./tsconfig.node.json" }]
}
```

### **Step 3: Create tsconfig.node.json**

Create a new file `frontend/tsconfig.node.json` with this content:

```json
{
  "compilerOptions": {
    "composite": true,
    "skipLibCheck": true,
    "module": "ESNext",
    "moduleResolution": "bundler",
    "allowSyntheticDefaultImports": true
  },
  "include": ["vite.config.ts"]
}
```

### **Step 4: Return to Root**
```bash
cd ..
```

---

## 📋 COMPLETE FIX SEQUENCE

### **Windows**

```bash
REM 1. Install PostgreSQL
REM Download from https://www.postgresql.org/download/windows/
REM Or use: choco install postgresql

REM 2. Install Redis
REM Download from https://github.com/microsoftarchive/redis/releases
REM Or use: choco install redis-64

REM 3. Create tsconfig.json
cd frontend

REM Create tsconfig.json file with content above
REM (Use your editor to create the file)

REM Create tsconfig.node.json file with content above
REM (Use your editor to create the file)

cd ..

REM 4. Verify installations
psql --version
redis-cli --version

REM 5. Run verification again
python check_installation.py
```

### **macOS**

```bash
# 1. Install PostgreSQL
brew install postgresql@15
brew services start postgresql@15

# 2. Install Redis
brew install redis
brew services start redis

# 3. Create tsconfig.json
cd frontend

# Create tsconfig.json file with content above
# (Use your editor to create the file)

# Create tsconfig.node.json file with content above
# (Use your editor to create the file)

cd ..

# 4. Verify installations
psql --version
redis-cli --version

# 5. Run verification again
python check_installation.py
```

### **Linux**

```bash
# 1. Install PostgreSQL
sudo apt update
sudo apt install postgresql postgresql-contrib -y
sudo systemctl start postgresql

# 2. Install Redis
sudo apt install redis-server -y
sudo systemctl start redis-server

# 3. Create tsconfig.json
cd frontend

# Create tsconfig.json file with content above
# (Use your editor to create the file)

# Create tsconfig.node.json file with content above
# (Use your editor to create the file)

cd ..

# 4. Verify installations
psql --version
redis-cli --version

# 5. Run verification again
python check_installation.py
```

---

## ✅ AFTER FIXES

### **Step 1: Verify PostgreSQL**
```bash
psql --version
```

**Expected**: `psql (PostgreSQL) 15.x` or higher

### **Step 2: Verify Redis**
```bash
redis-cli --version
```

**Expected**: `Redis server v=6.x` or higher

### **Step 3: Verify tsconfig.json**

#### **Windows**
```bash
dir frontend\tsconfig.json
```

#### **macOS/Linux**
```bash
ls -la frontend/tsconfig.json
```

**Expected**: File exists

### **Step 4: Run Verification Again**
```bash
python check_installation.py
```

**Expected Output**:
```
==================================================
✓ SUCCESS - All critical checks passed!
==================================================

Passed:  47
Failed:  0
Warnings: 0
```

---

## 🚀 AFTER ALL FIXES

Once all checks pass, you can start the services:

### **Terminal 1: Backend API**
```bash
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux
uvicorn main:app --reload
```

### **Terminal 2: Frontend**
```bash
cd frontend
npm run dev
```

### **Terminal 3: Redis**
```bash
redis-server
```

### **Terminal 4: PostgreSQL (if needed)**
```bash
# PostgreSQL usually runs as a service
# Windows: Services > PostgreSQL
# macOS: brew services list
# Linux: sudo systemctl status postgresql
```

---

## 📊 EXPECTED FINAL VERIFICATION

```
==================================================
VERIFICATION SUMMARY
==================================================

Passed:  47
Failed:  0
Warnings: 0
Total:   47

==================================================
✓ SUCCESS - All critical checks passed!
==================================================

Ready to start services:
  Terminal 1: uvicorn main:app --reload
  Terminal 2: cd frontend && npm run dev
  Terminal 3: redis-server
```

---

## 🔍 TROUBLESHOOTING

### **PostgreSQL Still Not Found**

1. **Check if installed**:
   ```bash
   # Windows
   "C:\Program Files\PostgreSQL\15\bin\psql.exe" --version
   ```

2. **Add to PATH**:
   - Windows: Add `C:\Program Files\PostgreSQL\15\bin` to PATH
   - macOS/Linux: Usually automatic with Homebrew

3. **Restart terminal** after adding to PATH

### **Redis Still Not Found**

1. **Check if installed**:
   ```bash
   # Windows (if using WSL)
   wsl redis-cli --version
   ```

2. **Add to PATH**:
   - Windows: Add Redis installation folder to PATH
   - macOS/Linux: Usually automatic with Homebrew

3. **Restart terminal** after adding to PATH

### **tsconfig.json Creation Issues**

If you can't create files manually:

```bash
# Windows
cd frontend
echo {} > tsconfig.json
cd ..
```

Then edit the file with the content provided above.

---

## 📋 FINAL CHECKLIST

- [ ] PostgreSQL installed and accessible (`psql --version` works)
- [ ] Redis installed and accessible (`redis-cli --version` works)
- [ ] `frontend/tsconfig.json` created
- [ ] `frontend/tsconfig.node.json` created
- [ ] Verification script passes all checks
- [ ] Ready to start services

---

**Status**: ✅ **ALL FIXES PROVIDED**

**Next action**: Install PostgreSQL and Redis, create tsconfig files

**Time to complete**: 10-20 minutes

