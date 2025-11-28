# 🚀 FORENSMART - COMPLETE INSTALLATION GUIDE

**Date**: November 26, 2025
**Status**: ✅ INSTALLATION GUIDE COMPLETE
**Scope**: Local installation + Virtual Environment setup for all modules

---

## 📋 TABLE OF CONTENTS

1. [System Requirements](#system-requirements)
2. [Python Setup](#python-setup)
3. [Virtual Environment Setup](#virtual-environment-setup)
4. [Backend Dependencies](#backend-dependencies)
5. [Frontend Dependencies](#frontend-dependencies)
6. [Database Setup](#database-setup)
7. [Installation Commands](#installation-commands)
8. [Verification](#verification)
9. [Troubleshooting](#troubleshooting)

---

## 🖥️ SYSTEM REQUIREMENTS

### **Minimum Requirements**
- **OS**: Windows 10+, macOS 10.14+, Ubuntu 18.04+
- **RAM**: 8 GB minimum, 16 GB recommended
- **Disk**: 20 GB free space
- **CPU**: Dual-core processor minimum

### **Software Requirements**
- **Python**: 3.11+ (3.12 recommended)
- **Node.js**: 18+ (for frontend)
- **PostgreSQL**: 13+ (for database)
- **Redis**: 6+ (for caching)
- **Docker**: 20+ (optional, for containerization)
- **Git**: 2.30+ (for version control)

---

## 🐍 PYTHON SETUP

### **Windows**

#### **Step 1: Download Python**
1. Visit https://www.python.org/downloads/
2. Download Python 3.12 (or latest)
3. Run installer
4. ✅ **CHECK**: "Add Python to PATH"
5. Click "Install Now"

#### **Step 2: Verify Installation**
```bash
python --version
pip --version
```

#### **Step 3: Upgrade pip**
```bash
python -m pip install --upgrade pip
```

### **macOS**

#### **Using Homebrew (Recommended)**
```bash
# Install Homebrew if not installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python@3.12

# Verify
python3 --version
pip3 --version
```

#### **Using Python.org**
1. Download from https://www.python.org/downloads/macos/
2. Run installer
3. Follow prompts

### **Linux (Ubuntu/Debian)**

```bash
# Update package manager
sudo apt update
sudo apt upgrade -y

# Install Python
sudo apt install python3.12 python3.12-venv python3-pip -y

# Verify
python3 --version
pip3 --version
```

### **Linux (Fedora/RHEL)**

```bash
# Install Python
sudo dnf install python3.12 python3.12-devel python3-pip -y

# Verify
python3 --version
pip3 --version
```

---

## 🔐 VIRTUAL ENVIRONMENT SETUP

### **Windows**

#### **Create Virtual Environment**
```bash
# Navigate to project directory
cd c:\Forensmart

# Create venv
python -m venv venv

# Activate venv
venv\Scripts\activate

# You should see (venv) in your terminal
```

#### **Deactivate Virtual Environment**
```bash
deactivate
```

### **macOS/Linux**

#### **Create Virtual Environment**
```bash
# Navigate to project directory
cd ~/Forensmart

# Create venv
python3 -m venv venv

# Activate venv
source venv/bin/activate

# You should see (venv) in your terminal
```

#### **Deactivate Virtual Environment**
```bash
deactivate
```

---

## 📦 BACKEND DEPENDENCIES

### **Location: c:\Forensmart\requirements.txt**

The file contains all Python dependencies organized by category:

#### **Core Dependencies**
```
streamlit>=1.28.0
streamlit-extras>=0.3.0
```

#### **Data Processing**
```
pandas>=1.5.0
numpy>=1.24.0
scipy>=1.11.0
```

#### **Web Framework & API**
```
fastapi>=0.104.0
uvicorn>=0.24.0
requests>=2.31.0
httpx>=0.25.0
```

#### **Database**
```
sqlalchemy>=2.0.0
psycopg2-binary>=2.9.0
redis>=5.0.0
```

#### **Machine Learning & AI**
```
scikit-learn>=1.3.0
tensorflow>=2.13.0
transformers>=4.34.0
openai>=1.0.0
anthropic>=0.7.0
```

#### **Security**
```
cryptography>=41.0.0
pyjwt>=2.8.0
bcrypt>=4.1.0
```

#### **Development & Testing**
```
pytest>=7.4.0
pytest-asyncio>=0.21.0
pytest-cov>=4.1.0
pylint>=3.0.0
flake8>=6.1.0
black>=23.12.0
```

---

## 📦 FRONTEND DEPENDENCIES

### **Location: c:\Forensmart\frontend\package.json**

#### **Core Dependencies**
```json
{
  "react": "^18.2.0",
  "react-dom": "^18.2.0",
  "react-router-dom": "^6.20.0",
  "typescript": "^5.3.0"
}
```

#### **State Management & Forms**
```json
{
  "zustand": "^4.4.0",
  "react-hook-form": "^7.48.0",
  "@hookform/resolvers": "^3.3.0",
  "zod": "^3.22.0"
}
```

#### **HTTP & Data**
```json
{
  "axios": "^1.6.0",
  "recharts": "^2.10.0"
}
```

#### **UI & Styling**
```json
{
  "tailwindcss": "^3.4.0",
  "lucide-react": "^0.294.0",
  "@radix-ui/react-dialog": "^1.1.1"
}
```

---

## 💾 DATABASE SETUP

### **PostgreSQL Installation**

#### **Windows**
1. Download from https://www.postgresql.org/download/windows/
2. Run installer
3. Set password for postgres user
4. Keep default port 5432
5. Complete installation

#### **macOS**
```bash
brew install postgresql@15
brew services start postgresql@15
```

#### **Linux (Ubuntu)**
```bash
sudo apt install postgresql postgresql-contrib -y
sudo systemctl start postgresql
```

### **Create Database**

```bash
# Connect to PostgreSQL
psql -U postgres

# Create database
CREATE DATABASE forensmart;

# Create user
CREATE USER forensmart_user WITH PASSWORD 'secure_password';

# Grant privileges
GRANT ALL PRIVILEGES ON DATABASE forensmart TO forensmart_user;

# Exit
\q
```

### **Redis Installation**

#### **Windows**
1. Download from https://github.com/microsoftarchive/redis/releases
2. Or use WSL: `wsl` then `sudo apt install redis-server`

#### **macOS**
```bash
brew install redis
brew services start redis
```

#### **Linux**
```bash
sudo apt install redis-server -y
sudo systemctl start redis-server
```

---

## 🚀 INSTALLATION COMMANDS

### **STEP 1: Clone Repository**
```bash
git clone https://github.com/yourusername/forensmart.git
cd forensmart
```

### **STEP 2: Backend Setup**

#### **Windows**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Verify installation
pip list
```

#### **macOS/Linux**
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
python3 -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Verify installation
pip list
```

### **STEP 3: Frontend Setup**

#### **All Platforms**
```bash
# Navigate to frontend directory
cd frontend

# Install Node.js dependencies
npm install

# Verify installation
npm list

# Return to root
cd ..
```

### **STEP 4: Environment Configuration**

#### **Create .env file**
```bash
# Create .env in root directory
cp .env.example .env

# Edit .env with your settings
```

#### **.env Template**
```
# Database
DATABASE_URL=postgresql://forensmart_user:secure_password@localhost:5432/forensmart

# Redis
REDIS_URL=redis://localhost:6379

# API
API_PORT=8000
API_HOST=0.0.0.0

# Frontend
FRONTEND_URL=http://localhost:3000

# Security
SECRET_KEY=your-secret-key-here
JWT_SECRET=your-jwt-secret-here

# AI/LLM
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key

# AWS (if using S3)
AWS_ACCESS_KEY_ID=your-aws-key
AWS_SECRET_ACCESS_KEY=your-aws-secret
AWS_S3_BUCKET=your-bucket-name
```

### **STEP 5: Database Migration**

```bash
# Activate virtual environment (if not already)
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

# Run migrations
alembic upgrade head

# Verify database
psql -U forensmart_user -d forensmart -c "\dt"
```

### **STEP 6: Start Services**

#### **Backend (Terminal 1)**
```bash
# Activate virtual environment
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

# Start API server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

#### **Frontend (Terminal 2)**
```bash
# Navigate to frontend
cd frontend

# Start development server
npm run dev
```

#### **Redis (Terminal 3)**
```bash
# Windows (WSL): redis-server
# macOS: redis-server
# Linux: redis-server
redis-server
```

---

## ✅ VERIFICATION

### **Backend Verification**

```bash
# Check Python version
python --version

# Check pip packages
pip list | grep -E "fastapi|sqlalchemy|redis"

# Test API
curl http://localhost:8000/api/health

# Expected response: {"status": "ok"}
```

### **Frontend Verification**

```bash
# Check Node version
node --version

# Check npm packages
npm list react react-router-dom zustand

# Frontend should be running at http://localhost:5173
```

### **Database Verification**

```bash
# Connect to database
psql -U forensmart_user -d forensmart

# List tables
\dt

# Check connections
SELECT datname, count(*) FROM pg_stat_activity GROUP BY datname;

# Exit
\q
```

### **Redis Verification**

```bash
# Connect to Redis
redis-cli

# Ping Redis
PING

# Expected response: PONG

# Exit
EXIT
```

---

## 🔧 TROUBLESHOOTING

### **Python Issues**

#### **"python: command not found"**
```bash
# Windows: Use python instead of python3
python --version

# macOS/Linux: Use python3
python3 --version

# Or add alias
alias python=python3
```

#### **"pip: command not found"**
```bash
# Windows
python -m pip --version

# macOS/Linux
python3 -m pip --version
```

#### **Virtual environment not activating**
```bash
# Windows: Make sure you're in the right directory
cd c:\Forensmart
venv\Scripts\activate

# macOS/Linux: Make sure you're in the right directory
cd ~/Forensmart
source venv/bin/activate
```

### **Dependency Issues**

#### **"ModuleNotFoundError"**
```bash
# Reinstall dependencies
pip install --upgrade --force-reinstall -r requirements.txt
```

#### **Version conflicts**
```bash
# Clear pip cache
pip cache purge

# Reinstall
pip install -r requirements.txt
```

### **Database Issues**

#### **"Connection refused"**
```bash
# Check if PostgreSQL is running
# Windows: Services > PostgreSQL
# macOS: brew services list
# Linux: sudo systemctl status postgresql

# Start PostgreSQL
# Windows: net start postgresql-x64-15
# macOS: brew services start postgresql
# Linux: sudo systemctl start postgresql
```

#### **"FATAL: role 'forensmart_user' does not exist"**
```bash
# Create user
psql -U postgres -c "CREATE USER forensmart_user WITH PASSWORD 'password';"
psql -U postgres -c "GRANT ALL PRIVILEGES ON DATABASE forensmart TO forensmart_user;"
```

### **Frontend Issues**

#### **"npm: command not found"**
```bash
# Install Node.js from https://nodejs.org/
# Or use package manager
# macOS: brew install node
# Linux: sudo apt install nodejs npm
```

#### **Port already in use**
```bash
# Frontend (change port)
npm run dev -- --port 3000

# Backend (change port)
uvicorn main:app --port 8001
```

---

## 📊 INSTALLATION CHECKLIST

### **System Setup**
- [ ] Python 3.11+ installed
- [ ] Node.js 18+ installed
- [ ] PostgreSQL 13+ installed
- [ ] Redis 6+ installed
- [ ] Git installed

### **Backend Setup**
- [ ] Repository cloned
- [ ] Virtual environment created
- [ ] Virtual environment activated
- [ ] requirements.txt installed
- [ ] .env file created
- [ ] Database configured
- [ ] Migrations run

### **Frontend Setup**
- [ ] Node modules installed
- [ ] .env configured
- [ ] Build verified

### **Services Running**
- [ ] PostgreSQL running
- [ ] Redis running
- [ ] Backend API running
- [ ] Frontend dev server running

### **Verification**
- [ ] API responding (http://localhost:8000/api/health)
- [ ] Frontend accessible (http://localhost:5173)
- [ ] Database connected
- [ ] Redis connected

---

## 🎯 QUICK START SUMMARY

### **Windows**
```bash
# 1. Create venv
python -m venv venv

# 2. Activate venv
venv\Scripts\activate

# 3. Install backend deps
pip install -r requirements.txt

# 4. Install frontend deps
cd frontend && npm install && cd ..

# 5. Create .env
copy .env.example .env

# 6. Start services
# Terminal 1: uvicorn main:app --reload
# Terminal 2: cd frontend && npm run dev
# Terminal 3: redis-server
```

### **macOS/Linux**
```bash
# 1. Create venv
python3 -m venv venv

# 2. Activate venv
source venv/bin/activate

# 3. Install backend deps
pip install -r requirements.txt

# 4. Install frontend deps
cd frontend && npm install && cd ..

# 5. Create .env
cp .env.example .env

# 6. Start services
# Terminal 1: uvicorn main:app --reload
# Terminal 2: cd frontend && npm run dev
# Terminal 3: redis-server
```

---

## 📞 SUPPORT

If you encounter issues:

1. Check this troubleshooting guide
2. Review error messages carefully
3. Check system requirements
4. Verify all services are running
5. Check .env configuration
6. Review logs for errors

---

**Status**: ✅ **INSTALLATION GUIDE COMPLETE**

**All dependencies documented and ready for installation**

