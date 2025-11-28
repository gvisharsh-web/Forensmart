# 🚀 FORENSMART - INSTALL REMAINING DEPENDENCIES

**Date**: November 26, 2025
**Status**: ✅ INSTALLATION COMMANDS COMPLETE
**Scope**: All CMD commands to install missing dependencies

---

## 📋 TABLE OF CONTENTS

1. [Prerequisites](#prerequisites)
2. [Backend Dependencies](#backend-dependencies)
3. [Frontend Dependencies](#frontend-dependencies)
4. [Database Setup](#database-setup)
5. [Complete Installation](#complete-installation)

---

## ✅ PREREQUISITES

### **Step 1: Activate Virtual Environment**

#### **Windows**
```bash
venv\Scripts\activate
```

#### **macOS/Linux**
```bash
source venv/bin/activate
```

**Expected Output**: `(venv)` prefix in your terminal

---

## 📦 BACKEND DEPENDENCIES

### **Option 1: Install All at Once**

```bash
pip install -r requirements.txt
```

### **Option 2: Install Individual Packages**

#### **Web Framework & API**
```bash
pip install fastapi==0.104.0
pip install uvicorn==0.24.0
pip install streamlit==1.28.0
pip install streamlit-extras==0.3.0
```

#### **Data Processing**
```bash
pip install pandas==1.5.0
pip install numpy==1.24.0
pip install scipy==1.11.0
```

#### **Database & Cache**
```bash
pip install sqlalchemy==2.0.0
pip install psycopg2-binary==2.9.0
pip install redis==5.0.0
```

#### **Machine Learning**
```bash
pip install scikit-learn==1.3.0
pip install tensorflow==2.13.0
pip install transformers==4.34.0
pip install joblib==1.3.0
```

#### **AI & LLM**
```bash
pip install openai==1.0.0
pip install anthropic==0.7.0
```

#### **Cloud & Storage**
```bash
pip install boto3==1.28.0
pip install botocore==1.31.0
```

#### **HTTP & Requests**
```bash
pip install requests==2.31.0
pip install httpx==0.25.0
pip install aiofiles==23.0.0
```

#### **Configuration & Environment**
```bash
pip install python-dotenv==1.0.0
pip install pydantic==2.5.0
pip install pyyaml==6.0.0
```

#### **Security & Encryption**
```bash
pip install cryptography==41.0.0
pip install pyjwt==2.8.0
pip install bcrypt==4.1.0
```

#### **Utilities**
```bash
pip install python-dateutil==2.8.0
pip install pytz==2023.3
pip install uuid6==1.0.0
```

#### **Scheduling & Automation**
```bash
pip install schedule==1.2.0
pip install APScheduler==3.10.0
```

#### **Logging & Monitoring**
```bash
pip install loguru==0.7.0
pip install sentry-sdk==1.38.0
```

#### **Testing**
```bash
pip install pytest==7.4.0
pip install pytest-asyncio==0.21.0
pip install pytest-cov==4.1.0
```

#### **Code Quality**
```bash
pip install pylint==3.0.0
pip install flake8==6.1.0
pip install black==23.12.0
pip install isort==5.13.0
```

#### **Development**
```bash
pip install ipython==8.18.0
pip install jupyter==1.0.0
pip install ipdb==0.13.0
```

#### **Visualization**
```bash
pip install plotly==5.0.0
pip install pyvis==0.3.0
pip install matplotlib==3.8.0
pip install folium==0.14.0
pip install streamlit-folium==0.15.0
pip install googlemaps==4.10.0
```

#### **PDF & Reporting**
```bash
pip install reportlab==4.0.0
pip install pypdf==3.17.0
```

---

## 🎨 FRONTEND DEPENDENCIES

### **Step 1: Navigate to Frontend Directory**

```bash
cd frontend
```

### **Option 1: Install All at Once**

```bash
npm install
```

### **Option 2: Install Individual Packages**

#### **Core React**
```bash
npm install react@18.2.0
npm install react-dom@18.2.0
npm install react-router-dom@6.20.0
```

#### **TypeScript**
```bash
npm install --save-dev typescript@5.3.0
```

#### **State Management**
```bash
npm install zustand@4.4.0
```

#### **Forms & Validation**
```bash
npm install react-hook-form@7.48.0
npm install @hookform/resolvers@3.3.0
npm install zod@3.22.0
```

#### **HTTP Client**
```bash
npm install axios@1.6.0
```

#### **Styling**
```bash
npm install tailwindcss@3.4.0
npm install --save-dev postcss@8.4.0
npm install --save-dev autoprefixer@10.4.0
```

#### **UI Components**
```bash
npm install lucide-react@0.294.0
npm install @radix-ui/react-dialog@1.1.1
npm install @radix-ui/react-dropdown-menu@2.0.5
npm install @radix-ui/react-progress@1.0.3
npm install @radix-ui/react-tabs@1.0.4
```

#### **Charts**
```bash
npm install recharts@2.10.0
```

#### **Utilities**
```bash
npm install class-variance-authority@0.7.0
npm install clsx@2.0.0
npm install tailwind-merge@2.2.0
```

#### **Build Tool**
```bash
npm install --save-dev vite@5.0.0
npm install --save-dev @vitejs/plugin-react@4.2.0
```

#### **Linting & Testing**
```bash
npm install --save-dev eslint@8.0.0
npm install --save-dev @typescript-eslint/eslint-plugin@6.0.0
npm install --save-dev @typescript-eslint/parser@6.0.0
npm install --save-dev eslint-plugin-react-hooks@4.6.0
npm install --save-dev eslint-plugin-react-refresh@0.4.0
npm install --save-dev vitest@1.0.0
```

### **Step 2: Return to Root Directory**

```bash
cd ..
```

---

## 💾 DATABASE SETUP

### **PostgreSQL Installation**

#### **Windows**
1. Download from https://www.postgresql.org/download/windows/
2. Run installer
3. Set password for postgres user
4. Keep default port 5432

#### **macOS**
```bash
brew install postgresql@15
brew services start postgresql@15
```

#### **Linux (Ubuntu)**
```bash
sudo apt update
sudo apt install postgresql postgresql-contrib -y
sudo systemctl start postgresql
```

### **Create PostgreSQL Database**

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

#### **Windows (WSL)**
```bash
wsl
sudo apt install redis-server -y
redis-server
```

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

## 🚀 COMPLETE INSTALLATION SEQUENCE

### **Step 1: Create Virtual Environment**

#### **Windows**
```bash
python -m venv venv
venv\Scripts\activate
```

#### **macOS/Linux**
```bash
python3 -m venv venv
source venv/bin/activate
```

### **Step 2: Upgrade pip**

```bash
python -m pip install --upgrade pip
```

### **Step 3: Install Backend Dependencies**

```bash
pip install -r requirements.txt
```

### **Step 4: Install Frontend Dependencies**

```bash
cd frontend
npm install
cd ..
```

### **Step 5: Create .env File**

#### **Windows**
```bash
copy .env.example .env
```

#### **macOS/Linux**
```bash
cp .env.example .env
```

### **Step 6: Configure .env**

Edit `.env` with your settings:
```
DATABASE_URL=postgresql://forensmart_user:secure_password@localhost:5432/forensmart
REDIS_URL=redis://localhost:6379
API_PORT=8000
API_HOST=0.0.0.0
FRONTEND_URL=http://localhost:3000
SECRET_KEY=your-secret-key-here
JWT_SECRET=your-jwt-secret-here
```

### **Step 7: Verify Installation**

```bash
python check_installation.py
```

---

## 📋 QUICK COPY-PASTE COMMANDS

### **Windows - Complete Setup**

```bash
REM 1. Create and activate venv
python -m venv venv
venv\Scripts\activate

REM 2. Upgrade pip
python -m pip install --upgrade pip

REM 3. Install backend dependencies
pip install -r requirements.txt

REM 4. Install frontend dependencies
cd frontend
npm install
cd ..

REM 5. Create .env
copy .env.example .env

REM 6. Verify
python check_installation.py
```

### **macOS/Linux - Complete Setup**

```bash
# 1. Create and activate venv
python3 -m venv venv
source venv/bin/activate

# 2. Upgrade pip
python3 -m pip install --upgrade pip

# 3. Install backend dependencies
pip install -r requirements.txt

# 4. Install frontend dependencies
cd frontend
npm install
cd ..

# 5. Create .env
cp .env.example .env

# 6. Verify
python check_installation.py
```

---

## 🔧 TROUBLESHOOTING

### **If pip install fails**

```bash
# Clear pip cache
pip cache purge

# Try again
pip install -r requirements.txt
```

### **If npm install fails**

```bash
# Clear npm cache
npm cache clean --force

# Try again
cd frontend
npm install
cd ..
```

### **If specific package fails**

```bash
# Upgrade setuptools
pip install --upgrade setuptools

# Try installing the package again
pip install package_name
```

### **If virtual environment issues**

```bash
# Delete old venv
rmdir /s venv  # Windows
rm -rf venv    # macOS/Linux

# Create new venv
python -m venv venv

# Activate and reinstall
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

---

## ✅ VERIFICATION CHECKLIST

After running all commands:

- [ ] Virtual environment created and activated
- [ ] pip upgraded
- [ ] Backend dependencies installed (88 packages)
- [ ] Frontend dependencies installed (20+ packages)
- [ ] .env file created
- [ ] PostgreSQL installed and running
- [ ] Redis installed and running
- [ ] All verification checks passed

---

## 🎯 NEXT STEPS

After installation:

### **Terminal 1: Start Backend API**
```bash
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux
uvicorn main:app --reload
```

### **Terminal 2: Start Frontend**
```bash
cd frontend
npm run dev
```

### **Terminal 3: Start Redis**
```bash
redis-server
```

---

## 📊 INSTALLATION SUMMARY

| Component | Packages | Command | Time |
|-----------|----------|---------|------|
| Backend | 88 | `pip install -r requirements.txt` | 5-10 min |
| Frontend | 20+ | `npm install` | 2-5 min |
| Total | 108+ | See above | 8-16 min |

---

**Status**: ✅ **ALL INSTALLATION COMMANDS PROVIDED**

**Ready to**: Install remaining dependencies

**Time to complete**: 8-16 minutes

