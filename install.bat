@echo off
REM FORENSMART - COMPLETE INSTALLATION SCRIPT FOR WINDOWS
REM Date: November 26, 2025
REM This script installs all dependencies for Forensmart

echo.
echo ========================================
echo FORENSMART - COMPLETE INSTALLATION
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.11+ from https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [1/8] Python version check... OK
python --version

REM Check if Node.js is installed
node --version >nul 2>&1
if errorlevel 1 (
    echo WARNING: Node.js is not installed
    echo Please install Node.js 18+ from https://nodejs.org/
    echo Continuing with backend setup only...
) else (
    echo [2/8] Node.js version check... OK
    node --version
)

REM Create virtual environment
echo.
echo [3/8] Creating Python virtual environment...
if exist venv (
    echo Virtual environment already exists
) else (
    python -m venv venv
    echo Virtual environment created
)

REM Activate virtual environment
echo [4/8] Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo [5/8] Upgrading pip...
python -m pip install --upgrade pip

REM Install backend dependencies
echo [6/8] Installing backend dependencies...
echo This may take several minutes...
pip install -r requirements.txt

if errorlevel 1 (
    echo ERROR: Failed to install backend dependencies
    pause
    exit /b 1
)

REM Install frontend dependencies
echo [7/8] Installing frontend dependencies...
if exist frontend (
    cd frontend
    call npm install
    if errorlevel 1 (
        echo WARNING: Failed to install frontend dependencies
        cd ..
    ) else (
        echo Frontend dependencies installed successfully
        cd ..
    )
) else (
    echo WARNING: Frontend directory not found
)

REM Create .env file if it doesn't exist
echo [8/8] Setting up environment configuration...
if exist .env (
    echo .env file already exists
) else (
    if exist .env.example (
        copy .env.example .env
        echo .env file created from template
    ) else (
        echo Creating default .env file...
        (
            echo # Database
            echo DATABASE_URL=postgresql://forensmart_user:secure_password@localhost:5432/forensmart
            echo.
            echo # Redis
            echo REDIS_URL=redis://localhost:6379
            echo.
            echo # API
            echo API_PORT=8000
            echo API_HOST=0.0.0.0
            echo.
            echo # Frontend
            echo FRONTEND_URL=http://localhost:3000
            echo.
            echo # Security
            echo SECRET_KEY=your-secret-key-here
            echo JWT_SECRET=your-jwt-secret-here
        ) > .env
        echo .env file created
    )
)

echo.
echo ========================================
echo INSTALLATION COMPLETE!
echo ========================================
echo.
echo Next steps:
echo 1. Make sure PostgreSQL is running
echo 2. Make sure Redis is running
echo 3. Open 3 terminals and run:
echo    Terminal 1: venv\Scripts\activate ^& uvicorn main:app --reload
echo    Terminal 2: cd frontend ^& npm run dev
echo    Terminal 3: redis-server
echo.
echo Frontend: http://localhost:5173
echo Backend API: http://localhost:8000
echo.
pause
