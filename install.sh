#!/bin/bash

# FORENSMART - COMPLETE INSTALLATION SCRIPT FOR MACOS/LINUX
# Date: November 26, 2025
# This script installs all dependencies for Forensmart

echo ""
echo "========================================"
echo "FORENSMART - COMPLETE INSTALLATION"
echo "========================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.11+ from https://www.python.org/downloads/"
    exit 1
fi

echo "[1/8] Python version check... OK"
python3 --version

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "WARNING: Node.js is not installed"
    echo "Please install Node.js 18+ from https://nodejs.org/"
    echo "Continuing with backend setup only..."
else
    echo "[2/8] Node.js version check... OK"
    node --version
fi

# Create virtual environment
echo ""
echo "[3/8] Creating Python virtual environment..."
if [ -d "venv" ]; then
    echo "Virtual environment already exists"
else
    python3 -m venv venv
    echo "Virtual environment created"
fi

# Activate virtual environment
echo "[4/8] Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "[5/8] Upgrading pip..."
python3 -m pip install --upgrade pip

# Install backend dependencies
echo "[6/8] Installing backend dependencies..."
echo "This may take several minutes..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install backend dependencies"
    exit 1
fi

# Install frontend dependencies
echo "[7/8] Installing frontend dependencies..."
if [ -d "frontend" ]; then
    cd frontend
    npm install
    if [ $? -ne 0 ]; then
        echo "WARNING: Failed to install frontend dependencies"
        cd ..
    else
        echo "Frontend dependencies installed successfully"
        cd ..
    fi
else
    echo "WARNING: Frontend directory not found"
fi

# Create .env file if it doesn't exist
echo "[8/8] Setting up environment configuration..."
if [ -f ".env" ]; then
    echo ".env file already exists"
else
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo ".env file created from template"
    else
        echo "Creating default .env file..."
        cat > .env << 'EOF'
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
EOF
        echo ".env file created"
    fi
fi

echo ""
echo "========================================"
echo "INSTALLATION COMPLETE!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Make sure PostgreSQL is running"
echo "2. Make sure Redis is running"
echo "3. Open 3 terminals and run:"
echo "   Terminal 1: source venv/bin/activate && uvicorn main:app --reload"
echo "   Terminal 2: cd frontend && npm run dev"
echo "   Terminal 3: redis-server"
echo ""
echo "Frontend: http://localhost:5173"
echo "Backend API: http://localhost:8000"
echo ""
