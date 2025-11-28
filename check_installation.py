#!/usr/bin/env python3
"""
FORENSMART - INSTALLATION VERIFICATION SCRIPT
Date: November 26, 2025
This script checks all installations in both local and venv
"""

import subprocess
import sys
import os
from pathlib import Path

# Color codes
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

# Counters
passed = 0
failed = 0
warnings = 0

def print_header(title):
    """Print section header"""
    print(f"\n{BLUE}{'='*50}")
    print(f"{title}")
    print(f"{'='*50}{RESET}\n")

def check_command(cmd, name):
    """Check if a command exists and works"""
    global passed, failed, warnings
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            output = result.stdout.strip().split('\n')[0]
            print(f"{GREEN}✓ PASSED{RESET} - {name}: {output}")
            passed += 1
            return True
        else:
            print(f"{RED}✗ FAILED{RESET} - {name}: {result.stderr.strip()}")
            failed += 1
            return False
    except Exception as e:
        print(f"{YELLOW}⚠ WARNING{RESET} - {name}: {str(e)}")
        warnings += 1
        return False

def check_package(package_name, package_type="pip"):
    """Check if a package is installed"""
    global passed, failed, warnings
    try:
        if package_type == "pip":
            result = subprocess.run(f"pip show {package_name}", shell=True, capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                version = [line.split(": ")[1] for line in result.stdout.split('\n') if line.startswith("Version")][0]
                print(f"{GREEN}✓ PASSED{RESET} - {package_name}: v{version}")
                passed += 1
                return True
            else:
                print(f"{RED}✗ FAILED{RESET} - {package_name}: Not installed")
                failed += 1
                return False
        elif package_type == "npm":
            result = subprocess.run(f"npm list {package_name}", shell=True, capture_output=True, text=True, timeout=5, cwd="frontend")
            if result.returncode == 0 or "deduped" in result.stdout:
                print(f"{GREEN}✓ PASSED{RESET} - {package_name}: Installed")
                passed += 1
                return True
            else:
                print(f"{YELLOW}⚠ WARNING{RESET} - {package_name}: May not be installed")
                warnings += 1
                return False
    except Exception as e:
        print(f"{YELLOW}⚠ WARNING{RESET} - {package_name}: {str(e)}")
        warnings += 1
        return False

def check_file(filepath, name):
    """Check if a file exists"""
    global passed, failed, warnings
    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        print(f"{GREEN}✓ PASSED{RESET} - {name}: Exists ({size} bytes)")
        passed += 1
        return True
    else:
        print(f"{RED}✗ FAILED{RESET} - {name}: Not found")
        failed += 1
        return False

def check_directory(dirpath, name):
    """Check if a directory exists"""
    global passed, failed, warnings
    if os.path.isdir(dirpath):
        item_count = len(os.listdir(dirpath))
        print(f"{GREEN}✓ PASSED{RESET} - {name}: Exists ({item_count} items)")
        passed += 1
        return True
    else:
        print(f"{RED}✗ FAILED{RESET} - {name}: Not found")
        failed += 1
        return False

def main():
    """Main verification function"""
    global passed, failed, warnings
    
    print(f"\n{BLUE}{'='*50}")
    print("FORENSMART - INSTALLATION VERIFICATION")
    print(f"{'='*50}{RESET}")
    
    # ========================================
    # SECTION 1: SYSTEM TOOLS
    # ========================================
    print_header("SECTION 1: SYSTEM TOOLS")
    
    check_command("python --version", "Python")
    check_command("pip --version", "pip")
    check_command("node --version", "Node.js")
    check_command("npm --version", "npm")
    check_command("git --version", "Git")
    
    # ========================================
    # SECTION 2: VIRTUAL ENVIRONMENT
    # ========================================
    print_header("SECTION 2: VIRTUAL ENVIRONMENT")
    
    check_directory("venv", "Virtual Environment (venv)")
    
    if os.path.isdir("venv"):
        if sys.platform == "win32":
            check_file("venv\\Scripts\\activate.bat", "venv activation script (Windows)")
        else:
            check_file("venv/bin/activate", "venv activation script (Unix)")
    
    # ========================================
    # SECTION 3: BACKEND DEPENDENCIES
    # ========================================
    print_header("SECTION 3: BACKEND DEPENDENCIES")
    
    backend_packages = [
        "fastapi", "uvicorn", "sqlalchemy", "psycopg2-binary", "redis",
        "pandas", "numpy", "scipy", "scikit-learn", "tensorflow",
        "transformers", "openai", "anthropic", "cryptography", "pyjwt",
        "bcrypt", "pytest", "requests", "aiofiles", "python-dotenv",
        "pydantic"
    ]
    
    print(f"Checking {len(backend_packages)} backend packages...")
    for package in backend_packages:
        check_package(package, "pip")
    
    # ========================================
    # SECTION 4: FRONTEND DEPENDENCIES
    # ========================================
    print_header("SECTION 4: FRONTEND DEPENDENCIES")
    
    check_directory("frontend", "Frontend directory")
    check_directory("frontend/node_modules", "Frontend node_modules")
    
    if os.path.isdir("frontend/node_modules"):
        frontend_packages = [
            "react", "react-dom", "react-router-dom", "typescript",
            "tailwindcss", "zustand", "axios", "react-hook-form",
            "zod", "recharts", "lucide-react"
        ]
        
        print(f"Checking {len(frontend_packages)} frontend packages...")
        for package in frontend_packages:
            check_package(package, "npm")
    
    # ========================================
    # SECTION 5: DATABASES
    # ========================================
    print_header("SECTION 5: DATABASES")
    
    check_command("psql --version", "PostgreSQL")
    check_command("redis-cli --version", "Redis")
    
    # ========================================
    # SECTION 6: CONFIGURATION FILES
    # ========================================
    print_header("SECTION 6: CONFIGURATION FILES")
    
    check_file(".env", ".env file")
    check_file("requirements.txt", "requirements.txt")
    check_file("frontend/package.json", "frontend/package.json")
    check_file("frontend/tsconfig.json", "frontend/tsconfig.json")
    
    # ========================================
    # SUMMARY
    # ========================================
    print_header("VERIFICATION SUMMARY")
    
    print(f"{GREEN}Passed:  {passed}{RESET}")
    print(f"{RED}Failed:  {failed}{RESET}")
    print(f"{YELLOW}Warnings: {warnings}{RESET}")
    print(f"Total:   {passed + failed + warnings}")
    
    print()
    
    if failed == 0:
        print(f"{GREEN}{'='*50}")
        print("✓ SUCCESS - All critical checks passed!")
        print(f"{'='*50}{RESET}")
        print("\nReady to start services:")
        print("  Terminal 1: uvicorn main:app --reload")
        print("  Terminal 2: cd frontend && npm run dev")
        print("  Terminal 3: redis-server")
        return 0
    else:
        print(f"{RED}{'='*50}")
        print(f"✗ ERROR - {failed} critical check(s) failed!")
        print(f"{'='*50}{RESET}")
        print("\nPlease fix the issues above before proceeding.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
