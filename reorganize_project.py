#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ForenSmart Project Reorganization Script
Reorganizes the entire project structure for better organization
"""

import os
import shutil
import sys
from pathlib import Path

# Fix encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Project root
PROJECT_ROOT = Path(__file__).parent

def create_directories():
    """Create new directory structure"""
    print("📁 Creating new directory structure...")
    
    dirs = [
        "modules/approval",
        "modules/consent",
        "modules/extraction",
        "modules/analysis",
        "modules/storage",
        "modules/ui",
        "modules/automation",
        "modules/reporting",
        "modules/shared",
        "modules/adapters",
        "data/artifacts",
        "data/audit",
        "data/consent_records",
        "data/case_snapshots",
        "data/reports",
        "docs",
        "scripts",
        "tests",
        ".backups",
    ]
    
    for dir_path in dirs:
        full_path = PROJECT_ROOT / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"  ✅ Created: {dir_path}")

def move_approval_files():
    """Move approval-related files"""
    print("\n📦 Moving approval files...")
    
    files = [
        ("approval_manager.py", "modules/approval/manager.py"),
        ("approval_sync.py", "modules/approval/sync.py"),
        ("approval_utils.py", "modules/approval/utils.py"),
        ("approval_redirect.py", "modules/approval/redirect.py"),
        ("approval_auto_extraction.py", "modules/approval/auto_extraction.py"),
    ]
    
    for src, dst in files:
        src_path = PROJECT_ROOT / "modules" / src
        dst_path = PROJECT_ROOT / dst
        if src_path.exists():
            shutil.move(str(src_path), str(dst_path))
            print(f"  ✅ Moved: {src} → {dst}")

def move_consent_files():
    """Move consent-related files"""
    print("\n📦 Moving consent files...")
    
    files = [
        ("consent.py", "modules/consent/models.py"),
        ("consent_manager.py", "modules/consent/manager.py"),
        ("consent_portal.py", "modules/consent/portal.py"),
        ("consent_portal_enhanced.py", "modules/consent/enhanced.py"),
    ]
    
    for src, dst in files:
        src_path = PROJECT_ROOT / "modules" / src
        dst_path = PROJECT_ROOT / dst
        if src_path.exists():
            shutil.move(str(src_path), str(dst_path))
            print(f"  ✅ Moved: {src} → {dst}")

def move_extraction_files():
    """Move extraction-related files"""
    print("\n📦 Moving extraction files...")
    
    files = [
        ("data_extraction_orchestrator.py", "modules/extraction/orchestrator.py"),
        ("extraction_validator.py", "modules/extraction/validator.py"),
        ("extraction_progress.py", "modules/extraction/progress.py"),
        ("extraction_ui.py", "modules/extraction/ui.py"),
    ]
    
    for src, dst in files:
        src_path = PROJECT_ROOT / "modules" / src
        dst_path = PROJECT_ROOT / dst
        if src_path.exists():
            shutil.move(str(src_path), str(dst_path))
            print(f"  ✅ Moved: {src} → {dst}")

def move_analysis_files():
    """Move analysis-related files"""
    print("\n📦 Moving analysis files...")
    
    files = [
        ("comms_analyzer.py", "modules/analysis/comms_analyzer.py"),
        ("location_intelligence.py", "modules/analysis/location_intelligence.py"),
        ("suspicious_classifier.py", "modules/analysis/suspicious_classifier.py"),
    ]
    
    for src, dst in files:
        src_path = PROJECT_ROOT / "modules" / src
        dst_path = PROJECT_ROOT / dst
        if src_path.exists():
            shutil.move(str(src_path), str(dst_path))
            print(f"  ✅ Moved: {src} → {dst}")

def move_storage_files():
    """Move storage-related files"""
    print("\n📦 Moving storage files...")
    
    files = [
        ("storage_manager.py", "modules/storage/manager.py"),
        ("storage_ui.py", "modules/storage/ui.py"),
    ]
    
    for src, dst in files:
        src_path = PROJECT_ROOT / "modules" / src
        dst_path = PROJECT_ROOT / dst
        if src_path.exists():
            shutil.move(str(src_path), str(dst_path))
            print(f"  ✅ Moved: {src} → {dst}")

def move_ui_files():
    """Move UI-related files"""
    print("\n📦 Moving UI files...")
    
    files = [
        ("progress_ui.py", "modules/ui/progress_ui.py"),
        ("media_viewer.py", "modules/ui/media_viewer.py"),
        ("suspicious_comms_ui.py", "modules/ui/suspicious_comms_ui.py"),
    ]
    
    for src, dst in files:
        src_path = PROJECT_ROOT / "modules" / src
        dst_path = PROJECT_ROOT / dst
        if src_path.exists():
            shutil.move(str(src_path), str(dst_path))
            print(f"  ✅ Moved: {src} → {dst}")

def move_shared_files():
    """Move shared utility files"""
    print("\n📦 Moving shared files...")
    
    files = [
        ("shared_utils.py", "modules/shared/utils.py"),
        ("error_checker.py", "modules/shared/error_checker.py"),
        ("device_manager.py", "modules/shared/device_manager.py"),
        ("device_detector.py", "modules/shared/device_detector.py"),
        ("file_handler.py", "modules/shared/file_handler.py"),
        ("unified_error_system.py", "modules/shared/unified_error_system.py"),
        ("app_error_checker.py", "modules/shared/app_error_checker.py"),
    ]
    
    for src, dst in files:
        src_path = PROJECT_ROOT / "modules" / src
        dst_path = PROJECT_ROOT / dst
        if src_path.exists():
            shutil.move(str(src_path), str(dst_path))
            print(f"  ✅ Moved: {src} → {dst}")

def move_adapters():
    """Move adapters folder"""
    print("\n📦 Moving adapters...")
    
    src_path = PROJECT_ROOT / "modules" / "adapters"
    dst_path = PROJECT_ROOT / "modules" / "adapters"
    
    if src_path.exists() and src_path != dst_path:
        shutil.move(str(src_path), str(dst_path))
        print(f"  ✅ Adapters already in correct location")

def move_data_folders():
    """Move data folders"""
    print("\n📦 Moving data folders...")
    
    folders = [
        ("artifacts", "data/artifacts"),
        ("audit", "data/audit"),
        ("consent_records", "data/consent_records"),
        ("case_snapshots", "data/case_snapshots"),
        ("reports", "data/reports"),
    ]
    
    for src, dst in folders:
        src_path = PROJECT_ROOT / src
        dst_path = PROJECT_ROOT / dst
        
        if src_path.exists() and src_path != dst_path:
            # Move contents if destination already has data
            if dst_path.exists():
                for item in src_path.iterdir():
                    dst_item = dst_path / item.name
                    if dst_item.exists():
                        shutil.rmtree(str(dst_item))
                    shutil.move(str(item), str(dst_item))
                shutil.rmtree(str(src_path))
            else:
                shutil.move(str(src_path), str(dst_path))
            print(f"  ✅ Moved: {src} → {dst}")

def move_documentation():
    """Move documentation files"""
    print("\n📦 Moving documentation files...")
    
    # Move all .md and .txt files to docs/
    for file in PROJECT_ROOT.glob("*.md"):
        if file.name != "README.md":
            dst = PROJECT_ROOT / "docs" / file.name
            shutil.move(str(file), str(dst))
            print(f"  ✅ Moved: {file.name} → docs/")
    
    for file in PROJECT_ROOT.glob("*.txt"):
        dst = PROJECT_ROOT / "docs" / file.name
        shutil.move(str(file), str(dst))
        print(f"  ✅ Moved: {file.name} → docs/")

def move_scripts():
    """Move utility scripts"""
    print("\n📦 Moving scripts...")
    
    scripts = [
        "fix_forensmart_lint.py",
    ]
    
    for script in scripts:
        src_path = PROJECT_ROOT / script
        dst_path = PROJECT_ROOT / "scripts" / script
        if src_path.exists():
            shutil.move(str(src_path), str(dst_path))
            print(f"  ✅ Moved: {script} → scripts/")

def backup_old_files():
    """Backup old app files"""
    print("\n📦 Backing up old files...")
    
    old_files = [
        "app_patched_fixed_final.py",
        "app_patched_fixed_noconsent.py",
        "app_patched.py.backup.20250-02T162912",
        "app_patched.py.backup.20250-02T162920",
    ]
    
    for file in old_files:
        src_path = PROJECT_ROOT / file
        if src_path.exists():
            dst_path = PROJECT_ROOT / ".backups" / file
            shutil.move(str(src_path), str(dst_path))
            print(f"  ✅ Backed up: {file}")

def remove_old_folders():
    """Remove old/duplicate folders"""
    print("\n📦 Removing old folders...")
    
    folders = [
        "ui_components",
        "src",
    ]
    
    for folder in folders:
        folder_path = PROJECT_ROOT / folder
        if folder_path.exists():
            shutil.rmtree(str(folder_path))
            print(f"  ✅ Removed: {folder}/")

def create_init_files():
    """Create __init__.py files"""
    print("\n📦 Creating __init__.py files...")
    
    init_dirs = [
        "modules/approval",
        "modules/consent",
        "modules/extraction",
        "modules/analysis",
        "modules/storage",
        "modules/ui",
        "modules/automation",
        "modules/reporting",
        "modules/shared",
        "modules/adapters",
    ]
    
    for dir_path in init_dirs:
        init_file = PROJECT_ROOT / dir_path / "__init__.py"
        init_file.touch()
        print(f"  ✅ Created: {dir_path}/__init__.py")

def create_setup_files():
    """Create setup.py and pyproject.toml"""
    print("\n📦 Creating setup files...")
    
    # setup.py
    setup_py = PROJECT_ROOT / "setup.py"
    if not setup_py.exists():
        setup_py.write_text('''from setuptools import setup, find_packages

setup(
    name="forensmart",
    version="2.0.0",
    description="Professional Digital Forensic Analysis Platform",
    author="ForenSmart Team",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "streamlit>=1.28.0",
        "pandas>=1.5.0",
        "numpy>=1.24.0",
        "plotly>=5.0.0",
        "openai>=1.0.0",
        "schedule>=1.2.0",
        "requests>=2.31.0",
    ],
)
''')
        print("  ✅ Created: setup.py")
    
    # pyproject.toml
    pyproject = PROJECT_ROOT / "pyproject.toml"
    if not pyproject.exists():
        pyproject.write_text('''[build-system]
requires = ["setuptools>=65.0"]
build-backend = "setuptools.build_meta"

[project]
name = "forensmart"
version = "2.0.0"
description = "Professional Digital Forensic Analysis Platform"
requires-python = ">=3.9"

[tool.black]
line-length = 100

[tool.ruff]
line-length = 100
select = ["E", "F", "W"]
''')
        print("  ✅ Created: pyproject.toml")

def update_gitignore():
    """Update .gitignore"""
    print("\n📦 Updating .gitignore...")
    
    gitignore_path = PROJECT_ROOT / ".gitignore"
    gitignore_content = '''# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
.venv/
env/

# IDE
.vscode/
.idea/
*.swp
*.swo
*.sublime-project
*.sublime-workspace

# Data (runtime)
data/
artifacts/
audit/
consent_records/
case_snapshots/
reports/

# Backups
.backups/
*.bak
*.backup
*.tmp

# Logs
*.log
logs/

# Environment
.env
.env.local
.env.*.local

# OS
.DS_Store
Thumbs.db
.directory

# Testing
.pytest_cache/
.coverage
htmlcov/
.tox/

# Build
build/
dist/
*.egg-info/
.eggs/

# Cache
.ruff_cache/
.mypy_cache/
.dmypy.json
dmypy.json

# IDE specific
.vscode/settings.json
.vscode/launch.json
'''
    
    gitignore_path.write_text(gitignore_content)
    print("  ✅ Updated: .gitignore")

def print_summary():
    """Print reorganization summary"""
    print("\n" + "="*60)
    print("✅ PROJECT REORGANIZATION COMPLETE!")
    print("="*60)
    print("""
New Structure:
├── app.py (main entry point)
├── modules/
│   ├── approval/ (approval system)
│   ├── consent/ (consent management)
│   ├── extraction/ (data extraction)
│   ├── analysis/ (data analysis)
│   ├── storage/ (storage management)
│   ├── ui/ (UI components)
│   ├── automation/ (automation features)
│   ├── reporting/ (report generation)
│   ├── shared/ (shared utilities)
│   └── adapters/ (device adapters)
├── pages/ (Streamlit pages)
├── data/ (runtime data)
├── docs/ (documentation)
├── scripts/ (utility scripts)
├── tests/ (unit tests)
└── .backups/ (old files)

Next Steps:
1. Update imports in all files
2. Test the application
3. Commit to git
4. Deploy to production

Run: streamlit run app.py
""")

def main():
    """Main reorganization function"""
    print("🚀 Starting ForenSmart Project Reorganization...")
    print("="*60)
    
    try:
        create_directories()
        move_approval_files()
        move_consent_files()
        move_extraction_files()
        move_analysis_files()
        move_storage_files()
        move_ui_files()
        move_shared_files()
        move_adapters()
        move_data_folders()
        move_documentation()
        move_scripts()
        backup_old_files()
        remove_old_folders()
        create_init_files()
        create_setup_files()
        update_gitignore()
        print_summary()
        
        print("\n✅ Reorganization completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during reorganization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
