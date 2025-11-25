#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fix imports after reorganization
"""

import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

# Import mapping
IMPORT_MAPPING = {
    # Approval imports
    "from modules.approval_manager import": "from modules.approval.manager import",
    "from modules.approval_sync import": "from modules.approval.sync import",
    "from modules.approval_utils import": "from modules.approval.utils import",
    "from modules.approval_redirect import": "from modules.approval.redirect import",
    "from modules.approval_auto_extraction import": "from modules.approval.auto_extraction import",
    
    # Consent imports
    "from modules.consent import": "from modules.consent.models import",
    "from modules.consent_manager import": "from modules.consent.manager import",
    "from modules.consent_portal import": "from modules.consent.portal import",
    "from modules.consent_portal_enhanced import": "from modules.consent.enhanced import",
    
    # Extraction imports
    "from modules.data_extraction_orchestrator import": "from modules.extraction.orchestrator import",
    "from modules.extraction_validator import": "from modules.extraction.validator import",
    "from modules.extraction_progress import": "from modules.extraction.progress import",
    "from modules.extraction_ui import": "from modules.extraction.ui import",
    
    # Analysis imports
    "from modules.comms_analyzer import": "from modules.analysis.comms_analyzer import",
    "from modules.location_intelligence import": "from modules.analysis.location_intelligence import",
    "from modules.suspicious_classifier import": "from modules.analysis.suspicious_classifier import",
    
    # Storage imports
    "from modules.storage_manager import": "from modules.storage.manager import",
    "from modules.storage_ui import": "from modules.storage.ui import",
    
    # UI imports
    "from modules.progress_ui import": "from modules.ui.progress_ui import",
    "from modules.media_viewer import": "from modules.ui.media_viewer import",
    "from modules.suspicious_comms_ui import": "from modules.ui.suspicious_comms_ui import",
    
    # Shared imports
    "from modules.shared_utils import": "from modules.shared.utils import",
    "from modules.error_checker import": "from modules.shared.error_checker import",
    "from modules.device_manager import": "from modules.shared.device_manager import",
    "from modules.device_detector import": "from modules.shared.device_detector import",
    "from modules.file_handler import": "from modules.shared.file_handler import",
    "from modules.unified_error_system import": "from modules.shared.unified_error_system import",
    "from modules.app_error_checker import": "from modules.shared.app_error_checker import",
}

def fix_imports_in_file(file_path):
    """Fix imports in a single file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Apply all import mappings
        for old_import, new_import in IMPORT_MAPPING.items():
            content = content.replace(old_import, new_import)
        
        # If content changed, write it back
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        return False
    except Exception as e:
        print(f"  Error fixing {file_path}: {e}")
        return False

def main():
    """Fix imports in all Python files"""
    print("Fixing imports in all Python files...")
    print("="*60)
    
    fixed_count = 0
    
    # Fix imports in modules
    for py_file in PROJECT_ROOT.rglob("*.py"):
        # Skip reorganize script and this script
        if py_file.name in ["reorganize_project.py", "fix_imports.py"]:
            continue
        
        # Skip __pycache__
        if "__pycache__" in str(py_file):
            continue
        
        if fix_imports_in_file(py_file):
            print(f"  Fixed: {py_file.relative_to(PROJECT_ROOT)}")
            fixed_count += 1
    
    print("="*60)
    print(f"Fixed imports in {fixed_count} files")
    print("\nNext steps:")
    print("1. Test the application: streamlit run app.py")
    print("2. Check for any remaining import errors")
    print("3. Commit to git")

if __name__ == "__main__":
    main()
