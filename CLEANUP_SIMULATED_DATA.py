#!/usr/bin/env python3
"""
CLEANUP SIMULATED DATA SCRIPT
Removes all simulated test results and mock data from ForenSmart modules
Creates a detailed report of what was cleaned

Usage:
    python CLEANUP_SIMULATED_DATA.py
"""

import os
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# ============================================================================
# CONFIGURATION
# ============================================================================

FORENSMART_ROOT = r"c:\Forensmart"
MODULES_PATH = os.path.join(FORENSMART_ROOT, "modules")
REPORT_FILE = os.path.join(FORENSMART_ROOT, "CLEANUP_REPORT.txt")

# Files to scan for simulated data
FILES_TO_SCAN = [
    "modules/consent/ui.py",
    "modules/consent/models.py",
    "modules/extraction/ui_device_selector.py",
    "modules/extraction/adapters/adb_adapter.py",
    "modules/extraction/ui_extraction_results.py",
    "modules/extraction/ui.py",
    "modules/extraction/ui_consent_check.py",
    "modules/extraction/ui_extraction_progress.py",
    "modules/extraction/ui_consent_approval.py",
    "modules/extraction/adapters/email_adapter.py",
    "modules/extraction/adapters/google_drive_adapter.py",
    "modules/extraction/adapters/hdd_adapter.py",
    "modules/extraction/adapters/instagram_adapter.py",
    "modules/extraction/adapters/ios_adapter.py",
    "modules/extraction/adapters/onedrive_adapter.py",
    "modules/extraction/adapters/snapchat_adapter.py",
    "modules/analysis/media_viewer.py",
]

# Patterns to identify simulated data
SIMULATED_PATTERNS = [
    r"# Simulated.*?\n.*?\n",  # Simulated comments
    r"devices = \[.*?\]",  # Device lists
    r"'id': 'device_\d+'",  # Device IDs
    r"'name': '[^']*'",  # Device names
    r"'status': '(connected|offline)'",  # Status
    r"mock_.*?=.*?\n",  # Mock variables
    r"test_data.*?=.*?\n",  # Test data
    r"sample_.*?=.*?\n",  # Sample data
    r"SIMULATED_.*?=.*?\n",  # Simulated constants
]

# ============================================================================
# CLEANUP FUNCTIONS
# ============================================================================

class SimulatedDataCleaner:
    """Clean simulated data from ForenSmart modules"""
    
    def __init__(self):
        self.report = []
        self.files_cleaned = 0
        self.lines_removed = 0
        self.patterns_found = {}
        
    def log(self, message: str):
        """Log message to report"""
        try:
            print(message)
        except UnicodeEncodeError:
            print(message.encode('utf-8', errors='replace').decode('utf-8'))
        self.report.append(message)
    
    def scan_file(self, file_path: str) -> Tuple[int, List[str]]:
        """Scan file for simulated data patterns"""
        full_path = os.path.join(FORENSMART_ROOT, file_path)
        
        if not os.path.exists(full_path):
            return 0, []
        
        try:
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            matches = []
            for pattern in SIMULATED_PATTERNS:
                found = re.findall(pattern, content, re.MULTILINE | re.DOTALL)
                if found:
                    matches.extend(found)
                    pattern_key = pattern[:50]
                    self.patterns_found[pattern_key] = self.patterns_found.get(pattern_key, 0) + len(found)
            
            return len(matches), matches
        except Exception as e:
            self.log(f"  [ERROR] Failed to scan {file_path}: {e}")
            return 0, []
    
    def remove_simulated_device_lists(self, file_path: str) -> int:
        """Remove simulated device lists from file"""
        full_path = os.path.join(FORENSMART_ROOT, file_path)
        
        if not os.path.exists(full_path):
            return 0
        
        try:
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            new_lines = []
            i = 0
            removed_count = 0
            
            while i < len(lines):
                line = lines[i]
                
                # Check if this is a simulated device list
                if "# Simulated device list" in line or "devices = [" in line:
                    # Skip until we find the closing bracket
                    while i < len(lines) and "]" not in lines[i]:
                        removed_count += 1
                        i += 1
                    if i < len(lines):
                        removed_count += 1
                        i += 1
                    continue
                
                # Check for mock data
                if "mock_" in line or "test_data" in line or "sample_" in line or "SIMULATED_" in line:
                    # Skip this line if it's an assignment
                    if "=" in line and not line.strip().startswith("#"):
                        removed_count += 1
                        i += 1
                        continue
                
                new_lines.append(line)
                i += 1
            
            # Write cleaned content back
            if removed_count > 0:
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.writelines(new_lines)
                self.files_cleaned += 1
                self.lines_removed += removed_count
            
            return removed_count
        except Exception as e:
            self.log(f"  [ERROR] Failed to clean {file_path}: {e}")
            return 0
    
    def clean_all_files(self):
        """Clean all files in the list"""
        self.log("=" * 80)
        self.log("FORENSMART SIMULATED DATA CLEANUP")
        self.log("=" * 80)
        self.log(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log("")
        
        self.log("SCANNING FILES FOR SIMULATED DATA...")
        self.log("-" * 80)
        
        total_matches = 0
        for file_path in FILES_TO_SCAN:
            matches, found_items = self.scan_file(file_path)
            if matches > 0:
                self.log(f"  [{matches}] {file_path}")
                total_matches += matches
        
        self.log("")
        self.log(f"Total simulated data patterns found: {total_matches}")
        self.log("")
        
        self.log("REMOVING SIMULATED DATA...")
        self.log("-" * 80)
        
        for file_path in FILES_TO_SCAN:
            removed = self.remove_simulated_device_lists(file_path)
            if removed > 0:
                self.log(f"  [CLEANED] {file_path} - {removed} lines removed")
        
        self.log("")
        self.log("CLEANUP SUMMARY")
        self.log("-" * 80)
        self.log(f"Files cleaned: {self.files_cleaned}")
        self.log(f"Total lines removed: {self.lines_removed}")
        self.log(f"Patterns found: {len(self.patterns_found)}")
        
        if self.patterns_found:
            self.log("")
            self.log("Pattern breakdown:")
            for pattern, count in sorted(self.patterns_found.items(), key=lambda x: x[1], reverse=True):
                self.log(f"  - {pattern[:60]}: {count} occurrences")
        
        self.log("")
        self.log("=" * 80)
        self.log(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log("=" * 80)
        self.log("")
        self.log("[DONE] CLEANUP COMPLETE!")
        self.log("")
        self.log("NEXT STEPS:")
        self.log("1. Review the changes in the modules")
        self.log("2. Test the application with real data")
        self.log("3. Verify all features work correctly")
        self.log("")
        self.log("ROLLBACK AVAILABLE:")
        self.log("If needed, restore from the rollback copy:")
        rollback_dirs = [d for d in os.listdir("c:\\") if d.startswith("Forensmart_ROLLBACK_")]
        if rollback_dirs:
            latest_rollback = sorted(rollback_dirs)[-1]
            self.log(f"  Copy-Item -Path 'c:\\{latest_rollback}' -Destination 'c:\\Forensmart' -Recurse -Force")
        else:
            self.log("  No rollback copies found")
    
    def save_report(self):
        """Save report to file"""
        try:
            with open(REPORT_FILE, 'w', encoding='utf-8') as f:
                f.write("\n".join(self.report))
            self.log(f"\n[OK] Report saved to: {REPORT_FILE}")
        except Exception as e:
            self.log(f"[ERROR] Failed to save report: {e}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    cleaner = SimulatedDataCleaner()
    cleaner.clean_all_files()
    cleaner.save_report()
    
    print("\n" + "=" * 80)
    print("CLEANUP PROCESS COMPLETE")
    print("=" * 80)
