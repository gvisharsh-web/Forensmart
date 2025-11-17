"""Comprehensive error checking with auto-recovery across all ForenSmart modules."""
from __future__ import annotations

import os
import json
import logging
import shutil
from typing import Dict, List, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class AutoFixer:
    """Automatic error fixing strategies."""
    
    @staticmethod
    def create_directory(path: str) -> bool:
        """Create missing directory."""
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to create directory {path}: {e}")
            return False
    
    @staticmethod
    def fix_permissions(path: str) -> bool:
        """Fix directory permissions."""
        try:
            os.chmod(path, 0o755)
            logger.info(f"Fixed permissions for: {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to fix permissions for {path}: {e}")
            return False
    
    @staticmethod
    def validate_json_file(path: str) -> bool:
        """Validate and repair JSON files."""
        try:
            with open(path, 'r') as f:
                json.load(f)
            return True
        except json.JSONDecodeError:
            logger.warning(f"JSON file corrupted: {path}, attempting repair...")
            try:
                # Backup original
                backup_path = f"{path}.backup"
                shutil.copy(path, backup_path)
                # Write empty valid JSON
                with open(path, 'w') as f:
                    json.dump({}, f)
                logger.info(f"Repaired JSON file: {path} (backup: {backup_path})")
                return True
            except Exception as e:
                logger.error(f"Failed to repair JSON file: {e}")
                return False
        except Exception as e:
            logger.error(f"Failed to validate JSON file: {e}")
            return False
    
    @staticmethod
    def cleanup_orphaned_files(directory: str, max_age_days: int = 30) -> int:
        """Remove orphaned/old files."""
        import time
        cleaned = 0
        try:
            now = time.time()
            for root, dirs, files in os.walk(directory):
                for file in files:
                    filepath = os.path.join(root, file)
                    file_age = (now - os.path.getmtime(filepath)) / (24 * 3600)
                    if file_age > max_age_days and file.startswith('.'):
                        os.remove(filepath)
                        cleaned += 1
                        logger.info(f"Cleaned orphaned file: {filepath}")
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
        return cleaned


class AppErrorChecker:
    """Check for errors across all ForenSmart modules and data."""

    @staticmethod
    def check_device_detection() -> Dict[str, Any]:
        """Check device detection setup."""
        from modules.device_detector import DeviceDetector
        
        result = {
            "status": "ok",
            "errors": [],
            "warnings": [],
            "device": None,
        }

        diagnosis = DeviceDetector.diagnose()
        
        if not diagnosis["adb_found"]:
            result["status"] = "error"
            result["errors"].extend(diagnosis["errors"])
        else:
            result["warnings"].extend(diagnosis["warnings"])
            if diagnosis["authorized_device"]:
                result["device"] = diagnosis["authorized_device"]
            else:
                result["status"] = "warning"

        return result

    @staticmethod
    def check_consent_module() -> Dict[str, Any]:
        """Check consent module integrity."""
        result = {
            "status": "ok",
            "errors": [],
            "warnings": [],
        }

        try:
            from modules.consent import ConsentManager, ConsentLevel
            cm = ConsentManager()
            
            # Check if sessions can be created
            test_case = "TEST_CASE_CHECK"
            try:
                session = cm.create_session(test_case)
                if not session:
                    result["errors"].append("Failed to create test consent session")
                    result["status"] = "error"
            except ValueError:
                # Session already exists, that's ok
                pass
            except Exception as e:
                result["errors"].append(f"Consent module error: {e}")
                result["status"] = "error"
                
        except ImportError as e:
            result["errors"].append(f"Cannot import ConsentManager: {e}")
            result["status"] = "error"

        return result

    @staticmethod
    def check_extraction_module() -> Dict[str, Any]:
        """Check extraction module integrity."""
        result = {
            "status": "ok",
            "errors": [],
            "warnings": [],
        }

        try:
            from modules.data_extraction_orchestrator import DataExtractionOrchestrator
            from modules.consent import ConsentManager
            
            cm = ConsentManager()
            orchestrator = DataExtractionOrchestrator(cm)
            
            if not orchestrator:
                result["errors"].append("Failed to initialize DataExtractionOrchestrator")
                result["status"] = "error"
                
        except ImportError as e:
            result["errors"].append(f"Cannot import extraction module: {e}")
            result["status"] = "error"
        except Exception as e:
            result["errors"].append(f"Extraction module error: {e}")
            result["status"] = "error"

        return result

    @staticmethod
    def check_approval_system() -> Dict[str, Any]:
        """Check approval/consent portal system."""
        result = {
            "status": "ok",
            "errors": [],
            "warnings": [],
            "approvals_file": None,
        }

        try:
            from modules.approval_utils import get_approvals_file
            
            approvals_file = get_approvals_file()
            result["approvals_file"] = str(approvals_file)
            
            # Check if file is writable
            try:
                approvals_file.parent.mkdir(parents=True, exist_ok=True)
                test_data = {"_test": "ok"}
                approvals_file.write_text(json.dumps(test_data))
                approvals_file.unlink()  # Clean up
            except Exception as e:
                result["warnings"].append(f"Approvals file not writable: {e}")
                
        except ImportError as e:
            result["errors"].append(f"Cannot import approval_utils: {e}")
            result["status"] = "error"

        return result

    @staticmethod
    def check_artifacts_directory() -> Dict[str, Any]:
        """Check artifacts directory structure with auto-fix."""
        result = {
            "status": "ok",
            "errors": [],
            "warnings": [],
            "artifacts_dir": "artifacts",
            "writable": False,
            "auto_fixed": [],
        }

        artifacts_dir = Path("artifacts")
        
        if not artifacts_dir.exists():
            result["warnings"].append("artifacts/ directory does not exist")
            if AutoFixer.create_directory("artifacts"):
                result["auto_fixed"].append("Created artifacts/ directory")
                result["status"] = "warning"
            else:
                result["errors"].append("Cannot create artifacts/ directory")
                result["status"] = "error"
        
        # Check if writable
        try:
            test_file = artifacts_dir / ".write_test"
            test_file.write_text("test")
            test_file.unlink()
            result["writable"] = True
        except Exception as e:
            result["warnings"].append(f"artifacts/ directory not writable: {e}")
            if AutoFixer.fix_permissions("artifacts"):
                result["auto_fixed"].append("Fixed artifacts/ permissions")
                result["status"] = "warning"
            else:
                result["errors"].append(f"Cannot fix artifacts/ permissions: {e}")
                result["status"] = "error"

        return result

    @staticmethod
    def check_reports_directory() -> Dict[str, Any]:
        """Check reports directory structure with auto-fix."""
        result = {
            "status": "ok",
            "errors": [],
            "warnings": [],
            "reports_dir": "reports",
            "writable": False,
            "auto_fixed": [],
        }

        reports_dir = Path("reports")
        
        if not reports_dir.exists():
            result["warnings"].append("reports/ directory does not exist")
            if AutoFixer.create_directory("reports"):
                result["auto_fixed"].append("Created reports/ directory")
                result["status"] = "warning"
            else:
                result["errors"].append("Cannot create reports/ directory")
                result["status"] = "error"
        
        # Check if writable
        try:
            test_file = reports_dir / ".write_test"
            test_file.write_text("test")
            test_file.unlink()
            result["writable"] = True
        except Exception as e:
            result["warnings"].append(f"reports/ directory not writable: {e}")
            if AutoFixer.fix_permissions("reports"):
                result["auto_fixed"].append("Fixed reports/ permissions")
                result["status"] = "warning"
            else:
                result["errors"].append(f"Cannot fix reports/ permissions: {e}")
                result["status"] = "error"

        return result

    @staticmethod
    def check_all() -> Dict[str, Any]:
        """Run all checks and return comprehensive report."""
        return {
            "device_detection": AppErrorChecker.check_device_detection(),
            "consent_module": AppErrorChecker.check_consent_module(),
            "extraction_module": AppErrorChecker.check_extraction_module(),
            "approval_system": AppErrorChecker.check_approval_system(),
            "artifacts_directory": AppErrorChecker.check_artifacts_directory(),
            "reports_directory": AppErrorChecker.check_reports_directory(),
        }

    @staticmethod
    def render_diagnostics_ui():
        """Render diagnostics UI in Streamlit with auto-recovery info."""
        try:
            import streamlit as st
        except ImportError:
            print("Streamlit not available for UI rendering")
            return

        st.markdown("## 🔧 System Diagnostics & Auto-Recovery")
        
        checks = AppErrorChecker.check_all()
        
        for check_name, check_result in checks.items():
            status = check_result.get("status", "unknown")
            icon = "✅" if status == "ok" else "⚠️" if status == "warning" else "❌"
            
            with st.expander(f"{icon} {check_name.replace('_', ' ').title()}"):
                if status != "ok":
                    st.write(f"**Status**: {status.upper()}")
                
                # Show auto-recovered items
                if check_result.get("auto_recovered"):
                    st.success("**Auto-Recovered**:")
                    for recovered in check_result["auto_recovered"]:
                        st.write(f"✅ {recovered}")
                
                # Show auto-fixed items
                if check_result.get("auto_fixed"):
                    st.success("**Auto-Fixed**:")
                    for fixed in check_result["auto_fixed"]:
                        st.write(f"✅ {fixed}")
                
                if check_result.get("errors"):
                    st.error("**Errors**:")
                    for error in check_result["errors"]:
                        st.write(f"- {error}")
                
                if check_result.get("warnings"):
                    st.warning("**Warnings**:")
                    for warning in check_result["warnings"]:
                        st.write(f"- {warning}")
                
                if status == "ok" and not check_result.get("auto_recovered") and not check_result.get("auto_fixed"):
                    st.success("✅ No issues detected")


__all__ = ["AppErrorChecker"]
