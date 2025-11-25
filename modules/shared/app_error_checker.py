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
        from modules.shared.device_detector import DeviceDetector
        
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
            from modules.consent.models import ConsentManager, ConsentLevel
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
            from modules.extraction.orchestrator import DataExtractionOrchestrator
            from modules.consent.models import ConsentManager
            
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
            from modules.approval.utils import get_approvals_file
            
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
        """Render diagnostics UI in Streamlit with actionable steps and AI-like redirects."""
        try:
            import streamlit as st
        except ImportError:
            print("Streamlit not available for UI rendering")
            return

        st.markdown("## 🔧 System Diagnostics & Auto-Recovery")
        
        checks = AppErrorChecker.check_all()
        
        # Summary section
        st.markdown("### 📊 System Health Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        error_count = sum(1 for c in checks.values() if c.get("status") == "error")
        warning_count = sum(1 for c in checks.values() if c.get("status") == "warning")
        ok_count = sum(1 for c in checks.values() if c.get("status") == "ok")
        
        with col1:
            st.metric("✅ Healthy", ok_count)
        with col2:
            st.metric("⚠️ Warnings", warning_count)
        with col3:
            st.metric("❌ Errors", error_count)
        with col4:
            health_pct = int((ok_count / len(checks)) * 100) if checks else 0
            st.metric("🏥 Health", f"{health_pct}%")
        
        st.divider()
        
        # Detailed checks with actionable steps
        for check_name, check_result in checks.items():
            status = check_result.get("status", "unknown")
            icon = "✅" if status == "ok" else "⚠️" if status == "warning" else "❌"
            
            with st.expander(f"{icon} {check_name.replace('_', ' ').title()}", expanded=(status != "ok")):
                if status != "ok":
                    st.write(f"**Status**: {status.upper()}")
                
                # Show auto-recovered items
                if check_result.get("auto_recovered"):
                    st.success("**✅ Auto-Recovered**:")
                    for recovered in check_result["auto_recovered"]:
                        st.write(f"✅ {recovered}")
                
                # Show auto-fixed items
                if check_result.get("auto_fixed"):
                    st.success("**✅ Auto-Fixed**:")
                    for fixed in check_result["auto_fixed"]:
                        st.write(f"✅ {fixed}")
                
                # Show errors with action steps
                if check_result.get("errors"):
                    st.error("**❌ Errors Detected**:")
                    for error in check_result["errors"]:
                        st.write(f"- {error}")
                    
                    # AI-like actionable steps
                    st.markdown("**🤖 Recommended Actions:**")
                    actions = AppErrorChecker._get_action_steps(check_name, check_result)
                    for i, action in enumerate(actions, 1):
                        st.write(f"{i}. {action}")
                    
                    # Redirect to relevant module
                    redirect = AppErrorChecker._get_module_redirect(check_name)
                    if redirect:
                        st.info(f"💡 **Next Step**: {redirect['message']}")
                        if st.button(f"🔗 Go to {redirect['module']}", key=f"redirect_{check_name}"):
                            st.session_state['active_tab'] = redirect['tab']
                            st.rerun()
                
                # Show warnings with action steps
                if check_result.get("warnings"):
                    st.warning("**⚠️ Warnings**:")
                    for warning in check_result["warnings"]:
                        st.write(f"- {warning}")
                    
                    # AI-like actionable steps for warnings
                    st.markdown("**🤖 Recommended Actions:**")
                    actions = AppErrorChecker._get_warning_steps(check_name, check_result)
                    for i, action in enumerate(actions, 1):
                        st.write(f"{i}. {action}")
                
                if status == "ok" and not check_result.get("auto_recovered") and not check_result.get("auto_fixed"):
                    st.success("✅ **No issues detected** - System is healthy!")
    
    @staticmethod
    def _get_action_steps(check_name: str, result: Dict[str, Any]) -> List[str]:
        """Get AI-like actionable steps based on error type."""
        steps = []
        
        if "device" in check_name.lower():
            steps = [
                "Connect your Android device via USB cable",
                "Enable USB Debugging on your device (Settings > Developer Options)",
                "Accept the RSA fingerprint prompt on your device",
                "Restart ADB daemon: adb kill-server && adb start-server",
                "Verify connection: adb devices",
                "Return to Consent tab and click 'Refresh device detection'"
            ]
        elif "consent" in check_name.lower():
            steps = [
                "Check if consent module is properly initialized",
                "Verify approval_utils.py exists in modules/",
                "Ensure approvals.json file has proper permissions",
                "Go to Consent tab and create a new case session",
                "Generate an approval link and share with nominee"
            ]
        elif "extraction" in check_name.lower():
            steps = [
                "Verify device is connected and authorized",
                "Ensure sufficient storage space on device",
                "Check that consent level is at least STANDARD",
                "Verify nominee has approved the extraction request",
                "Go to Extraction tab and try again"
            ]
        elif "approval" in check_name.lower():
            steps = [
                "Check ~/.forensmart/approvals.json file exists",
                "Verify JSON file is not corrupted",
                "Ensure nominee has clicked approval link",
                "Check approval link expiration (24 hours)",
                "Generate new approval link if expired"
            ]
        elif "artifacts" in check_name.lower():
            steps = [
                "Create artifacts/ directory if missing",
                "Fix directory permissions (chmod 755)",
                "Ensure sufficient disk space available",
                "Check that artifacts/ is writable",
                "Restart the application"
            ]
        elif "reports" in check_name.lower():
            steps = [
                "Create reports/ directory if missing",
                "Fix directory permissions (chmod 755)",
                "Ensure sufficient disk space available",
                "Check that reports/ is writable",
                "Restart the application"
            ]
        
        return steps if steps else ["Check system logs for more details", "Contact support if issue persists"]
    
    @staticmethod
    def _get_warning_steps(check_name: str, result: Dict[str, Any]) -> List[str]:
        """Get AI-like actionable steps for warnings."""
        steps = []
        
        if "device" in check_name.lower():
            steps = [
                "Device may be temporarily disconnected",
                "Try reconnecting the device",
                "Verify USB cable is working properly",
                "Check device is not in sleep mode",
                "Click 'Refresh device detection' in Consent tab"
            ]
        elif "directory" in check_name.lower():
            steps = [
                "Directory will be created automatically on first use",
                "Ensure parent directory has write permissions",
                "Check available disk space",
                "No immediate action required"
            ]
        
        return steps if steps else ["Monitor the situation", "Perform action if issue worsens"]
    
    @staticmethod
    def _get_module_redirect(check_name: str) -> Optional[Dict[str, str]]:
        """Get AI-like module redirect suggestions."""
        redirects = {
            "device_detection": {
                "module": "Device Manager",
                "tab": "Diagnostics",
                "message": "Go to Diagnostics tab to manage devices"
            },
            "consent_module": {
                "module": "Consent Manager",
                "tab": "Consent",
                "message": "Go to Consent tab to manage approvals"
            },
            "extraction_module": {
                "module": "Extraction",
                "tab": "Extraction",
                "message": "Go to Extraction tab to start extraction"
            },
            "approval_system": {
                "module": "Consent Manager",
                "tab": "Consent",
                "message": "Go to Consent tab to generate approval links"
            },
            "artifacts_directory": {
                "module": "Storage",
                "tab": "Storage",
                "message": "Go to Storage tab to manage artifacts"
            },
            "reports_directory": {
                "module": "Reports",
                "tab": "Reports",
                "message": "Go to Reports tab to manage reports"
            }
        }
        
        return redirects.get(check_name)


__all__ = ["AppErrorChecker"]
