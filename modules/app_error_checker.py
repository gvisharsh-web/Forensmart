"""Comprehensive error checking across all ForenSmart modules."""
from __future__ import annotations

import os
import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


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
        """Check artifacts directory structure."""
        result = {
            "status": "ok",
            "errors": [],
            "warnings": [],
            "artifacts_dir": "artifacts",
            "writable": False,
        }

        artifacts_dir = Path("artifacts")
        
        if not artifacts_dir.exists():
            result["warnings"].append("artifacts/ directory does not exist")
            try:
                artifacts_dir.mkdir(parents=True, exist_ok=True)
                result["status"] = "warning"
            except Exception as e:
                result["errors"].append(f"Cannot create artifacts/ directory: {e}")
                result["status"] = "error"
        
        # Check if writable
        try:
            test_file = artifacts_dir / ".write_test"
            test_file.write_text("test")
            test_file.unlink()
            result["writable"] = True
        except Exception as e:
            result["errors"].append(f"artifacts/ directory not writable: {e}")
            result["status"] = "error"

        return result

    @staticmethod
    def check_reports_directory() -> Dict[str, Any]:
        """Check reports directory structure."""
        result = {
            "status": "ok",
            "errors": [],
            "warnings": [],
            "reports_dir": "reports",
            "writable": False,
        }

        reports_dir = Path("reports")
        
        if not reports_dir.exists():
            result["warnings"].append("reports/ directory does not exist")
            try:
                reports_dir.mkdir(parents=True, exist_ok=True)
                result["status"] = "warning"
            except Exception as e:
                result["errors"].append(f"Cannot create reports/ directory: {e}")
                result["status"] = "error"
        
        # Check if writable
        try:
            test_file = reports_dir / ".write_test"
            test_file.write_text("test")
            test_file.unlink()
            result["writable"] = True
        except Exception as e:
            result["errors"].append(f"reports/ directory not writable: {e}")
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
        """Render diagnostics UI in Streamlit."""
        try:
            import streamlit as st
        except ImportError:
            print("Streamlit not available for UI rendering")
            return

        st.markdown("## 🔧 System Diagnostics")
        
        checks = AppErrorChecker.check_all()
        
        for check_name, check_result in checks.items():
            status = check_result.get("status", "unknown")
            icon = "✅" if status == "ok" else "⚠️" if status == "warning" else "❌"
            
            with st.expander(f"{icon} {check_name.replace('_', ' ').title()}"):
                if status != "ok":
                    st.write(f"**Status**: {status.upper()}")
                
                if check_result.get("errors"):
                    st.error("**Errors**:")
                    for error in check_result["errors"]:
                        st.write(f"- {error}")
                
                if check_result.get("warnings"):
                    st.warning("**Warnings**:")
                    for warning in check_result["warnings"]:
                        st.write(f"- {warning}")
                
                if status == "ok":
                    st.success("No issues detected")


__all__ = ["AppErrorChecker"]
