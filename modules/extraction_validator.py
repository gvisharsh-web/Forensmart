"""Smart extraction validation and error prevention."""
from __future__ import annotations

import os
import logging
import shutil
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class ExtractionValidator:
    """Validate extraction prerequisites and prevent errors."""

    @staticmethod
    def check_device_ready(device_id: str) -> Tuple[bool, List[str]]:
        """Check if device is ready for extraction."""
        errors = []

        if not device_id or device_id == "UNKNOWN_DEVICE":
            errors.append("❌ Device ID not set or unknown - Connect device via USB and enable USB Debugging")
            return False, errors

        try:
            from modules.device_detector import DeviceDetector
            
            devices = DeviceDetector.list_devices()
            
            if not devices:
                errors.append("❌ No devices found - Ensure device is connected via USB and ADB is installed")
                return False, errors
            
            device_found = any(d["serial"] == device_id for d in devices)
            
            if not device_found:
                available = ", ".join([d["serial"] for d in devices])
                errors.append(f"❌ Device {device_id} not found. Available: {available}")
                return False, errors
            
            # Check if authorized
            auth_device = DeviceDetector.get_authorized_device()
            if not auth_device:
                errors.append(f"❌ Device {device_id} is not authorized - Accept RSA prompt on device")
                return False, errors
            
            if auth_device["serial"] != device_id:
                errors.append(f"❌ Device {device_id} is not authorized - Use authorized device: {auth_device['serial']}")
                return False, errors
                
        except Exception as e:
            errors.append(f"❌ Device check failed: {str(e)} - Check ADB installation and device connection")
            return False, errors

        return True, []

    @staticmethod
    def check_storage_space(case_id: str, min_mb: int = 500) -> Tuple[bool, List[str]]:
        """Check if sufficient storage space is available."""
        errors = []

        try:
            artifacts_dir = Path("artifacts")
            if not artifacts_dir.exists():
                artifacts_dir.mkdir(parents=True, exist_ok=True)

            # Get free space
            stat = shutil.disk_usage(str(artifacts_dir))
            free_mb = stat.free / (1024 * 1024)

            if free_mb < min_mb:
                errors.append(
                    f"Insufficient storage space: {free_mb:.1f}MB available, "
                    f"{min_mb}MB required"
                )
                return False, errors

            logger.info(f"Storage check passed: {free_mb:.1f}MB available")
            return True, []

        except Exception as e:
            errors.append(f"Storage check failed: {e}")
            return False, errors

    @staticmethod
    def check_consent_level(session: Any, required_level: Any) -> Tuple[bool, List[str]]:
        """Check if consent level is sufficient."""
        errors = []

        if not session:
            errors.append("No consent session found")
            return False, errors

        if session.level.value < required_level.value:
            errors.append(
                f"Insufficient consent level: {session.level.name} < {required_level.name}"
            )
            return False, errors

        logger.info(f"Consent check passed: {session.level.name}")
        return True, []

    @staticmethod
    def check_approval_status(case_id: str) -> Tuple[bool, List[str]]:
        """Check if nominee has approved extraction."""
        errors = []

        try:
            from modules.approval_utils import get_approval_decision
            
            decision = get_approval_decision(case_id)
            
            if decision == "denied":
                errors.append("❌ Nominee denied extraction request - Generate a new approval link in the Consent tab")
                return False, errors
            
            if decision != "approved":
                errors.append("⏳ Awaiting nominee approval for extraction - Share approval link from Consent tab")
                return False, errors

            logger.info("Approval check passed")
            return True, []

        except Exception as e:
            errors.append(f"❌ Approval check failed: {str(e)} - Check approval system and try again")
            return False, errors

    @staticmethod
    def check_directories_writable(case_id: str) -> Tuple[bool, List[str]]:
        """Check if required directories are writable."""
        errors = []
        dirs_to_check = ["artifacts", "reports"]

        for dir_name in dirs_to_check:
            dir_path = Path(dir_name)
            
            if not dir_path.exists():
                try:
                    dir_path.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    errors.append(f"Cannot create {dir_name}/ directory: {e}")
                    continue

            # Test write permission
            try:
                test_file = dir_path / ".write_test"
                test_file.write_text("test")
                test_file.unlink()
            except Exception as e:
                errors.append(f"{dir_name}/ directory not writable: {e}")

        if errors:
            return False, errors

        logger.info("Directory check passed")
        return True, []

    @staticmethod
    def validate_extraction_ready(
        case_id: str,
        device_id: str,
        session: Any,
        required_level: Any
    ) -> Dict[str, Any]:
        """Comprehensive extraction readiness check."""
        result = {
            "ready": True,
            "errors": [],
            "warnings": [],
            "checks": {
                "device": {"passed": False, "errors": []},
                "storage": {"passed": False, "errors": []},
                "consent": {"passed": False, "errors": []},
                "approval": {"passed": False, "errors": []},
                "directories": {"passed": False, "errors": []},
            }
        }

        # Device check
        passed, errors = ExtractionValidator.check_device_ready(device_id)
        result["checks"]["device"]["passed"] = passed
        result["checks"]["device"]["errors"] = errors
        if not passed:
            result["ready"] = False
            result["errors"].extend(errors)

        # Storage check
        passed, errors = ExtractionValidator.check_storage_space(case_id)
        result["checks"]["storage"]["passed"] = passed
        result["checks"]["storage"]["errors"] = errors
        if not passed:
            result["ready"] = False
            result["errors"].extend(errors)

        # Consent check
        passed, errors = ExtractionValidator.check_consent_level(session, required_level)
        result["checks"]["consent"]["passed"] = passed
        result["checks"]["consent"]["errors"] = errors
        if not passed:
            result["ready"] = False
            result["errors"].extend(errors)

        # Approval check
        passed, errors = ExtractionValidator.check_approval_status(case_id)
        result["checks"]["approval"]["passed"] = passed
        result["checks"]["approval"]["errors"] = errors
        if not passed:
            result["warnings"].extend(errors)

        # Directory check
        passed, errors = ExtractionValidator.check_directories_writable(case_id)
        result["checks"]["directories"]["passed"] = passed
        result["checks"]["directories"]["errors"] = errors
        if not passed:
            result["ready"] = False
            result["errors"].extend(errors)

        return result


__all__ = ["ExtractionValidator"]
