"""Robust device detection with fallbacks and error handling."""
from __future__ import annotations

import os
import subprocess
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path

logger = logging.getLogger(__name__)


class DeviceDetector:
    """Reliable device detection with multiple fallback strategies."""

    @staticmethod
    def find_adb_executable() -> Optional[str]:
        """Find ADB executable in common locations."""
        candidates = [
            "adb",
            os.path.join("driver_bundle", "platform-tools", "adb.exe"),
            os.path.join("driver_bundle", "platform-tools", "adb"),
            os.path.join("platform-tools", "adb.exe"),
            os.path.join("platform-tools", "adb"),
            os.path.expanduser("~/.android/platform-tools/adb"),
            "C:\\android-sdk\\platform-tools\\adb.exe",
            "/usr/bin/adb",
            "/usr/local/bin/adb",
        ]

        for candidate in candidates:
            try:
                result = subprocess.run(
                    [candidate, "version"],
                    capture_output=True,
                    timeout=5,
                    text=True
                )
                if result.returncode == 0:
                    logger.info(f"Found ADB at: {candidate}")
                    return candidate
            except Exception:
                continue

        return None

    @staticmethod
    def list_devices(adb_path: Optional[str] = None) -> List[Dict[str, str]]:
        """List all connected devices."""
        adb = adb_path or DeviceDetector.find_adb_executable()
        if not adb:
            logger.warning("ADB not found in PATH or common locations")
            return []

        try:
            result = subprocess.run(
                [adb, "devices"],
                capture_output=True,
                timeout=10,
                text=True
            )
            if result.returncode != 0:
                logger.error(f"ADB devices failed: {result.stderr}")
                return []

            devices = []
            for line in result.stdout.splitlines()[1:]:
                line = line.strip()
                if not line:
                    continue
                if "\t" in line:
                    serial, status = line.split("\t", 1)
                else:
                    serial, status = line, ""
                devices.append({"serial": serial.strip(), "status": status.strip()})

            logger.info(f"Found {len(devices)} device(s)")
            return devices
        except subprocess.TimeoutExpired:
            logger.error("ADB devices command timed out")
            return []
        except Exception as e:
            logger.error(f"Failed to list devices: {e}")
            return []

    @staticmethod
    def get_authorized_device(adb_path: Optional[str] = None) -> Optional[Dict[str, str]]:
        """Get first authorized device."""
        devices = DeviceDetector.list_devices(adb_path)
        for device in devices:
            if device.get("status") == "device":
                logger.info(f"Found authorized device: {device['serial']}")
                return device
        logger.warning("No authorized devices found")
        return None

    @staticmethod
    def get_device_info(device_serial: str, adb_path: Optional[str] = None) -> Dict[str, Any]:
        """Get device information."""
        adb = adb_path or DeviceDetector.find_adb_executable()
        if not adb:
            return {"error": "ADB not found"}

        info = {"serial": device_serial}

        # Get model
        try:
            result = subprocess.run(
                [adb, "-s", device_serial, "shell", "getprop", "ro.product.model"],
                capture_output=True,
                timeout=5,
                text=True
            )
            if result.returncode == 0:
                info["model"] = result.stdout.strip()
        except Exception as e:
            logger.debug(f"Failed to get model: {e}")

        # Get Android version
        try:
            result = subprocess.run(
                [adb, "-s", device_serial, "shell", "getprop", "ro.build.version.release"],
                capture_output=True,
                timeout=5,
                text=True
            )
            if result.returncode == 0:
                info["android_version"] = result.stdout.strip()
        except Exception as e:
            logger.debug(f"Failed to get Android version: {e}")

        # Check root access
        try:
            result = subprocess.run(
                [adb, "-s", device_serial, "shell", "id"],
                capture_output=True,
                timeout=5,
                text=True
            )
            if result.returncode == 0:
                info["has_root"] = "uid=0" in result.stdout
        except Exception as e:
            logger.debug(f"Failed to check root: {e}")

        return info

    @staticmethod
    def diagnose() -> Dict[str, Any]:
        """Run full device detection diagnostics."""
        diagnosis = {
            "adb_found": False,
            "adb_path": None,
            "devices": [],
            "authorized_device": None,
            "errors": [],
            "warnings": [],
        }

        # Check ADB
        adb_path = DeviceDetector.find_adb_executable()
        if adb_path:
            diagnosis["adb_found"] = True
            diagnosis["adb_path"] = adb_path
        else:
            diagnosis["errors"].append("ADB executable not found in PATH or common locations")
            return diagnosis

        # List devices
        devices = DeviceDetector.list_devices(adb_path)
        diagnosis["devices"] = devices

        if not devices:
            diagnosis["warnings"].append("No devices listed by ADB")
        else:
            # Check for unauthorized devices
            for device in devices:
                if device.get("status") == "unauthorized":
                    diagnosis["warnings"].append(
                        f"Device {device['serial']} is unauthorized. "
                        "Accept the RSA prompt on the device."
                    )
                elif device.get("status") == "offline":
                    diagnosis["warnings"].append(
                        f"Device {device['serial']} is offline. "
                        "Check USB connection."
                    )

        # Get authorized device
        auth_device = DeviceDetector.get_authorized_device(adb_path)
        if auth_device:
            diagnosis["authorized_device"] = auth_device
            device_info = DeviceDetector.get_device_info(auth_device["serial"], adb_path)
            diagnosis["device_info"] = device_info
        else:
            diagnosis["warnings"].append("No authorized devices found")

        return diagnosis


__all__ = ["DeviceDetector"]
