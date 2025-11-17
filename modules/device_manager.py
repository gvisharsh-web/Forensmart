"""Enhanced device management with multi-device support."""
from __future__ import annotations

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DeviceInfo:
    """Device information container."""
    serial: str
    status: str
    model: Optional[str] = None
    android_version: Optional[str] = None
    has_root: bool = False
    battery_level: Optional[int] = None
    storage_free_mb: Optional[int] = None
    is_connected: bool = False


class DeviceManager:
    """Manage multiple connected devices."""

    @staticmethod
    def list_all_devices() -> List[DeviceInfo]:
        """List all connected devices with details."""
        try:
            from modules.device_detector import DeviceDetector
            
            devices = DeviceDetector.list_devices()
            device_list = []
            
            for device in devices:
                serial = device.get("serial")
                status = device.get("status")
                
                # Get detailed info if authorized
                if status == "device":
                    info = DeviceDetector.get_device_info(serial)
                    device_obj = DeviceInfo(
                        serial=serial,
                        status=status,
                        model=info.get("model"),
                        android_version=info.get("android_version"),
                        has_root=info.get("has_root", False),
                        is_connected=True
                    )
                else:
                    device_obj = DeviceInfo(
                        serial=serial,
                        status=status,
                        is_connected=False
                    )
                
                device_list.append(device_obj)
            
            return device_list
        except Exception as e:
            logger.error(f"Failed to list devices: {e}")
            return []

    @staticmethod
    def get_device_by_serial(serial: str) -> Optional[DeviceInfo]:
        """Get specific device by serial."""
        devices = DeviceManager.list_all_devices()
        for device in devices:
            if device.serial == serial:
                return device
        return None

    @staticmethod
    def get_authorized_devices() -> List[DeviceInfo]:
        """Get only authorized devices."""
        devices = DeviceManager.list_all_devices()
        return [d for d in devices if d.status == "device"]

    @staticmethod
    def get_unauthorized_devices() -> List[DeviceInfo]:
        """Get unauthorized devices."""
        devices = DeviceManager.list_all_devices()
        return [d for d in devices if d.status == "unauthorized"]

    @staticmethod
    def get_offline_devices() -> List[DeviceInfo]:
        """Get offline devices."""
        devices = DeviceManager.list_all_devices()
        return [d for d in devices if d.status == "offline"]

    @staticmethod
    def get_device_health(serial: str) -> Dict[str, Any]:
        """Get device health status."""
        device = DeviceManager.get_device_by_serial(serial)
        if not device:
            return {"status": "unknown", "errors": ["Device not found"]}

        health = {
            "serial": serial,
            "status": device.status,
            "connected": device.is_connected,
            "issues": [],
            "warnings": [],
        }

        if not device.is_connected:
            health["issues"].append("Device not connected")
        
        if device.status == "unauthorized":
            health["issues"].append("Device unauthorized - accept RSA prompt")
        
        if device.status == "offline":
            health["issues"].append("Device offline - check USB connection")
        
        if device.battery_level is not None and device.battery_level < 20:
            health["warnings"].append(f"Low battery: {device.battery_level}%")
        
        if device.storage_free_mb is not None and device.storage_free_mb < 500:
            health["warnings"].append(f"Low storage: {device.storage_free_mb}MB free")

        return health

    @staticmethod
    def select_device_for_extraction(case_id: str, session: Any) -> Optional[str]:
        """Select best device for extraction."""
        authorized = DeviceManager.get_authorized_devices()
        
        if not authorized:
            logger.warning("No authorized devices available")
            return None
        
        # Prefer device with most storage
        best_device = max(
            authorized,
            key=lambda d: d.storage_free_mb or 0
        )
        
        logger.info(f"Selected device {best_device.serial} for extraction")
        return best_device.serial

    @staticmethod
    def validate_device_for_extraction(serial: str) -> Dict[str, Any]:
        """Validate device is suitable for extraction."""
        device = DeviceManager.get_device_by_serial(serial)
        
        result = {
            "valid": True,
            "errors": [],
            "warnings": [],
        }

        if not device:
            result["valid"] = False
            result["errors"].append(f"Device {serial} not found")
            return result

        if not device.is_connected:
            result["valid"] = False
            result["errors"].append("Device not connected")

        if device.status != "device":
            result["valid"] = False
            result["errors"].append(f"Device status: {device.status}")

        if device.battery_level and device.battery_level < 20:
            result["warnings"].append(f"Low battery: {device.battery_level}%")

        if device.storage_free_mb and device.storage_free_mb < 1000:
            result["warnings"].append(f"Low storage: {device.storage_free_mb}MB")

        return result


__all__ = ["DeviceManager", "DeviceInfo"]
