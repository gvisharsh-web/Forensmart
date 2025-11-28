"""
ADAPTER EXCEPTIONS - Custom Exception Classes
Defines all custom exceptions for adapter operations

This module provides:
- AdapterException (base exception)
- DeviceNotFound
- ConnectionFailed
- AuthenticationFailed
- ExtractionFailed
- UnsupportedDevice
- PermissionDenied
"""

import logging

logger = logging.getLogger(__name__)


# ============================================================================
# BASE ADAPTER EXCEPTION
# ============================================================================

class AdapterException(Exception):
    """Base exception for all adapter operations"""
    
    def __init__(self, message: str, error_code: str = None, device_id: str = None):
        self.message = message
        self.error_code = error_code or "ADAPTER_ERROR"
        self.device_id = device_id
        super().__init__(self.message)
    
    def __str__(self):
        if self.device_id:
            return f"[{self.error_code}] Device: {self.device_id} - {self.message}"
        return f"[{self.error_code}] {self.message}"


# ============================================================================
# SPECIFIC ADAPTER EXCEPTIONS
# ============================================================================

class DeviceNotFound(AdapterException):
    """Raised when device cannot be found or detected"""
    
    def __init__(self, device_id: str = None, message: str = None):
        msg = message or f"Device not found: {device_id}" if device_id else "No devices found"
        super().__init__(msg, "DEVICE_NOT_FOUND", device_id)
        logger.error(f"❌ {msg}")


class ConnectionFailed(AdapterException):
    """Raised when connection to device fails"""
    
    def __init__(self, device_id: str = None, reason: str = None):
        msg = f"Connection failed" + (f": {reason}" if reason else "")
        super().__init__(msg, "CONNECTION_FAILED", device_id)
        logger.error(f"❌ {msg}")


class AuthenticationFailed(AdapterException):
    """Raised when authentication to device fails"""
    
    def __init__(self, device_id: str = None, reason: str = None):
        msg = f"Authentication failed" + (f": {reason}" if reason else "")
        super().__init__(msg, "AUTH_FAILED", device_id)
        logger.error(f"❌ {msg}")


class ExtractionFailed(AdapterException):
    """Raised when data extraction fails"""
    
    def __init__(self, device_id: str = None, module: str = None, reason: str = None):
        msg = f"Extraction failed"
        if module:
            msg += f" for module: {module}"
        if reason:
            msg += f": {reason}"
        super().__init__(msg, "EXTRACTION_FAILED", device_id)
        logger.error(f"❌ {msg}")


class UnsupportedDevice(AdapterException):
    """Raised when device type is not supported"""
    
    def __init__(self, device_type: str = None, device_id: str = None):
        msg = f"Unsupported device type" + (f": {device_type}" if device_type else "")
        super().__init__(msg, "UNSUPPORTED_DEVICE", device_id)
        logger.error(f"❌ {msg}")


class PermissionDenied(AdapterException):
    """Raised when permission is denied for operation"""
    
    def __init__(self, device_id: str = None, operation: str = None, reason: str = None):
        msg = f"Permission denied"
        if operation:
            msg += f" for operation: {operation}"
        if reason:
            msg += f": {reason}"
        super().__init__(msg, "PERMISSION_DENIED", device_id)
        logger.error(f"❌ {msg}")
