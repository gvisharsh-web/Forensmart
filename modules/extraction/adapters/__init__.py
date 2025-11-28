"""
ADAPTERS PACKAGE - Device Adapter Framework
Provides adapters for connecting to and extracting data from various devices

This package provides:
- AdapterBase: Abstract base class for all adapters
- AdapterFactory: Factory for creating adapters
- DeviceDetector: Device detection and identification
- Custom exceptions for adapter operations
"""

from .exceptions import (
    AdapterException,
    DeviceNotFound,
    ConnectionFailed,
    AuthenticationFailed,
    ExtractionFailed,
    UnsupportedDevice,
    PermissionDenied
)

from .base import AdapterBase

from .device_detector import DeviceDetector, get_device_detector

from .factory import AdapterFactory, get_adapter_factory

__all__ = [
    # Exceptions
    'AdapterException',
    'DeviceNotFound',
    'ConnectionFailed',
    'AuthenticationFailed',
    'ExtractionFailed',
    'UnsupportedDevice',
    'PermissionDenied',
    
    # Base class
    'AdapterBase',
    
    # Device detection
    'DeviceDetector',
    'get_device_detector',
    
    # Factory
    'AdapterFactory',
    'get_adapter_factory',
]
