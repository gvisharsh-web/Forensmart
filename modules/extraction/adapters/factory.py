"""
ADAPTER FACTORY - Factory Pattern for Adapter Creation
Creates appropriate adapter instances based on device type

This module provides:
- AdapterFactory class
- Adapter registration
- Device type validation
"""

import logging
from typing import Dict, Type, Optional, Any

from .base import AdapterBase
from .exceptions import UnsupportedDevice, AdapterException

logger = logging.getLogger(__name__)


# ============================================================================
# ADAPTER FACTORY CLASS
# ============================================================================

class AdapterFactory:
    """Factory for creating adapter instances"""
    
    def __init__(self):
        """Initialize adapter factory"""
        self.adapters: Dict[str, Type[AdapterBase]] = {}
        self._register_default_adapters()
        logger.info("✅ Adapter factory initialized")
    
    def _register_default_adapters(self):
        """Register default adapters"""
        # Note: Actual adapter classes will be imported when they're created
        # For now, we just define the mapping
        self.adapter_types = {
            'Android': 'ADBAdapter',
            'iOS': 'iOSAdapter',
            'HDD': 'HDDAdapter',
            'WhatsApp': 'WhatsAppAdapter',
            'Instagram': 'InstagramAdapter',
            'Telegram': 'TelegramAdapter',
            'Facebook': 'FacebookAdapter',
            'Snapchat': 'SnapchatAdapter',
            'GoogleDrive': 'GoogleDriveAdapter',
            'Email': 'EmailAdapter'
        }
        logger.info(f"✅ Registered {len(self.adapter_types)} adapter types")
    
    def register_adapter(self, device_type: str, adapter_class: Type[AdapterBase]) -> bool:
        """Register a new adapter"""
        try:
            self.adapters[device_type] = adapter_class
            logger.info(f"✅ Adapter registered: {device_type} -> {adapter_class.__name__}")
            return True
        except Exception as e:
            logger.error(f"❌ Error registering adapter: {e}")
            return False
    
    def create_adapter(
        self,
        device_type: str,
        device_id: str,
        case_id: str,
        consent_manager=None
    ) -> Optional[AdapterBase]:
        """Create adapter instance for device type"""
        try:
            if not self.validate_adapter_type(device_type):
                raise UnsupportedDevice(device_type=device_type, device_id=device_id)
            
            if device_type not in self.adapters:
                logger.warning(f"⚠️ Adapter not yet implemented: {device_type}")
                return None
            
            adapter_class = self.adapters[device_type]
            adapter = adapter_class(device_id, case_id, consent_manager)
            
            logger.info(f"✅ Adapter created: {device_type} for device {device_id}")
            return adapter
        except UnsupportedDevice as e:
            logger.error(f"❌ {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Error creating adapter: {e}")
            return None
    
    def get_available_adapters(self) -> Dict[str, str]:
        """Get all available adapter types"""
        return self.adapter_types.copy()
    
    def validate_adapter_type(self, device_type: str) -> bool:
        """Validate that adapter type is supported"""
        if device_type not in self.adapter_types:
            logger.warning(f"⚠️ Unknown adapter type: {device_type}")
            return False
        logger.info(f"✅ Adapter type validated: {device_type}")
        return True
    
    def get_adapter_for_device(self, device_id: str, device_type: str) -> Optional[Type[AdapterBase]]:
        """Get adapter class for device"""
        if device_type not in self.adapters:
            logger.warning(f"⚠️ No adapter registered for type: {device_type}")
            return None
        return self.adapters[device_type]
    
    def list_supported_devices(self) -> list:
        """Get list of supported device types"""
        return list(self.adapter_types.keys())
    
    def get_factory_summary(self) -> Dict[str, Any]:
        """Get factory summary"""
        return {
            'total_adapter_types': len(self.adapter_types),
            'registered_adapters': len(self.adapters),
            'supported_devices': self.list_supported_devices(),
            'implemented_adapters': list(self.adapters.keys())
        }


# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

_adapter_factory_instance: Optional[AdapterFactory] = None


def get_adapter_factory() -> AdapterFactory:
    """Get global adapter factory instance"""
    global _adapter_factory_instance
    if _adapter_factory_instance is None:
        _adapter_factory_instance = AdapterFactory()
    return _adapter_factory_instance
