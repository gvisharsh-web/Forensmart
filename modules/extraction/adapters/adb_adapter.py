"""
ADB ADAPTER - Android Device Extraction via ADB
Handles extraction from Android devices using Android Debug Bridge

This module provides:
- ADBAdapter class for Android extraction
- ADB connection management
- Data extraction methods
- Error handling for ADB operations
"""

import logging
import subprocess
from typing import Dict, List, Optional, Any
from datetime import datetime

from modules.consent.models import ConsentLevel, MODULE_MIN_LEVELS
from modules.shared.utils import ArtifactPathBuilder, ResultsRepository
from .base import AdapterBase
from .exceptions import ConnectionFailed, ExtractionFailed, PermissionDenied

logger = logging.getLogger(__name__)


# ============================================================================
# ADB ADAPTER CLASS
# ============================================================================

class ADBAdapter(AdapterBase):
    """Android device adapter using ADB"""
    
    def __init__(self, device_id: str, case_id: str, consent_manager=None):
        """Initialize ADB adapter"""
        super().__init__(device_id, case_id, consent_manager)
        self.adapter_type = "Android"
        self.adb_path = "adb"  # Assumes adb is in PATH
        logger.info(f"✅ ADB Adapter initialized for device: {device_id}")
    
    # ========================================================================
    # CONNECTION MANAGEMENT
    # ========================================================================
    
    def connect(self) -> bool:
        """Establish connection to Android device"""
        try:
            logger.info(f"🔌 Connecting to Android device: {self.device_id}")
            
            # Check if device is available
            result = subprocess.run(
                [self.adb_path, '-s', self.device_id, 'shell', 'echo', 'test'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                self.is_connected = True
                self.extraction_status = "connected"
                logger.info(f"✅ Connected to Android device: {self.device_id}")
                return True
            else:
                raise ConnectionFailed(self.device_id, "ADB connection failed")
        except Exception as e:
            logger.error(f"❌ Connection error: {e}")
            raise ConnectionFailed(self.device_id, str(e))
    
    def disconnect(self) -> bool:
        """Close connection to Android device"""
        try:
            logger.info(f"🔌 Disconnecting from Android device: {self.device_id}")
            self.is_connected = False
            self.extraction_status = "disconnected"
            logger.info(f"✅ Disconnected from Android device: {self.device_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Disconnection error: {e}")
            return False
    
    # ========================================================================
    # DATA EXTRACTION METHODS
    # ========================================================================
    
    def extract_data(self) -> Dict[str, Any]:
        """Extract all data from Android device"""
        try:
            if not self.validate_connection():
                return {'error': 'Device not connected'}
            
            logger.info(f"📱 Starting extraction from Android device: {self.device_id}")
            
            results = {
                'device_id': self.device_id,
                'case_id': self.case_id,
                'adapter_type': self.adapter_type,
                'timestamp': datetime.now().isoformat(),
                'modules': {}
            }
            
            # Extract device info
            if self.check_consent('device_info', MODULE_MIN_LEVELS.get('device_info', ConsentLevel.STANDARD)):
                results['modules']['device_info'] = self.extract_device_info()
            
            # Extract communications
            if self.check_consent('communications', MODULE_MIN_LEVELS.get('communications', ConsentLevel.LEGAL)):
                results['modules']['communications'] = self.extract_communications()
            
            # Extract location
            if self.check_consent('location', MODULE_MIN_LEVELS.get('location', ConsentLevel.STANDARD)):
                results['modules']['location'] = self.extract_location()
            
            # Extract media
            if self.check_consent('media', MODULE_MIN_LEVELS.get('media', ConsentLevel.FULL)):
                results['modules']['media'] = self.extract_media()
            
            # Extract apps
            if self.check_consent('security', MODULE_MIN_LEVELS.get('security', ConsentLevel.FULL)):
                results['modules']['apps'] = self.extract_apps()
            
            # Save results
            self.save_results(results, 'android_extraction')
            
            logger.info(f"✅ Extraction complete from Android device: {self.device_id}")
            return results
        except Exception as e:
            logger.error(f"❌ Extraction error: {e}")
            return {'error': str(e), 'device_id': self.device_id}
    
    def extract_device_info(self) -> Dict[str, Any]:
        """Extract device information"""
        try:
            logger.info(f"📱 Extracting device info from: {self.device_id}")
            
            info = {
                'device_id': self.device_id,
                'model': self._execute_adb_command('getprop ro.product.model'),
                'manufacturer': self._execute_adb_command('getprop ro.product.manufacturer'),
                'android_version': self._execute_adb_command('getprop ro.build.version.release'),
                'api_level': self._execute_adb_command('getprop ro.build.version.sdk'),
                'serial': self._execute_adb_command('getprop ro.serialno'),
                'imei': self._execute_adb_command('dumpsys iphonesubinfo | grep Device ID'),
                'timestamp': datetime.now().isoformat()
            }
            
            self.log_operation('extract_device_info', 'success')
            return info
        except Exception as e:
            logger.error(f"❌ Error extracting device info: {e}")
            return self.handle_error(e, 'extract_device_info')
    
    def extract_communications(self) -> Dict[str, Any]:
        """Extract SMS, calls, and contacts"""
        try:
            logger.info(f"💬 Extracting communications from: {self.device_id}")
            
            communications = {
                'sms': self.extract_sms(),
                'calls': self.extract_call_logs(),
                'contacts': self.extract_contacts(),
                'timestamp': datetime.now().isoformat()
            }
            
            self.log_operation('extract_communications', 'success')
            return communications
        except Exception as e:
            logger.error(f"❌ Error extracting communications: {e}")
            return self.handle_error(e, 'extract_communications')
    
    def extract_sms(self) -> List[Dict[str, Any]]:
        """Extract SMS messages"""
        try:
            logger.info(f"📨 Extracting SMS from: {self.device_id}")
            
            # Simulated SMS extraction
            sms_list = [
                {
                    'id': 1,
                    'address': '+1234567890',
                    'body': 'Sample SMS message',
                    'date': datetime.now().isoformat(),
                    'type': 'received'
                }
            ]
            
            return sms_list
        except Exception as e:
            logger.error(f"❌ Error extracting SMS: {e}")
            return []
    
    def extract_call_logs(self) -> List[Dict[str, Any]]:
        """Extract call logs"""
        try:
            logger.info(f"📞 Extracting call logs from: {self.device_id}")
            
            # Simulated call log extraction
            calls = [
                {
                    'id': 1,
                    'number': '+1234567890',
                    'duration': 120,
                    'date': datetime.now().isoformat(),
                    'type': 'incoming'
                }
            ]
            
            return calls
        except Exception as e:
            logger.error(f"❌ Error extracting call logs: {e}")
            return []
    
    def extract_contacts(self) -> List[Dict[str, Any]]:
        """Extract contacts"""
        try:
            logger.info(f"👥 Extracting contacts from: {self.device_id}")
            
            # Simulated contacts extraction
            contacts = [
                {
                    'id': 1,
                    'name': 'John Doe',
                    'phone': '+1234567890',
                    'email': 'john@example.com'
                }
            ]
            
            return contacts
        except Exception as e:
            logger.error(f"❌ Error extracting contacts: {e}")
            return []
    
    def extract_location(self) -> Dict[str, Any]:
        """Extract location data"""
        try:
            logger.info(f"📍 Extracting location data from: {self.device_id}")
            
            location = {
                'gps_data': [],
                'wifi_networks': [],
                'timestamp': datetime.now().isoformat()
            }
            
            return location
        except Exception as e:
            logger.error(f"❌ Error extracting location: {e}")
            return self.handle_error(e, 'extract_location')
    
    def extract_media(self) -> Dict[str, Any]:
        """Extract media files"""
        try:
            logger.info(f"🎬 Extracting media from: {self.device_id}")
            
            media = {
                'photos': [],
                'videos': [],
                'audio': [],
                'timestamp': datetime.now().isoformat()
            }
            
            return media
        except Exception as e:
            logger.error(f"❌ Error extracting media: {e}")
            return self.handle_error(e, 'extract_media')
    
    def extract_apps(self) -> List[Dict[str, Any]]:
        """Extract installed apps"""
        try:
            logger.info(f"📦 Extracting apps from: {self.device_id}")
            
            # Simulated apps extraction
            apps = [
                {
                    'package': 'com.example.app',
                    'name': 'Example App',
                    'version': '1.0.0',
                    'install_date': datetime.now().isoformat()
                }
            ]
            
            return apps
        except Exception as e:
            logger.error(f"❌ Error extracting apps: {e}")
            return []
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def _execute_adb_command(self, command: str) -> str:
        """Execute ADB command and return output"""
        try:
            result = subprocess.run(
                [self.adb_path, '-s', self.device_id, 'shell'] + command.split(),
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                logger.warning(f"⚠️ ADB command failed: {command}")
                return ""
        except Exception as e:
            logger.error(f"❌ Error executing ADB command: {e}")
            return ""
    
    def check_root_access(self) -> bool:
        """Check if device has root access"""
        try:
            result = subprocess.run(
                [self.adb_path, '-s', self.device_id, 'shell', 'id'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            return 'uid=0' in result.stdout
        except Exception as e:
            logger.error(f"❌ Error checking root access: {e}")
            return False
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get device information"""
        try:
            info = {
                'device_id': self.device_id,
                'adapter_type': self.adapter_type,
                'is_connected': self.is_connected,
                'model': self._execute_adb_command('getprop ro.product.model'),
                'android_version': self._execute_adb_command('getprop ro.build.version.release'),
                'has_root': self.check_root_access(),
                'timestamp': datetime.now().isoformat()
            }
            
            return info
        except Exception as e:
            logger.error(f"❌ Error getting device info: {e}")
            return {}
