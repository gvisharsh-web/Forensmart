"""
DEVICE DETECTOR - Detect Connected Devices
Scans for and identifies connected devices (Android, iOS, HDD)

This module provides:
- DeviceDetector class
- Device detection methods
- Device type identification
- Device capability checking
"""

import logging
import subprocess
import platform
import os
from typing import Dict, List, Optional, Any
from datetime import datetime

# Import validators
try:
    from modules.shared.validators import validate_device_id, validate_file_path
    VALIDATORS_AVAILABLE = True
except ImportError:
    VALIDATORS_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("⚠️ Validators not available")

logger = logging.getLogger(__name__)


# ============================================================================
# DEVICE DETECTOR CLASS
# ============================================================================

class DeviceDetector:
    """Detect and identify connected devices"""
    
    def __init__(self):
        """Initialize device detector"""
        self.detected_devices: Dict[str, Dict[str, Any]] = {}
        self.last_detection_time: Optional[datetime] = None
        logger.info("✅ Device detector initialized")
    
    # ========================================================================
    # DEVICE DETECTION METHODS
    # ========================================================================
    
    def detect_all_devices(self) -> Dict[str, Dict[str, Any]]:
        """Detect all connected devices"""
        try:
            logger.info("🔍 Starting device detection...")
            self.detected_devices = {}
            
            # Detect Android devices
            android_devices = self.detect_android_devices()
            self.detected_devices.update(android_devices)
            
            # Detect iOS devices
            ios_devices = self.detect_ios_devices()
            self.detected_devices.update(ios_devices)
            
            # Detect storage devices
            storage_devices = self.detect_storage_devices()
            self.detected_devices.update(storage_devices)
            
            self.last_detection_time = datetime.now()
            logger.info(f"✅ Device detection complete: {len(self.detected_devices)} devices found")
            return self.detected_devices
        except Exception as e:
            logger.error(f"❌ Error detecting devices: {e}")
            return {}
    
    def detect_android_devices(self) -> Dict[str, Dict[str, Any]]:
        """Detect Android devices via ADB"""
        android_devices = {}
        try:
            logger.info("🔍 Scanning for Android devices...")
            
            # Run ADB devices command
            result = subprocess.run(
                ['adb', 'devices'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                for line in lines:
                    if line.strip() and '\t' in line:
                        device_id, status = line.split('\t')
                        device_id = device_id.strip()
                        status = status.strip()
                        
                        if status == 'device':
                            # ✅ Validate device_id
                            if not validate_device_id(device_id):
                                logger.warning(f"⚠️ Invalid device ID format: {device_id}")
                                continue
                            
                            device_info = self._get_android_device_info(device_id)
                            android_devices[device_id] = device_info
                            logger.info(f"✅ Android device found: {device_id}")
            
            return android_devices
        except FileNotFoundError:
            logger.warning("⚠️ ADB not found - Android device detection skipped")
            return {}
        except subprocess.TimeoutExpired:
            logger.error("❌ ADB command timeout - device detection failed")
            return {}
        except Exception as e:
            logger.error(f"❌ Error detecting Android devices: {e}", exc_info=True)
            return {}
    
    def detect_ios_devices(self) -> Dict[str, Dict[str, Any]]:
        """Detect iOS devices via iTunes/Xcode"""
        ios_devices = {}
        try:
            logger.info("🔍 Scanning for iOS devices...")
            
            # Platform-specific detection
            if platform.system() == 'Darwin':  # macOS
                result = subprocess.run(
                    ['system_profiler', 'SPUSBDataType'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                # ✅ Proper null check and logic
                if result.returncode == 0 and result.stdout:
                    if 'iPhone' in result.stdout or 'iPad' in result.stdout:
                        logger.info("✅ iOS device(s) detected")
                        # TODO: Parse detailed device info
                else:
                    logger.warning("⚠️ Could not get USB device info")
            
            return ios_devices
        except Exception as e:
            logger.warning(f"⚠️ Error detecting iOS devices: {e}")
            return {}
    
    def detect_storage_devices(self) -> Dict[str, Dict[str, Any]]:
        """Detect storage devices (HDD, USB)"""
        storage_devices = {}
        try:
            logger.info("🔍 Scanning for storage devices...")
            
            if platform.system() == 'Windows':
                # Windows: Check for mounted drives
                import string
                for drive in string.ascii_uppercase:
                    drive_path = f"{drive}:\\"
                    if os.path.exists(drive_path):
                        device_id = f"DRIVE_{drive}"
                        device_info = {
                            'device_id': device_id,
                            'device_type': 'HDD',
                            'path': drive_path,
                            'status': 'connected',
                            'capabilities': ['file_system', 'deleted_files', 'media'],
                            'timestamp': datetime.now().isoformat()
                        }
                        storage_devices[device_id] = device_info
                        logger.info(f"✅ Storage device found: {device_id}")
            
            elif platform.system() == 'Darwin':  # macOS
                result = subprocess.run(
                    ['diskutil', 'list'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                # TODO: Parse diskutil output
            
            elif platform.system() == 'Linux':
                result = subprocess.run(
                    ['lsblk', '-J'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                # TODO: Parse lsblk output
            
            return storage_devices
        except Exception as e:
            logger.warning(f"⚠️ Error detecting storage devices: {e}")
            return {}
    
    # ========================================================================
    # DEVICE INFORMATION METHODS
    # ========================================================================
    
    def _get_android_device_info(self, device_id: str) -> Dict[str, Any]:
        """Get detailed Android device information"""
        try:
            # Get device model
            result = subprocess.run(
                ['adb', '-s', device_id, 'shell', 'getprop', 'ro.product.model'],
                capture_output=True,
                text=True,
                timeout=5
            )
            model = result.stdout.strip() if result.returncode == 0 else 'Unknown'
            
            # Get battery level
            battery_level = 'N/A'
            try:
                result = subprocess.run(
                    ['adb', '-s', device_id, 'shell', 'dumpsys', 'battery'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if 'level:' in line:
                            battery_level = line.split(':')[1].strip() + '%'
                            break
            except:
                pass
            
            # Get storage info
            storage_total = 'N/A'
            try:
                result = subprocess.run(
                    ['adb', '-s', device_id, 'shell', 'df', '/data'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    if len(lines) > 1:
                        parts = lines[1].split()
                        if len(parts) >= 2:
                            total_kb = int(parts[1])
                            total_gb = total_kb / (1024 * 1024)
                            storage_total = f"{total_gb:.1f} GB"
            except:
                pass
            
            # Get Android version
            android_version = 'Unknown'
            try:
                result = subprocess.run(
                    ['adb', '-s', device_id, 'shell', 'getprop', 'ro.build.version.release'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    android_version = result.stdout.strip()
            except:
                pass
            
            return {
                'device_id': device_id,
                'device_type': 'Android',
                'model': model,
                'status': 'connected',
                'battery': battery_level,
                'storage': storage_total,
                'android_version': android_version,
                'capabilities': ['device_info', 'communications', 'media', 'location', 'apps', 'files'],
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"❌ Error getting Android device info: {e}")
            return {
                'device_id': device_id,
                'device_type': 'Android',
                'status': 'connected',
                'battery': 'N/A',
                'storage': 'N/A',
                'android_version': 'Unknown',
                'capabilities': []
            }
    
    def get_device_type(self, device_id: str) -> Optional[str]:
        """Get device type for given device ID"""
        if device_id in self.detected_devices:
            return self.detected_devices[device_id].get('device_type')
        return None
    
    def get_device_info(self, device_id: str) -> Optional[Dict[str, Any]]:
        """Get device information"""
        if device_id in self.detected_devices:
            return self.detected_devices[device_id]
        return None
    
    def validate_device(self, device_id: str) -> bool:
        """Validate that device exists and is accessible"""
        return device_id in self.detected_devices
    
    def list_available_devices(self) -> List[str]:
        """Get list of available device IDs"""
        return list(self.detected_devices.keys())
    
    def get_device_capabilities(self, device_id: str) -> List[str]:
        """Get capabilities for device"""
        if device_id in self.detected_devices:
            return self.detected_devices[device_id].get('capabilities', [])
        return []
    
    def get_detection_summary(self) -> Dict[str, Any]:
        """Get summary of detected devices"""
        return {
            'total_devices': len(self.detected_devices),
            'devices': list(self.detected_devices.keys()),
            'device_types': list(set(d.get('device_type') for d in self.detected_devices.values())),
            'last_detection': self.last_detection_time.isoformat() if self.last_detection_time else None,
            'timestamp': datetime.now().isoformat()
        }


# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

_device_detector_instance: Optional[DeviceDetector] = None


def get_device_detector() -> DeviceDetector:
    """Get global device detector instance"""
    global _device_detector_instance
    if _device_detector_instance is None:
        _device_detector_instance = DeviceDetector()
    return _device_detector_instance
