"""
EXTRACTION MODULE - Data Extractors
Handles extraction of different data types from devices

This module provides:
- DeviceInfoExtractor (device information)
- CommunicationExtractor (SMS, calls, contacts)
- LocationExtractor (GPS, cell towers)
- SecurityExtractor (passwords, authentication)
- MediaExtractor (photos, videos, audio)
- SystemExtractor (system logs, configuration)
"""

import os
import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from modules.shared.utils import ErrorHandlingLoopholes, get_cache_manager, ArtifactPathBuilder, ResultsRepository

# ============================================================================
# LOGGING SETUP
# ============================================================================

logger = logging.getLogger(__name__)

# ============================================================================
# BASE EXTRACTION MODULE CLASS
# ============================================================================

class ExtractionModule(ABC):
    """Base class for extraction modules"""

    def __init__(self, name: str, description: str):
        """Initialize extraction module"""
        self.name = name
        self.description = description
        self.extraction_time = None
        self.artifact_count = 0

    @abstractmethod
    def extract(self, device_id: str, **kwargs) -> Dict[str, Any]:
        """Extract data from device"""
        pass

    def get_info(self) -> Dict[str, str]:
        """Get module information"""
        return {
            'name': self.name,
            'description': self.description
        }
    
    # ========================================================================
    # ARTIFACT ROUTING
    # ========================================================================
    
    def save_extraction_results(self, case_id: str, results: Dict[str, Any]) -> bool:
        """Save extraction results to artifact storage"""
        try:
            # Resolve artifact path
            artifact_path = ArtifactPathBuilder.resolve(
                case_id, 
                "extraction", 
                ensure_dir=True
            )
            
            # Save by module name
            module_file = os.path.join(artifact_path, f"{self.name.lower().replace(' ', '_')}.json")
            
            with open(module_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"✅ {self.name} extraction saved to {module_file}")
            
            # Also save to results repository
            ResultsRepository.save(case_id, {self.name: results})
            
            return True
        except Exception as e:
            logger.error(f"❌ Error saving {self.name} extraction: {e}")
            return False
    
    def load_extraction_results(self, case_id: str) -> Optional[Dict[str, Any]]:
        """Load extraction results from artifact storage"""
        try:
            artifact_path = ArtifactPathBuilder.resolve(case_id, "extraction")
            module_file = os.path.join(artifact_path, f"{self.name.lower().replace(' ', '_')}.json")
            
            if os.path.exists(module_file):
                with open(module_file, 'r') as f:
                    results = json.load(f)
                
                logger.info(f"✅ {self.name} extraction loaded from {module_file}")
                return results
            
            return None
        except Exception as e:
            logger.error(f"❌ Error loading {self.name} extraction: {e}")
            return None


# ============================================================================
# DEVICE INFO EXTRACTOR
# ============================================================================

class DeviceInfoExtractor(ExtractionModule):
    """Extracts basic device information"""

    def __init__(self):
        super().__init__(
            "Device Information",
            "Basic device identification and hardware specs"
        )

    def extract(self, device_id: str, **kwargs) -> Dict[str, Any]:
        """Extract device information with error handling"""
        
        def _extract():
            start_time = datetime.now()
            
            # Check cache first
            cache_manager = get_cache_manager()
            cache_key = f"device_info_{device_id}"
            cached_data = cache_manager.get(cache_key)
            
            if cached_data:
                logger.info(f"Device info from cache: {device_id}")
                return cached_data
            
            # Simulate device info extraction
            device_info = {
                'device_id': device_id,
                'model': 'Samsung Galaxy S21',
                'manufacturer': 'Samsung',
                'os': 'Android 12',
                'os_version': '12.0.1',
                'imei': '123456789012345',
                'serial_number': 'RF8M70NRXXX',
                'phone_number': '+1-555-0123',
                'storage_total': '128GB',
                'storage_used': '85GB',
                'ram': '8GB',
                'cpu': 'Snapdragon 888',
                'battery_health': '85%',
                'screen_resolution': '1440x3200',
                'extraction_time': datetime.now().isoformat()
            }
            
            self.extraction_time = (datetime.now() - start_time).total_seconds()
            self.artifact_count = 1
            
            # Cache the result
            cache_manager.set(cache_key, device_info)
            
            logger.info(f"Device info extracted: {device_id}")
            
            return {
                'status': 'success',
                'data': device_info,
                'artifact_count': self.artifact_count,
                'extraction_time': self.extraction_time
            }
        
        # Use error handling loophole with retry
        result = ErrorHandlingLoopholes.auto_retry_on_error(
            _extract,
            max_attempts=3,
            delay=1.0,
            backoff=2.0
        )
        
        if result is None:
            return {
                'status': 'error',
                'error': 'Device info extraction failed after retries'
            }
        
        return result


# ============================================================================
# COMMUNICATION EXTRACTOR
# ============================================================================

class CommunicationExtractor(ExtractionModule):
    """Extracts communication data (SMS, calls, contacts)"""

    def __init__(self):
        super().__init__(
            "Communications",
            "SMS, call logs, contacts, and messaging data"
        )

    def extract(self, device_id: str, **kwargs) -> Dict[str, Any]:
        """Extract communication data with consent checks"""
        try:
            start_time = datetime.now()
            
            # Get consent manager from kwargs
            consent_manager = kwargs.get('consent_manager')
            case_id = kwargs.get('case_id')
            
            # Check consent if available
            if consent_manager and case_id:
                from modules.consent.models import ConsentLevel, MODULE_MIN_LEVELS
                
                session = consent_manager.get_session(case_id)
                if session:
                    min_level = MODULE_MIN_LEVELS.get('communications', ConsentLevel.LEGAL)
                    
                    if session.level < min_level:
                        logger.warning(f"Communications blocked: {session.level.name} < {min_level.name}")
                        return {
                            'status': 'consent_denied',
                            'message': f'Communications requires {min_level.name} consent',
                            'required_level': min_level.name,
                            'current_level': session.level.name
                        }
            
            # Simulate communication data extraction
            communications = {
                'sms_count': 245,
                'call_count': 89,
                'contact_count': 156,
                'sms': [
                    {
                        'id': 1,
                        'sender': '+1-555-0100',
                        'message': 'Hey, how are you?',
                        'timestamp': '2025-11-25 10:30:00',
                        'type': 'received'
                    },
                    {
                        'id': 2,
                        'sender': '+1-555-0101',
                        'message': 'Meeting at 3 PM',
                        'timestamp': '2025-11-25 11:15:00',
                        'type': 'received'
                    }
                ],
                'calls': [
                    {
                        'id': 1,
                        'number': '+1-555-0100',
                        'duration': 300,
                        'timestamp': '2025-11-25 09:45:00',
                        'type': 'incoming'
                    },
                    {
                        'id': 2,
                        'number': '+1-555-0102',
                        'duration': 120,
                        'timestamp': '2025-11-25 14:20:00',
                        'type': 'outgoing'
                    }
                ],
                'contacts': [
                    {
                        'id': 1,
                        'name': 'John Doe',
                        'phone': '+1-555-0100',
                        'email': 'john@example.com'
                    },
                    {
                        'id': 2,
                        'name': 'Jane Smith',
                        'phone': '+1-555-0101',
                        'email': 'jane@example.com'
                    }
                ],
                'extraction_time': datetime.now().isoformat()
            }
            
            self.extraction_time = (datetime.now() - start_time).total_seconds()
            self.artifact_count = len(communications['sms']) + len(communications['calls'])
            
            logger.info(f"Communications extracted: {device_id}")
            
            return {
                'status': 'success',
                'data': communications,
                'artifact_count': self.artifact_count,
                'extraction_time': self.extraction_time
            }
        
        except Exception as e:
            logger.error(f"Communications extraction failed: {e}", exc_info=True)
            return {
                'status': 'error',
                'error': str(e)
            }


# ============================================================================
# LOCATION EXTRACTOR
# ============================================================================

class LocationExtractor(ExtractionModule):
    """Extracts location and GPS data"""

    def __init__(self):
        super().__init__(
            "Location",
            "GPS coordinates, cell tower data, and location history"
        )

    def extract(self, device_id: str, **kwargs) -> Dict[str, Any]:
        """Extract location data with consent checks"""
        try:
            start_time = datetime.now()
            
            # Get consent manager from kwargs
            consent_manager = kwargs.get('consent_manager')
            case_id = kwargs.get('case_id')
            
            # Check consent if available
            if consent_manager and case_id:
                from modules.consent.models import ConsentLevel, MODULE_MIN_LEVELS
                
                session = consent_manager.get_session(case_id)
                if session:
                    min_level = MODULE_MIN_LEVELS.get('location', ConsentLevel.STANDARD)
                    
                    if session.level < min_level:
                        logger.warning(f"Location blocked: {session.level.name} < {min_level.name}")
                        return {
                            'status': 'consent_denied',
                            'message': f'Location requires {min_level.name} consent',
                            'required_level': min_level.name,
                            'current_level': session.level.name
                        }
            
            # Simulate location data extraction
            location_data = {
                'gps_points': 156,
                'cell_towers': 42,
                'gps': [
                    {
                        'id': 1,
                        'latitude': 40.7128,
                        'longitude': -74.0060,
                        'accuracy': 10,
                        'timestamp': '2025-11-25 08:00:00',
                        'location': 'New York, NY'
                    },
                    {
                        'id': 2,
                        'latitude': 40.7580,
                        'longitude': -73.9855,
                        'accuracy': 15,
                        'timestamp': '2025-11-25 09:30:00',
                        'location': 'Times Square, NY'
                    }
                ],
                'cell_towers': [
                    {
                        'id': 1,
                        'mcc': 310,
                        'mnc': 410,
                        'lac': 1234,
                        'cell_id': 5678,
                        'timestamp': '2025-11-25 08:15:00'
                    }
                ],
                'extraction_time': datetime.now().isoformat()
            }
            
            self.extraction_time = (datetime.now() - start_time).total_seconds()
            self.artifact_count = len(location_data['gps']) + len(location_data['cell_towers'])
            
            logger.info(f"Location data extracted: {device_id}")
            
            return {
                'status': 'success',
                'data': location_data,
                'artifact_count': self.artifact_count,
                'extraction_time': self.extraction_time
            }
        
        except Exception as e:
            logger.error(f"Location extraction failed: {e}", exc_info=True)
            return {
                'status': 'error',
                'error': str(e)
            }


# ============================================================================
# SECURITY EXTRACTOR
# ============================================================================

class SecurityExtractor(ExtractionModule):
    """Extracts security-related data"""

    def __init__(self):
        super().__init__(
            "Security",
            "Password strength, authentication methods, and security settings"
        )

    def extract(self, device_id: str, **kwargs) -> Dict[str, Any]:
        """Extract security data with consent checks"""
        try:
            start_time = datetime.now()
            
            # Get consent manager from kwargs
            consent_manager = kwargs.get('consent_manager')
            case_id = kwargs.get('case_id')
            
            # Check consent if available
            if consent_manager and case_id:
                from modules.consent.models import ConsentLevel, MODULE_MIN_LEVELS
                
                session = consent_manager.get_session(case_id)
                if session:
                    min_level = MODULE_MIN_LEVELS.get('security', ConsentLevel.STANDARD)
                    
                    if session.level < min_level:
                        logger.warning(f"Security blocked: {session.level.name} < {min_level.name}")
                        return {
                            'status': 'consent_denied',
                            'message': f'Security requires {min_level.name} consent',
                            'required_level': min_level.name,
                            'current_level': session.level.name
                        }
            
            # Simulate security data extraction
            security_data = {
                'lock_type': 'PIN',
                'pin_length': 6,
                'biometric_enabled': True,
                'fingerprint_count': 2,
                'face_recognition': True,
                'encryption_enabled': True,
                'security_apps': [
                    {'name': 'Google Play Protect', 'status': 'enabled'},
                    {'name': 'Norton Mobile Security', 'status': 'enabled'}
                ],
                'last_security_update': '2025-11-20',
                'security_patch_level': 'November 2025',
                'extraction_time': datetime.now().isoformat()
            }
            
            self.extraction_time = (datetime.now() - start_time).total_seconds()
            self.artifact_count = 1
            
            logger.info(f"Security data extracted: {device_id}")
            
            return {
                'status': 'success',
                'data': security_data,
                'artifact_count': self.artifact_count,
                'extraction_time': self.extraction_time
            }
        
        except Exception as e:
            logger.error(f"Security extraction failed: {e}", exc_info=True)
            return {
                'status': 'error',
                'error': str(e)
            }


# ============================================================================
# MEDIA EXTRACTOR
# ============================================================================

class MediaExtractor(ExtractionModule):
    """Extracts media files and content"""

    def __init__(self):
        super().__init__(
            "Media",
            "Photos, videos, audio files, and thumbnails"
        )

    def extract(self, device_id: str, **kwargs) -> Dict[str, Any]:
        """Extract media data with consent checks"""
        try:
            start_time = datetime.now()
            
            # Get consent manager from kwargs
            consent_manager = kwargs.get('consent_manager')
            case_id = kwargs.get('case_id')
            
            # Check consent if available
            if consent_manager and case_id:
                from modules.consent.models import ConsentLevel, MODULE_MIN_LEVELS
                
                session = consent_manager.get_session(case_id)
                if session:
                    min_level = MODULE_MIN_LEVELS.get('media', ConsentLevel.STANDARD)
                    
                    if session.level < min_level:
                        logger.warning(f"Media blocked: {session.level.name} < {min_level.name}")
                        return {
                            'status': 'consent_denied',
                            'message': f'Media requires {min_level.name} consent',
                            'required_level': min_level.name,
                            'current_level': session.level.name
                        }
            
            # Simulate media data extraction
            media_data = {
                'photos': 342,
                'videos': 28,
                'audio': 156,
                'photos_list': [
                    {
                        'id': 1,
                        'filename': 'IMG_20251125_100530.jpg',
                        'size': 2048576,
                        'timestamp': '2025-11-25 10:05:30',
                        'location': 'DCIM/Camera'
                    },
                    {
                        'id': 2,
                        'filename': 'IMG_20251125_120045.jpg',
                        'size': 1856432,
                        'timestamp': '2025-11-25 12:00:45',
                        'location': 'DCIM/Camera'
                    }
                ],
                'videos_list': [
                    {
                        'id': 1,
                        'filename': 'VID_20251125_143020.mp4',
                        'size': 52428800,
                        'duration': 120,
                        'timestamp': '2025-11-25 14:30:20',
                        'location': 'DCIM/Camera'
                    }
                ],
                'extraction_time': datetime.now().isoformat()
            }
            
            self.extraction_time = (datetime.now() - start_time).total_seconds()
            self.artifact_count = media_data['photos'] + media_data['videos'] + media_data['audio']
            
            logger.info(f"Media data extracted: {device_id}")
            
            return {
                'status': 'success',
                'data': media_data,
                'artifact_count': self.artifact_count,
                'extraction_time': self.extraction_time
            }
        
        except Exception as e:
            logger.error(f"Media extraction failed: {e}", exc_info=True)
            return {
                'status': 'error',
                'error': str(e)
            }


# ============================================================================
# SYSTEM EXTRACTOR
# ============================================================================

class SystemExtractor(ExtractionModule):
    """Extracts system-level data"""

    def __init__(self):
        super().__init__(
            "System",
            "System logs, configuration, and diagnostic information"
        )

    def extract(self, device_id: str, **kwargs) -> Dict[str, Any]:
        """Extract system data with consent checks"""
        try:
            start_time = datetime.now()
            
            # Get consent manager from kwargs
            consent_manager = kwargs.get('consent_manager')
            case_id = kwargs.get('case_id')
            
            # Check consent if available
            if consent_manager and case_id:
                from modules.consent.models import ConsentLevel, MODULE_MIN_LEVELS
                
                session = consent_manager.get_session(case_id)
                if session:
                    min_level = MODULE_MIN_LEVELS.get('system', ConsentLevel.FULL)
                    
                    if session.level < min_level:
                        logger.warning(f"System blocked: {session.level.name} < {min_level.name}")
                        return {
                            'status': 'consent_denied',
                            'message': f'System requires {min_level.name} consent',
                            'required_level': min_level.name,
                            'current_level': session.level.name
                        }
            
            # Simulate system data extraction
            system_data = {
                'system_logs': 5432,
                'installed_apps': 187,
                'system_services': 89,
                'system_logs_sample': [
                    {
                        'id': 1,
                        'timestamp': '2025-11-25 08:00:00',
                        'level': 'INFO',
                        'message': 'System boot completed'
                    },
                    {
                        'id': 2,
                        'timestamp': '2025-11-25 08:05:00',
                        'level': 'WARNING',
                        'message': 'Low memory warning'
                    }
                ],
                'installed_apps_sample': [
                    {'name': 'Gmail', 'package': 'com.google.android.gm', 'version': '2025.11.01'},
                    {'name': 'Chrome', 'package': 'com.android.chrome', 'version': '131.0.6778.0'}
                ],
                'extraction_time': datetime.now().isoformat()
            }
            
            self.extraction_time = (datetime.now() - start_time).total_seconds()
            self.artifact_count = system_data['system_logs'] + system_data['installed_apps']
            
            logger.info(f"System data extracted: {device_id}")
            
            return {
                'status': 'success',
                'data': system_data,
                'artifact_count': self.artifact_count,
                'extraction_time': self.extraction_time
            }
        
        except Exception as e:
            logger.error(f"System extraction failed: {e}", exc_info=True)
            return {
                'status': 'error',
                'error': str(e)
            }
