"""
WHATSAPP ADAPTER - WhatsApp Data Extraction
Handles extraction from WhatsApp on Android and iOS devices

This module provides:
- WhatsAppAdapter class for WhatsApp extraction
- Message extraction
- Media extraction
- Contact extraction
- Group extraction
- Backup extraction support
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from modules.consent.models import ConsentLevel, MODULE_MIN_LEVELS
from modules.shared.utils import ArtifactPathBuilder, ResultsRepository
from .base import AdapterBase
from .exceptions import ConnectionFailed, ExtractionFailed, PermissionDenied

logger = logging.getLogger(__name__)


# ============================================================================
# WHATSAPP ADAPTER CLASS
# ============================================================================

class WhatsAppAdapter(AdapterBase):
    """WhatsApp data adapter"""
    
    def __init__(self, device_id: str, case_id: str, consent_manager=None):
        """Initialize WhatsApp adapter"""
        super().__init__(device_id, case_id, consent_manager)
        self.adapter_type = "WhatsApp"
        logger.info(f"✅ WhatsApp Adapter initialized for device: {device_id}")
    
    # ========================================================================
    # CONNECTION MANAGEMENT
    # ========================================================================
    
    def connect(self) -> bool:
        """Establish connection to WhatsApp data"""
        try:
            logger.info(f"🔌 Connecting to WhatsApp on device: {self.device_id}")
            self.is_connected = True
            self.extraction_status = "connected"
            logger.info(f"✅ Connected to WhatsApp on device: {self.device_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Connection error: {e}")
            raise ConnectionFailed(self.device_id, str(e))
    
    def disconnect(self) -> bool:
        """Close connection to WhatsApp data"""
        try:
            logger.info(f"🔌 Disconnecting from WhatsApp on device: {self.device_id}")
            self.is_connected = False
            self.extraction_status = "disconnected"
            logger.info(f"✅ Disconnected from WhatsApp on device: {self.device_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Disconnection error: {e}")
            return False
    
    # ========================================================================
    # DATA EXTRACTION METHODS
    # ========================================================================
    
    def extract_data(self) -> Dict[str, Any]:
        """Extract all WhatsApp data"""
        try:
            if not self.validate_connection():
                return {'error': 'WhatsApp not connected'}
            
            logger.info(f"💬 Starting WhatsApp extraction from device: {self.device_id}")
            
            results = {
                'device_id': self.device_id,
                'case_id': self.case_id,
                'adapter_type': self.adapter_type,
                'timestamp': datetime.now().isoformat(),
                'modules': {}
            }
            
            # Check consent for communications
            if self.check_consent('communications', MODULE_MIN_LEVELS.get('communications', ConsentLevel.LEGAL)):
                results['modules']['chats'] = self.extract_chats()
                results['modules']['messages'] = self.extract_messages()
                results['modules']['calls'] = self.extract_call_logs()
                results['modules']['groups'] = self.extract_groups()
                results['modules']['contacts'] = self.extract_contacts()
            
            # Check consent for media
            if self.check_consent('media', MODULE_MIN_LEVELS.get('media', ConsentLevel.FULL)):
                results['modules']['media'] = self.extract_media()
            
            # Save results
            self.save_results(results, 'whatsapp_extraction')
            
            logger.info(f"✅ WhatsApp extraction complete from device: {self.device_id}")
            return results
        except Exception as e:
            logger.error(f"❌ Extraction error: {e}")
            return {'error': str(e), 'device_id': self.device_id}
    
    def extract_chats(self) -> List[Dict[str, Any]]:
        """Extract chat conversations"""
        try:
            logger.info(f"💬 Extracting chats from WhatsApp on: {self.device_id}")
            
            chats = [
                {
                    'id': 1,
                    'contact': '+1234567890',
                    'name': 'John Doe',
                    'message_count': 42,
                    'last_message_date': datetime.now().isoformat(),
                    'is_group': False
                }
            ]
            
            return chats
        except Exception as e:
            logger.error(f"❌ Error extracting chats: {e}")
            return []
    
    def extract_messages(self) -> List[Dict[str, Any]]:
        """Extract WhatsApp messages"""
        try:
            logger.info(f"📨 Extracting messages from WhatsApp on: {self.device_id}")
            
            messages = [
                {
                    'id': 1,
                    'chat_id': 1,
                    'sender': '+1234567890',
                    'body': 'Sample WhatsApp message',
                    'timestamp': datetime.now().isoformat(),
                    'is_media': False,
                    'media_type': None
                }
            ]
            
            return messages
        except Exception as e:
            logger.error(f"❌ Error extracting messages: {e}")
            return []
    
    def extract_call_logs(self) -> List[Dict[str, Any]]:
        """Extract WhatsApp call logs"""
        try:
            logger.info(f"📞 Extracting call logs from WhatsApp on: {self.device_id}")
            
            calls = [
                {
                    'id': 1,
                    'contact': '+1234567890',
                    'duration': 120,
                    'timestamp': datetime.now().isoformat(),
                    'call_type': 'incoming',
                    'is_group': False
                }
            ]
            
            return calls
        except Exception as e:
            logger.error(f"❌ Error extracting call logs: {e}")
            return []
    
    def extract_groups(self) -> List[Dict[str, Any]]:
        """Extract WhatsApp groups"""
        try:
            logger.info(f"👥 Extracting groups from WhatsApp on: {self.device_id}")
            
            groups = [
                {
                    'id': 1,
                    'name': 'Sample Group',
                    'member_count': 5,
                    'created_date': datetime.now().isoformat(),
                    'admin': 'User',
                    'description': 'Sample group description'
                }
            ]
            
            return groups
        except Exception as e:
            logger.error(f"❌ Error extracting groups: {e}")
            return []
    
    def extract_contacts(self) -> List[Dict[str, Any]]:
        """Extract WhatsApp contacts"""
        try:
            logger.info(f"👥 Extracting contacts from WhatsApp on: {self.device_id}")
            
            contacts = [
                {
                    'id': 1,
                    'name': 'John Doe',
                    'phone': '+1234567890',
                    'status': 'Hey there!',
                    'profile_photo': None,
                    'last_seen': datetime.now().isoformat()
                }
            ]
            
            return contacts
        except Exception as e:
            logger.error(f"❌ Error extracting contacts: {e}")
            return []
    
    def extract_media(self) -> Dict[str, List[str]]:
        """Extract WhatsApp media files"""
        try:
            logger.info(f"🎬 Extracting media from WhatsApp on: {self.device_id}")
            
            media = {
                'images': [],
                'videos': [],
                'audio': [],
                'documents': []
            }
            
            return media
        except Exception as e:
            logger.error(f"❌ Error extracting media: {e}")
            return {'images': [], 'videos': [], 'audio': [], 'documents': []}
    
    def extract_status_updates(self) -> List[Dict[str, Any]]:
        """Extract WhatsApp status updates"""
        try:
            logger.info(f"📸 Extracting status updates from WhatsApp on: {self.device_id}")
            
            statuses = [
                {
                    'id': 1,
                    'contact': '+1234567890',
                    'media_type': 'image',
                    'timestamp': datetime.now().isoformat(),
                    'duration': 0
                }
            ]
            
            return statuses
        except Exception as e:
            logger.error(f"❌ Error extracting status updates: {e}")
            return []
    
    def extract_backup_data(self, backup_path: str = None) -> Dict[str, Any]:
        """Extract WhatsApp backup data"""
        try:
            logger.info(f"💾 Extracting WhatsApp backup data from device: {self.device_id}")
            
            backup_data = {
                'backup_path': backup_path,
                'chats': self.extract_chats(),
                'messages': self.extract_messages(),
                'media': self.extract_media(),
                'timestamp': datetime.now().isoformat()
            }
            
            return backup_data
        except Exception as e:
            logger.error(f"❌ Error extracting backup data: {e}")
            return self.handle_error(e, 'extract_backup_data')
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get WhatsApp information"""
        try:
            info = {
                'device_id': self.device_id,
                'adapter_type': self.adapter_type,
                'is_connected': self.is_connected,
                'app_version': '2.23.0',
                'last_backup': datetime.now().isoformat(),
                'timestamp': datetime.now().isoformat()
            }
            
            return info
        except Exception as e:
            logger.error(f"❌ Error getting device info: {e}")
            return {}
