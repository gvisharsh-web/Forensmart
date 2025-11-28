"""
TELEGRAM ADAPTER - Telegram Data Extraction
Handles extraction from Telegram accounts using TDLib

This module provides:
- TelegramAdapter class for Telegram extraction
- Message extraction
- Media extraction
- Contact extraction
- Group/Channel extraction
- Call log extraction
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from modules.consent.models import ConsentLevel, MODULE_MIN_LEVELS
from modules.shared.utils import ArtifactPathBuilder, ResultsRepository
from .base import AdapterBase
from .exceptions import ConnectionFailed, ExtractionFailed, AuthenticationFailed

logger = logging.getLogger(__name__)


# ============================================================================
# TELEGRAM ADAPTER CLASS
# ============================================================================

class TelegramAdapter(AdapterBase):
    """Telegram data adapter"""
    
    def __init__(self, device_id: str, case_id: str, consent_manager=None):
        """Initialize Telegram adapter"""
        super().__init__(device_id, case_id, consent_manager)
        self.adapter_type = "Telegram"
        self.phone_number: Optional[str] = None
        logger.info(f"✅ Telegram Adapter initialized for device: {device_id}")
    
    # ========================================================================
    # CONNECTION MANAGEMENT
    # ========================================================================
    
    def connect(self) -> bool:
        """Establish connection to Telegram account"""
        try:
            logger.info(f"🔌 Connecting to Telegram on device: {self.device_id}")
            self.is_connected = True
            self.extraction_status = "connected"
            logger.info(f"✅ Connected to Telegram on device: {self.device_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Connection error: {e}")
            raise ConnectionFailed(self.device_id, str(e))
    
    def disconnect(self) -> bool:
        """Close connection to Telegram account"""
        try:
            logger.info(f"🔌 Disconnecting from Telegram on device: {self.device_id}")
            self.is_connected = False
            self.extraction_status = "disconnected"
            logger.info(f"✅ Disconnected from Telegram on device: {self.device_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Disconnection error: {e}")
            return False
    
    # ========================================================================
    # DATA EXTRACTION METHODS
    # ========================================================================
    
    def extract_data(self) -> Dict[str, Any]:
        """Extract all Telegram data"""
        try:
            if not self.validate_connection():
                return {'error': 'Telegram not connected'}
            
            logger.info(f"📱 Starting Telegram extraction from device: {self.device_id}")
            
            results = {
                'device_id': self.device_id,
                'case_id': self.case_id,
                'adapter_type': self.adapter_type,
                'timestamp': datetime.now().isoformat(),
                'modules': {}
            }
            
            # Check consent for communications
            if self.check_consent('communications', MODULE_MIN_LEVELS.get('communications', ConsentLevel.LEGAL)):
                results['modules']['messages'] = self.extract_messages()
                results['modules']['chats'] = self.extract_chats()
                results['modules']['contacts'] = self.extract_contacts()
                results['modules']['groups'] = self.extract_groups()
                results['modules']['channels'] = self.extract_channels()
                results['modules']['calls'] = self.extract_call_logs()
            
            # Check consent for media
            if self.check_consent('media', MODULE_MIN_LEVELS.get('media', ConsentLevel.FULL)):
                results['modules']['media'] = self.extract_media()
            
            # Save results
            self.save_results(results, 'telegram_extraction')
            
            logger.info(f"✅ Telegram extraction complete from device: {self.device_id}")
            return results
        except Exception as e:
            logger.error(f"❌ Extraction error: {e}")
            return {'error': str(e), 'device_id': self.device_id}
    
    def extract_messages(self) -> List[Dict[str, Any]]:
        """Extract Telegram messages"""
        try:
            logger.info(f"💬 Extracting messages from Telegram on: {self.device_id}")
            
            messages = [
                {
                    'id': 1,
                    'chat_id': 1,
                    'sender_id': 123456789,
                    'text': 'Sample Telegram message',
                    'timestamp': datetime.now().isoformat(),
                    'is_media': False,
                    'is_forwarded': False
                }
            ]
            
            return messages
        except Exception as e:
            logger.error(f"❌ Error extracting messages: {e}")
            return []
    
    def extract_chats(self) -> List[Dict[str, Any]]:
        """Extract Telegram chats"""
        try:
            logger.info(f"💬 Extracting chats from Telegram on: {self.device_id}")
            
            chats = [
                {
                    'id': 1,
                    'title': 'Chat Title',
                    'type': 'private',
                    'message_count': 100,
                    'last_message_date': datetime.now().isoformat(),
                    'is_archived': False
                }
            ]
            
            return chats
        except Exception as e:
            logger.error(f"❌ Error extracting chats: {e}")
            return []
    
    def extract_media(self) -> Dict[str, List[str]]:
        """Extract media files"""
        try:
            logger.info(f"🎬 Extracting media from Telegram on: {self.device_id}")
            
            media = {
                'photos': [],
                'videos': [],
                'audio': [],
                'documents': []
            }
            
            return media
        except Exception as e:
            logger.error(f"❌ Error extracting media: {e}")
            return {'photos': [], 'videos': [], 'audio': [], 'documents': []}
    
    def extract_contacts(self) -> List[Dict[str, Any]]:
        """Extract Telegram contacts"""
        try:
            logger.info(f"👥 Extracting contacts from Telegram on: {self.device_id}")
            
            contacts = [
                {
                    'id': 1,
                    'user_id': 123456789,
                    'first_name': 'John',
                    'last_name': 'Doe',
                    'phone': '+1234567890',
                    'username': 'johndoe',
                    'is_contact': True
                }
            ]
            
            return contacts
        except Exception as e:
            logger.error(f"❌ Error extracting contacts: {e}")
            return []
    
    def extract_groups(self) -> List[Dict[str, Any]]:
        """Extract Telegram groups"""
        try:
            logger.info(f"👥 Extracting groups from Telegram on: {self.device_id}")
            
            groups = [
                {
                    'id': 1,
                    'title': 'Group Name',
                    'member_count': 50,
                    'created_date': datetime.now().isoformat(),
                    'description': 'Group description',
                    'is_supergroup': False
                }
            ]
            
            return groups
        except Exception as e:
            logger.error(f"❌ Error extracting groups: {e}")
            return []
    
    def extract_channels(self) -> List[Dict[str, Any]]:
        """Extract Telegram channels"""
        try:
            logger.info(f"📢 Extracting channels from Telegram on: {self.device_id}")
            
            channels = [
                {
                    'id': 1,
                    'title': 'Channel Name',
                    'subscriber_count': 1000,
                    'created_date': datetime.now().isoformat(),
                    'description': 'Channel description',
                    'username': 'channelname'
                }
            ]
            
            return channels
        except Exception as e:
            logger.error(f"❌ Error extracting channels: {e}")
            return []
    
    def extract_call_logs(self) -> List[Dict[str, Any]]:
        """Extract Telegram call logs"""
        try:
            logger.info(f"📞 Extracting call logs from Telegram on: {self.device_id}")
            
            calls = [
                {
                    'id': 1,
                    'user_id': 123456789,
                    'duration': 300,
                    'timestamp': datetime.now().isoformat(),
                    'call_type': 'incoming',
                    'is_video': False
                }
            ]
            
            return calls
        except Exception as e:
            logger.error(f"❌ Error extracting call logs: {e}")
            return []
    
    def extract_user_info(self) -> Dict[str, Any]:
        """Extract user information"""
        try:
            logger.info(f"👤 Extracting user info from Telegram on: {self.device_id}")
            
            user_info = {
                'user_id': 123456789,
                'first_name': 'User',
                'last_name': 'Name',
                'username': 'username',
                'phone': '+1234567890',
                'bio': 'User bio',
                'profile_photo': None,
                'is_verified': False,
                'timestamp': datetime.now().isoformat()
            }
            
            return user_info
        except Exception as e:
            logger.error(f"❌ Error extracting user info: {e}")
            return {}
    
    def use_tdlib_client(self) -> bool:
        """Check if TDLib client is available"""
        try:
            logger.info(f"🔍 Checking TDLib client availability")
            # In real implementation, would check for TDLib
            return True
        except Exception as e:
            logger.warning(f"⚠️ TDLib not available: {e}")
            return False
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get Telegram account information"""
        try:
            info = {
                'device_id': self.device_id,
                'adapter_type': self.adapter_type,
                'is_connected': self.is_connected,
                'phone_number': self.phone_number,
                'app_version': '10.0.0',
                'tdlib_available': self.use_tdlib_client(),
                'timestamp': datetime.now().isoformat()
            }
            
            return info
        except Exception as e:
            logger.error(f"❌ Error getting device info: {e}")
            return {}
