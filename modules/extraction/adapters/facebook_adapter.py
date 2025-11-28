"""
FACEBOOK ADAPTER - Facebook Data Extraction
Handles extraction from Facebook accounts and Messenger

This module provides:
- FacebookAdapter class for Facebook extraction
- Messenger extraction
- Post extraction
- Photo extraction
- Friend list extraction
- Profile information extraction
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
# FACEBOOK ADAPTER CLASS
# ============================================================================

class FacebookAdapter(AdapterBase):
    """Facebook data adapter"""
    
    def __init__(self, device_id: str, case_id: str, consent_manager=None):
        """Initialize Facebook adapter"""
        super().__init__(device_id, case_id, consent_manager)
        self.adapter_type = "Facebook"
        self.user_id: Optional[str] = None
        logger.info(f"✅ Facebook Adapter initialized for device: {device_id}")
    
    # ========================================================================
    # CONNECTION MANAGEMENT
    # ========================================================================
    
    def connect(self) -> bool:
        """Establish connection to Facebook account"""
        try:
            logger.info(f"🔌 Connecting to Facebook on device: {self.device_id}")
            self.is_connected = True
            self.extraction_status = "connected"
            logger.info(f"✅ Connected to Facebook on device: {self.device_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Connection error: {e}")
            raise ConnectionFailed(self.device_id, str(e))
    
    def disconnect(self) -> bool:
        """Close connection to Facebook account"""
        try:
            logger.info(f"🔌 Disconnecting from Facebook on device: {self.device_id}")
            self.is_connected = False
            self.extraction_status = "disconnected"
            logger.info(f"✅ Disconnected from Facebook on device: {self.device_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Disconnection error: {e}")
            return False
    
    # ========================================================================
    # DATA EXTRACTION METHODS
    # ========================================================================
    
    def extract_data(self) -> Dict[str, Any]:
        """Extract all Facebook data"""
        try:
            if not self.validate_connection():
                return {'error': 'Facebook not connected'}
            
            logger.info(f"📱 Starting Facebook extraction from device: {self.device_id}")
            
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
                results['modules']['friends'] = self.extract_friends_list()
                results['modules']['profile'] = self.extract_profile_info()
            
            # Check consent for media
            if self.check_consent('media', MODULE_MIN_LEVELS.get('media', ConsentLevel.FULL)):
                results['modules']['posts'] = self.extract_posts()
                results['modules']['photos'] = self.extract_photos()
                results['modules']['videos'] = self.extract_videos()
            
            # Save results
            self.save_results(results, 'facebook_extraction')
            
            logger.info(f"✅ Facebook extraction complete from device: {self.device_id}")
            return results
        except Exception as e:
            logger.error(f"❌ Extraction error: {e}")
            return {'error': str(e), 'device_id': self.device_id}
    
    def extract_messages(self) -> List[Dict[str, Any]]:
        """Extract Facebook Messenger messages"""
        try:
            logger.info(f"💬 Extracting messages from Facebook on: {self.device_id}")
            
            messages = [
                {
                    'id': 1,
                    'thread_id': 1,
                    'sender_id': 123456789,
                    'body': 'Sample Facebook message',
                    'timestamp': datetime.now().isoformat(),
                    'is_media': False
                }
            ]
            
            return messages
        except Exception as e:
            logger.error(f"❌ Error extracting messages: {e}")
            return []
    
    def extract_posts(self) -> List[Dict[str, Any]]:
        """Extract Facebook posts"""
        try:
            logger.info(f"📝 Extracting posts from Facebook on: {self.device_id}")
            
            posts = [
                {
                    'id': 1,
                    'message': 'Sample post',
                    'likes': 50,
                    'comments': 10,
                    'shares': 5,
                    'timestamp': datetime.now().isoformat(),
                    'privacy': 'PUBLIC'
                }
            ]
            
            return posts
        except Exception as e:
            logger.error(f"❌ Error extracting posts: {e}")
            return []
    
    def extract_photos(self) -> List[Dict[str, Any]]:
        """Extract photos"""
        try:
            logger.info(f"📸 Extracting photos from Facebook on: {self.device_id}")
            
            photos = [
                {
                    'id': 1,
                    'album_id': 1,
                    'caption': 'Sample photo',
                    'timestamp': datetime.now().isoformat(),
                    'likes': 20,
                    'comments': 5
                }
            ]
            
            return photos
        except Exception as e:
            logger.error(f"❌ Error extracting photos: {e}")
            return []
    
    def extract_videos(self) -> List[Dict[str, Any]]:
        """Extract videos"""
        try:
            logger.info(f"🎬 Extracting videos from Facebook on: {self.device_id}")
            
            videos = [
                {
                    'id': 1,
                    'title': 'Sample video',
                    'description': 'Video description',
                    'duration': 120,
                    'timestamp': datetime.now().isoformat(),
                    'likes': 30,
                    'comments': 8
                }
            ]
            
            return videos
        except Exception as e:
            logger.error(f"❌ Error extracting videos: {e}")
            return []
    
    def extract_friends_list(self) -> List[Dict[str, Any]]:
        """Extract friends list"""
        try:
            logger.info(f"👥 Extracting friends from Facebook on: {self.device_id}")
            
            friends = [
                {
                    'id': 1,
                    'user_id': 987654321,
                    'name': 'Friend Name',
                    'profile_url': 'https://facebook.com/friend',
                    'added_date': datetime.now().isoformat()
                }
            ]
            
            return friends
        except Exception as e:
            logger.error(f"❌ Error extracting friends: {e}")
            return []
    
    def extract_profile_info(self) -> Dict[str, Any]:
        """Extract profile information"""
        try:
            logger.info(f"👤 Extracting profile info from Facebook on: {self.device_id}")
            
            profile = {
                'user_id': 123456789,
                'name': 'User Name',
                'email': 'user@example.com',
                'phone': '+1234567890',
                'bio': 'User bio',
                'location': 'City, Country',
                'profile_picture': None,
                'cover_photo': None,
                'friends_count': 500,
                'followers_count': 1000,
                'timestamp': datetime.now().isoformat()
            }
            
            return profile
        except Exception as e:
            logger.error(f"❌ Error extracting profile info: {e}")
            return {}
    
    def extract_comments(self) -> List[Dict[str, Any]]:
        """Extract comments"""
        try:
            logger.info(f"💬 Extracting comments from Facebook on: {self.device_id}")
            
            comments = [
                {
                    'id': 1,
                    'post_id': 1,
                    'author_id': 987654321,
                    'message': 'Sample comment',
                    'likes': 5,
                    'timestamp': datetime.now().isoformat()
                }
            ]
            
            return comments
        except Exception as e:
            logger.error(f"❌ Error extracting comments: {e}")
            return []
    
    def extract_events(self) -> List[Dict[str, Any]]:
        """Extract events"""
        try:
            logger.info(f"📅 Extracting events from Facebook on: {self.device_id}")
            
            events = [
                {
                    'id': 1,
                    'name': 'Event Name',
                    'description': 'Event description',
                    'start_time': datetime.now().isoformat(),
                    'end_time': datetime.now().isoformat(),
                    'location': 'Event location',
                    'attending': 100
                }
            ]
            
            return events
        except Exception as e:
            logger.error(f"❌ Error extracting events: {e}")
            return []
    
    def extract_groups(self) -> List[Dict[str, Any]]:
        """Extract groups"""
        try:
            logger.info(f"👥 Extracting groups from Facebook on: {self.device_id}")
            
            groups = [
                {
                    'id': 1,
                    'name': 'Group Name',
                    'description': 'Group description',
                    'member_count': 500,
                    'privacy': 'PUBLIC',
                    'created_date': datetime.now().isoformat()
                }
            ]
            
            return groups
        except Exception as e:
            logger.error(f"❌ Error extracting groups: {e}")
            return []
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get Facebook account information"""
        try:
            info = {
                'device_id': self.device_id,
                'adapter_type': self.adapter_type,
                'is_connected': self.is_connected,
                'user_id': self.user_id,
                'app_version': '400.0.0.0',
                'timestamp': datetime.now().isoformat()
            }
            
            return info
        except Exception as e:
            logger.error(f"❌ Error getting device info: {e}")
            return {}
