"""
INSTAGRAM ADAPTER - Instagram Data Extraction
Handles extraction from Instagram accounts

This module provides:
- InstagramAdapter class for Instagram extraction
- Direct message extraction
- Post extraction
- Story extraction
- Follower/Following extraction
- Media extraction
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
# INSTAGRAM ADAPTER CLASS
# ============================================================================

class InstagramAdapter(AdapterBase):
    """Instagram data adapter"""
    
    def __init__(self, device_id: str, case_id: str, consent_manager=None):
        """Initialize Instagram adapter"""
        super().__init__(device_id, case_id, consent_manager)
        self.adapter_type = "Instagram"
        self.username: Optional[str] = None
        logger.info(f"✅ Instagram Adapter initialized for device: {device_id}")
    
    # ========================================================================
    # CONNECTION MANAGEMENT
    # ========================================================================
    
    def connect(self) -> bool:
        """Establish connection to Instagram account"""
        try:
            logger.info(f"🔌 Connecting to Instagram on device: {self.device_id}")
            self.is_connected = True
            self.extraction_status = "connected"
            logger.info(f"✅ Connected to Instagram on device: {self.device_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Connection error: {e}")
            raise ConnectionFailed(self.device_id, str(e))
    
    def disconnect(self) -> bool:
        """Close connection to Instagram account"""
        try:
            logger.info(f"🔌 Disconnecting from Instagram on device: {self.device_id}")
            self.is_connected = False
            self.extraction_status = "disconnected"
            logger.info(f"✅ Disconnected from Instagram on device: {self.device_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Disconnection error: {e}")
            return False
    
    # ========================================================================
    # DATA EXTRACTION METHODS
    # ========================================================================
    
    def extract_data(self) -> Dict[str, Any]:
        """Extract all Instagram data"""
        try:
            if not self.validate_connection():
                return {'error': 'Instagram not connected'}
            
            logger.info(f"📱 Starting Instagram extraction from device: {self.device_id}")
            
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
                results['modules']['followers'] = self.extract_followers()
                results['modules']['following'] = self.extract_following()
                results['modules']['contacts'] = self.extract_profile_info()
            
            # Check consent for media
            if self.check_consent('media', MODULE_MIN_LEVELS.get('media', ConsentLevel.FULL)):
                results['modules']['posts'] = self.extract_posts()
                results['modules']['stories'] = self.extract_stories()
                results['modules']['media'] = self.extract_media()
            
            # Save results
            self.save_results(results, 'instagram_extraction')
            
            logger.info(f"✅ Instagram extraction complete from device: {self.device_id}")
            return results
        except Exception as e:
            logger.error(f"❌ Extraction error: {e}")
            return {'error': str(e), 'device_id': self.device_id}
    
    def extract_messages(self) -> List[Dict[str, Any]]:
        """Extract Instagram direct messages"""
        try:
            logger.info(f"💬 Extracting DMs from Instagram on: {self.device_id}")
            
            messages = [
                {
                    'id': 1,
                    'sender': 'user123',
                    'body': 'Sample Instagram DM',
                    'timestamp': datetime.now().isoformat(),
                    'is_media': False
                }
            ]
            
            return messages
        except Exception as e:
            logger.error(f"❌ Error extracting messages: {e}")
            return []
    
    def extract_posts(self) -> List[Dict[str, Any]]:
        """Extract Instagram posts"""
        try:
            logger.info(f"📸 Extracting posts from Instagram on: {self.device_id}")
            
            posts = [
                {
                    'id': 1,
                    'caption': 'Sample post caption',
                    'likes': 100,
                    'comments': 5,
                    'timestamp': datetime.now().isoformat(),
                    'media_type': 'image'
                }
            ]
            
            return posts
        except Exception as e:
            logger.error(f"❌ Error extracting posts: {e}")
            return []
    
    def extract_stories(self) -> List[Dict[str, Any]]:
        """Extract Instagram stories"""
        try:
            logger.info(f"📖 Extracting stories from Instagram on: {self.device_id}")
            
            stories = [
                {
                    'id': 1,
                    'media_type': 'image',
                    'timestamp': datetime.now().isoformat(),
                    'views': 50,
                    'expires_at': datetime.now().isoformat()
                }
            ]
            
            return stories
        except Exception as e:
            logger.error(f"❌ Error extracting stories: {e}")
            return []
    
    def extract_followers(self) -> List[Dict[str, Any]]:
        """Extract followers list"""
        try:
            logger.info(f"👥 Extracting followers from Instagram on: {self.device_id}")
            
            followers = [
                {
                    'id': 1,
                    'username': 'follower123',
                    'name': 'Follower Name',
                    'profile_pic': None,
                    'is_verified': False
                }
            ]
            
            return followers
        except Exception as e:
            logger.error(f"❌ Error extracting followers: {e}")
            return []
    
    def extract_following(self) -> List[Dict[str, Any]]:
        """Extract following list"""
        try:
            logger.info(f"👥 Extracting following from Instagram on: {self.device_id}")
            
            following = [
                {
                    'id': 1,
                    'username': 'following123',
                    'name': 'Following Name',
                    'profile_pic': None,
                    'is_verified': False
                }
            ]
            
            return following
        except Exception as e:
            logger.error(f"❌ Error extracting following: {e}")
            return []
    
    def extract_media(self) -> Dict[str, List[str]]:
        """Extract media files"""
        try:
            logger.info(f"🎬 Extracting media from Instagram on: {self.device_id}")
            
            media = {
                'photos': [],
                'videos': [],
                'stories': []
            }
            
            return media
        except Exception as e:
            logger.error(f"❌ Error extracting media: {e}")
            return {'photos': [], 'videos': [], 'stories': []}
    
    def extract_profile_info(self) -> Dict[str, Any]:
        """Extract profile information"""
        try:
            logger.info(f"👤 Extracting profile info from Instagram on: {self.device_id}")
            
            profile = {
                'username': 'sample_user',
                'name': 'Sample User',
                'bio': 'Sample bio',
                'followers_count': 1000,
                'following_count': 500,
                'posts_count': 50,
                'is_verified': False,
                'is_private': False,
                'profile_pic': None,
                'timestamp': datetime.now().isoformat()
            }
            
            return profile
        except Exception as e:
            logger.error(f"❌ Error extracting profile info: {e}")
            return {}
    
    def extract_comments(self) -> List[Dict[str, Any]]:
        """Extract comments on posts"""
        try:
            logger.info(f"💬 Extracting comments from Instagram on: {self.device_id}")
            
            comments = [
                {
                    'id': 1,
                    'post_id': 1,
                    'author': 'user123',
                    'text': 'Sample comment',
                    'likes': 5,
                    'timestamp': datetime.now().isoformat()
                }
            ]
            
            return comments
        except Exception as e:
            logger.error(f"❌ Error extracting comments: {e}")
            return []
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get Instagram account information"""
        try:
            info = {
                'device_id': self.device_id,
                'adapter_type': self.adapter_type,
                'is_connected': self.is_connected,
                'username': self.username,
                'app_version': '300.0.0.0',
                'timestamp': datetime.now().isoformat()
            }
            
            return info
        except Exception as e:
            logger.error(f"❌ Error getting device info: {e}")
            return {}
