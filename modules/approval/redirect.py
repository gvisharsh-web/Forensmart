"""Automatic redirect and extraction trigger after approval."""
from __future__ import annotations

import json
import logging
import time
from typing import Dict, Any, Optional, Callable
from pathlib import Path
from datetime import datetime
from urllib.parse import urlencode, quote

logger = logging.getLogger(__name__)

# NEW: Import ApprovalSync for approval verification
try:
    from modules.approval.sync import ApprovalSync
except ImportError:
    ApprovalSync = None  # Optional dependency


class ApprovalRedirect:
    """Handle approval redirects and auto-extraction triggers."""
    
    # Callbacks for extraction triggers
    _extraction_callbacks: Dict[str, Callable] = {}
    
    @staticmethod
    def register_extraction_callback(case_id: str, callback: Callable) -> None:
        """Register a callback to trigger extraction when approval is received."""
        ApprovalRedirect._extraction_callbacks[case_id] = callback
        logger.info(f"Registered extraction callback for {case_id}")
    
    @staticmethod
    def trigger_extraction(case_id: str, device_id: str, extraction_type: str = "android") -> bool:
        """Trigger extraction for a case after approval."""
        try:
            # NEW: Check approval status with ApprovalSync
            if ApprovalSync:
                if not ApprovalSync.is_approved(case_id):
                    logger.warning(f"Extraction not approved for {case_id}")
                    return False
            
            # Check if callback is registered
            if case_id in ApprovalRedirect._extraction_callbacks:
                callback = ApprovalRedirect._extraction_callbacks[case_id]
                callback(case_id, device_id, extraction_type)
                logger.info(f"Triggered extraction for {case_id}")
                return True
            else:
                logger.warning(f"No extraction callback registered for {case_id}")
                return False
        except Exception as e:
            logger.error(f"Failed to trigger extraction: {e}")
            return False
    
    @staticmethod
    def create_redirect_link(
        base_url: str,
        case_id: str,
        device_id: str,
        redirect_to: str = "extraction",
        extraction_type: str = "android"
    ) -> str:
        """Create a redirect link that triggers extraction after approval."""
        try:
            params = {
                "case_id": case_id,
                "device_id": device_id,
                "redirect_to": redirect_to,
                "extraction_type": extraction_type,
                "timestamp": datetime.now().isoformat()
            }
            
            query_string = urlencode(params)
            redirect_link = f"{base_url}?{query_string}"
            logger.info(f"Created redirect link for {case_id}")
            return redirect_link
        except Exception as e:
            logger.error(f"Failed to create redirect link: {e}")
            return ""
    
    @staticmethod
    def save_redirect_config(
        case_id: str,
        config: Dict[str, Any]
    ) -> bool:
        """Save redirect configuration for a case."""
        try:
            redirect_dir = Path("audit/redirects")
            redirect_dir.mkdir(parents=True, exist_ok=True)
            
            config_file = redirect_dir / f"{case_id}_redirect.json"
            config['created_at'] = datetime.now().isoformat()
            config['status'] = 'pending'
            
            config_file.write_text(json.dumps(config, indent=2))
            logger.info(f"Saved redirect config for {case_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to save redirect config: {e}")
            return False
    
    @staticmethod
    def get_redirect_config(case_id: str) -> Optional[Dict[str, Any]]:
        """Get redirect configuration for a case."""
        try:
            config_file = Path("audit/redirects") / f"{case_id}_redirect.json"
            if config_file.exists():
                return json.loads(config_file.read_text())
            return None
        except Exception as e:
            logger.error(f"Failed to get redirect config: {e}")
            return None
    
    @staticmethod
    def mark_redirect_completed(case_id: str) -> bool:
        """Mark redirect as completed."""
        try:
            config = ApprovalRedirect.get_redirect_config(case_id)
            if config:
                config['status'] = 'completed'
                config['completed_at'] = datetime.now().isoformat()
                
                config_file = Path("audit/redirects") / f"{case_id}_redirect.json"
                config_file.write_text(json.dumps(config, indent=2))
                logger.info(f"Marked redirect completed for {case_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to mark redirect completed: {e}")
            return False
    
    @staticmethod
    def create_approval_listener_url(
        base_dashboard_url: str,
        case_id: str,
        device_id: str,
        extraction_type: str = "android"
    ) -> str:
        """Create a special URL that listens for approval and redirects to dashboard."""
        try:
            # This URL will be embedded in the approval link
            # When the nominee approves, they'll be redirected here
            params = {
                "case_id": case_id,
                "device_id": device_id,
                "extraction_type": extraction_type,
                "auto_extract": "true",
                "timestamp": datetime.now().isoformat()
            }
            
            query_string = urlencode(params)
            listener_url = f"{base_dashboard_url}?{query_string}"
            logger.info(f"Created approval listener URL for {case_id}")
            return listener_url
        except Exception as e:
            logger.error(f"Failed to create listener URL: {e}")
            return ""


class ApprovalNotifier:
    """Notify dashboard of approvals and trigger extraction."""
    
    NOTIFICATION_FILE = Path("audit/approval_notifications.json")
    
    @classmethod
    def initialize(cls):
        """Create notification file if needed."""
        cls.NOTIFICATION_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not cls.NOTIFICATION_FILE.exists():
            cls.NOTIFICATION_FILE.write_text(json.dumps([], indent=2))
    
    @classmethod
    def notify_approval(
        cls,
        case_id: str,
        device_id: str,
        decision: str,
        nominee_name: Optional[str] = None,
        extraction_type: str = "android"
    ) -> bool:
        """Notify dashboard of approval decision."""
        try:
            cls.initialize()
            
            notification = {
                'id': int(time.time() * 1000),
                'timestamp': datetime.now().isoformat(),
                'case_id': case_id,
                'device_id': device_id,
                'decision': decision,
                'nominee_name': nominee_name or 'Unknown',
                'extraction_type': extraction_type,
                'status': 'pending',  # pending, acknowledged, completed
                'auto_extract': decision == 'approved'
            }
            
            # Read existing notifications
            notifications = json.loads(cls.NOTIFICATION_FILE.read_text())
            notifications.append(notification)
            
            # Keep only last 100 notifications
            notifications = notifications[-100:]
            
            # Write back
            cls.NOTIFICATION_FILE.write_text(json.dumps(notifications, indent=2))
            logger.info(f"Notified approval for {case_id}: {decision}")
            return True
        except Exception as e:
            logger.error(f"Failed to notify approval: {e}")
            return False
    
    @classmethod
    def get_pending_notifications(cls) -> list:
        """Get pending notifications for dashboard."""
        try:
            cls.initialize()
            notifications = json.loads(cls.NOTIFICATION_FILE.read_text())
            return [n for n in notifications if n['status'] == 'pending']
        except Exception as e:
            logger.error(f"Failed to get notifications: {e}")
            return []
    
    @classmethod
    def acknowledge_notification(cls, notification_id: int) -> bool:
        """Mark notification as acknowledged."""
        try:
            cls.initialize()
            notifications = json.loads(cls.NOTIFICATION_FILE.read_text())
            
            for n in notifications:
                if n['id'] == notification_id:
                    n['status'] = 'acknowledged'
                    n['acknowledged_at'] = datetime.now().isoformat()
                    break
            
            cls.NOTIFICATION_FILE.write_text(json.dumps(notifications, indent=2))
            logger.info(f"Acknowledged notification {notification_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to acknowledge notification: {e}")
            return False
    
    @classmethod
    def mark_extraction_completed(cls, notification_id: int) -> bool:
        """Mark extraction as completed for notification."""
        try:
            cls.initialize()
            notifications = json.loads(cls.NOTIFICATION_FILE.read_text())
            
            for n in notifications:
                if n['id'] == notification_id:
                    n['status'] = 'completed'
                    n['completed_at'] = datetime.now().isoformat()
                    break
            
            cls.NOTIFICATION_FILE.write_text(json.dumps(notifications, indent=2))
            logger.info(f"Marked extraction completed for notification {notification_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to mark extraction completed: {e}")
            return False


__all__ = ["ApprovalRedirect", "ApprovalNotifier"]
