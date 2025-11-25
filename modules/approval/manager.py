"""
Consolidated Approval Management System for ForenSmart
======================================================

Handles all approval-related operations:
- Approval synchronization (ApprovalSync)
- Approval utilities (file management)
- Approval notifications
- Auto-extraction triggers

This consolidates approval_sync.py, approval_utils.py, approval_redirect.py, 
and approval_auto_extraction.py into a single module.
"""

import json
import logging
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


class ApprovalManager:
    """Unified approval management system."""
    
    # Local cache for approvals
    _cache: Dict[str, Dict[str, Any]] = {}
    _cache_timestamp: Dict[str, float] = {}
    _cache_ttl = 30  # 30 seconds TTL
    
    @staticmethod
    def get_approvals_file() -> Path:
        """Get path to approvals file."""
        approvals_dir = Path('audit/approvals')
        approvals_dir.mkdir(parents=True, exist_ok=True)
        return approvals_dir / 'approvals.json'
    
    @staticmethod
    def _is_cache_valid(case_id: str) -> bool:
        """Check if cached approval is still valid."""
        if case_id not in ApprovalManager._cache_timestamp:
            return False
        age = time.time() - ApprovalManager._cache_timestamp[case_id]
        return age < ApprovalManager._cache_ttl
    
    @staticmethod
    def get_approval_status(case_id: str, use_cache: bool = True) -> Optional[Dict[str, Any]]:
        """Get approval status from cache or file."""
        # Check cache first
        if use_cache and ApprovalManager._is_cache_valid(case_id):
            return ApprovalManager._cache.get(case_id)
        
        # Read from file
        try:
            approvals_file = ApprovalManager.get_approvals_file()
            if approvals_file.exists():
                data = json.loads(approvals_file.read_text(encoding="utf-8"))
                if case_id in data:
                    approval = data[case_id]
                    # Update cache
                    ApprovalManager._cache[case_id] = approval
                    ApprovalManager._cache_timestamp[case_id] = time.time()
                    return approval
        except Exception as e:
            logger.error(f"Failed to read approval status: {e}")
        
        return None
    
    @staticmethod
    def is_approved(case_id: str) -> bool:
        """Check if case is approved."""
        status = ApprovalManager.get_approval_status(case_id)
        return status and status.get('decision') == 'approved'
    
    @staticmethod
    def is_denied(case_id: str) -> bool:
        """Check if case is denied."""
        status = ApprovalManager.get_approval_status(case_id)
        return status and status.get('decision') == 'denied'
    
    @staticmethod
    def get_approval_age_seconds(case_id: str) -> Optional[int]:
        """Get approval age in seconds."""
        status = ApprovalManager.get_approval_status(case_id)
        if not status:
            return None
        
        try:
            created_at = datetime.fromisoformat(status.get('timestamp', ''))
            age = datetime.now() - created_at
            return int(age.total_seconds())
        except Exception:
            return None
    
    @staticmethod
    def is_approval_expired(case_id: str, max_age_hours: int = 24) -> bool:
        """Check if approval has expired."""
        age_seconds = ApprovalManager.get_approval_age_seconds(case_id)
        if age_seconds is None:
            return False
        
        max_age_seconds = max_age_hours * 3600
        return age_seconds > max_age_seconds
    
    @staticmethod
    def mark_approved(case_id: str, nominee_name: str) -> bool:
        """Mark case as approved."""
        try:
            approvals_file = ApprovalManager.get_approvals_file()
            
            # Read existing approvals
            data = {}
            if approvals_file.exists():
                data = json.loads(approvals_file.read_text(encoding="utf-8"))
            
            # Update approval
            data[case_id] = {
                'case_id': case_id,
                'decision': 'approved',
                'nominee_name': nominee_name,
                'timestamp': datetime.now().isoformat()
            }
            
            # Write back
            approvals_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
            
            # Clear cache
            ApprovalManager._cache.pop(case_id, None)
            ApprovalManager._cache_timestamp.pop(case_id, None)
            
            logger.info(f"Marked {case_id} as approved")
            return True
            
        except Exception as e:
            logger.error(f"Failed to mark approval: {e}")
            return False
    
    @staticmethod
    def mark_denied(case_id: str, nominee_name: str) -> bool:
        """Mark case as denied."""
        try:
            approvals_file = ApprovalManager.get_approvals_file()
            
            # Read existing approvals
            data = {}
            if approvals_file.exists():
                data = json.loads(approvals_file.read_text(encoding="utf-8"))
            
            # Update approval
            data[case_id] = {
                'case_id': case_id,
                'decision': 'denied',
                'nominee_name': nominee_name,
                'timestamp': datetime.now().isoformat()
            }
            
            # Write back
            approvals_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
            
            # Clear cache
            ApprovalManager._cache.pop(case_id, None)
            ApprovalManager._cache_timestamp.pop(case_id, None)
            
            logger.info(f"Marked {case_id} as denied")
            return True
            
        except Exception as e:
            logger.error(f"Failed to mark denial: {e}")
            return False
    
    @staticmethod
    def get_all_approvals() -> Dict[str, Dict[str, Any]]:
        """Get all approvals."""
        try:
            approvals_file = ApprovalManager.get_approvals_file()
            if approvals_file.exists():
                return json.loads(approvals_file.read_text(encoding="utf-8"))
        except Exception as e:
            logger.error(f"Failed to read all approvals: {e}")
        
        return {}
    
    @staticmethod
    def check_approval_with_fallback(case_id: str, online_check_fn=None) -> Dict[str, Any]:
        """
        Check approval with fallback strategy:
        1. First try online check (if provided)
        2. Fall back to file-based check (offline)
        3. Return approval status with source
        
        Args:
            case_id: Case ID to check
            online_check_fn: Optional function to check online approval (e.g., API call)
            
        Returns:
            {
                'approved': bool,
                'source': 'online' | 'file' | 'none',
                'status': 'approved' | 'denied' | 'pending',
                'timestamp': ISO timestamp,
                'nominee_name': str
            }
        """
        result = {
            'approved': False,
            'source': 'none',
            'status': 'pending',
            'timestamp': None,
            'nominee_name': None
        }
        
        # Try online check first (if provided)
        if online_check_fn and callable(online_check_fn):
            try:
                online_result = online_check_fn(case_id)
                if online_result and online_result.get('approved'):
                    result['approved'] = True
                    result['source'] = 'online'
                    result['status'] = 'approved'
                    result['timestamp'] = online_result.get('timestamp')
                    result['nominee_name'] = online_result.get('nominee_name')
                    logger.info(f"Approval found via online check for {case_id}")
                    return result
                elif online_result and online_result.get('denied'):
                    result['approved'] = False
                    result['source'] = 'online'
                    result['status'] = 'denied'
                    result['timestamp'] = online_result.get('timestamp')
                    result['nominee_name'] = online_result.get('nominee_name')
                    logger.info(f"Approval denied via online check for {case_id}")
                    return result
            except Exception as e:
                logger.warning(f"Online approval check failed, falling back to file: {e}")
        
        # Fall back to file-based check (offline)
        try:
            approval_file = Path('audit/approvals') / f"{case_id}_approval.json"
            if approval_file.exists():
                approval_data = json.loads(approval_file.read_text(encoding="utf-8"))
                decision = approval_data.get('decision', 'pending')
                
                if decision == 'approved':
                    result['approved'] = True
                    result['source'] = 'file'
                    result['status'] = 'approved'
                    result['timestamp'] = approval_data.get('timestamp')
                    result['nominee_name'] = approval_data.get('nominee_name')
                    logger.info(f"Approval found via file for {case_id}")
                    return result
                elif decision == 'denied':
                    result['approved'] = False
                    result['source'] = 'file'
                    result['status'] = 'denied'
                    result['timestamp'] = approval_data.get('timestamp')
                    result['nominee_name'] = approval_data.get('nominee_name')
                    logger.info(f"Approval denied via file for {case_id}")
                    return result
        except Exception as e:
            logger.warning(f"File-based approval check failed: {e}")
        
        # No approval found
        logger.info(f"No approval found for {case_id} (checked online and file)")
        return result
    
    @staticmethod
    def clear_approval(case_id: str) -> bool:
        """Clear approval for a case."""
        try:
            approvals_file = ApprovalManager.get_approvals_file()
            
            if approvals_file.exists():
                data = json.loads(approvals_file.read_text(encoding="utf-8"))
                data.pop(case_id, None)
                approvals_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
            
            # Clear cache
            ApprovalManager._cache.pop(case_id, None)
            ApprovalManager._cache_timestamp.pop(case_id, None)
            
            logger.info(f"Cleared approval for {case_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear approval: {e}")
            return False


# Backward compatibility - keep old class names as aliases
ApprovalSync = ApprovalManager
