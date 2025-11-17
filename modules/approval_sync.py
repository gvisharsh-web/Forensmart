"""Real-time approval synchronization with cloud backend."""
from __future__ import annotations

import json
import logging
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


class ApprovalSync:
    """Synchronize approvals across local and cloud backends."""

    # Local cache
    _cache: Dict[str, Dict[str, Any]] = {}
    _cache_timestamp: Dict[str, float] = {}
    _cache_ttl = 300  # 5 minutes

    @staticmethod
    def _is_cache_valid(case_id: str) -> bool:
        """Check if cached approval is still valid."""
        if case_id not in ApprovalSync._cache_timestamp:
            return False
        
        age = time.time() - ApprovalSync._cache_timestamp[case_id]
        return age < ApprovalSync._cache_ttl

    @staticmethod
    def get_approval_status(case_id: str, use_cache: bool = True) -> Optional[Dict[str, Any]]:
        """Get approval status from cache or file."""
        # Check cache first
        if use_cache and ApprovalSync._is_cache_valid(case_id):
            return ApprovalSync._cache.get(case_id)

        # Read from file
        try:
            from modules.approval_utils import get_approvals_file
            
            approvals_file = get_approvals_file()
            if approvals_file.exists():
                data = json.loads(approvals_file.read_text(encoding="utf-8"))
                if case_id in data:
                    approval = data[case_id]
                    # Update cache
                    ApprovalSync._cache[case_id] = approval
                    ApprovalSync._cache_timestamp[case_id] = time.time()
                    return approval
        except Exception as e:
            logger.error(f"Failed to read approval status: {e}")

        return None

    @staticmethod
    def save_approval_status(
        case_id: str,
        decision: str,
        nominee_name: Optional[str] = None,
        message: Optional[str] = None
    ) -> bool:
        """Save approval status and sync to cache."""
        try:
            from modules.approval_utils import save_approval_decision
            
            # Save to file
            success = save_approval_decision(case_id, decision, nominee_name, message)
            
            if success:
                # Update cache
                approval = {
                    "decision": decision,
                    "nominee_name": nominee_name or "",
                    "message": message or "",
                    "timestamp": datetime.now().isoformat(),
                }
                ApprovalSync._cache[case_id] = approval
                ApprovalSync._cache_timestamp[case_id] = time.time()
                logger.info(f"Approval saved and cached for {case_id}")
                return True
            
            return False
        except Exception as e:
            logger.error(f"Failed to save approval status: {e}")
            return False

    @staticmethod
    def clear_cache(case_id: Optional[str] = None) -> None:
        """Clear approval cache."""
        if case_id:
            ApprovalSync._cache.pop(case_id, None)
            ApprovalSync._cache_timestamp.pop(case_id, None)
            logger.info(f"Cleared cache for {case_id}")
        else:
            ApprovalSync._cache.clear()
            ApprovalSync._cache_timestamp.clear()
            logger.info("Cleared all approval cache")

    @staticmethod
    def get_approval_history(case_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get approval history for a case."""
        try:
            from modules.approval_utils import get_approvals_file
            
            approvals_file = get_approvals_file()
            if approvals_file.exists():
                data = json.loads(approvals_file.read_text(encoding="utf-8"))
                if case_id in data:
                    # Return as list (single entry for now)
                    return [data[case_id]]
            
            return []
        except Exception as e:
            logger.error(f"Failed to get approval history: {e}")
            return []

    @staticmethod
    def is_approved(case_id: str) -> bool:
        """Check if case is approved."""
        status = ApprovalSync.get_approval_status(case_id)
        return status and status.get("decision") == "approved"

    @staticmethod
    def is_denied(case_id: str) -> bool:
        """Check if case is denied."""
        status = ApprovalSync.get_approval_status(case_id)
        return status and status.get("decision") == "denied"

    @staticmethod
    def is_pending(case_id: str) -> bool:
        """Check if approval is pending."""
        status = ApprovalSync.get_approval_status(case_id)
        return status is None

    @staticmethod
    def get_approval_age_seconds(case_id: str) -> Optional[int]:
        """Get age of approval in seconds."""
        status = ApprovalSync.get_approval_status(case_id)
        if not status:
            return None
        
        try:
            timestamp_str = status.get("timestamp")
            if timestamp_str:
                timestamp = datetime.fromisoformat(timestamp_str)
                age = datetime.now() - timestamp
                return int(age.total_seconds())
        except Exception as e:
            logger.error(f"Failed to calculate approval age: {e}")
        
        return None

    @staticmethod
    def is_approval_expired(case_id: str, max_age_hours: int = 24) -> bool:
        """Check if approval has expired."""
        age_seconds = ApprovalSync.get_approval_age_seconds(case_id)
        if age_seconds is None:
            return True
        
        max_age_seconds = max_age_hours * 3600
        return age_seconds > max_age_seconds


__all__ = ["ApprovalSync"]
