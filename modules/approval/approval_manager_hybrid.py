"""
Hybrid Approval Manager - Production-ready with online/offline support
Handles both Supabase (online) and file-based (offline) approvals
"""

import logging
import json
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class HybridApprovalManager:
    """
    Production-ready approval manager supporting:
    - Online approvals (Supabase)
    - Offline approvals (file-based)
    - Automatic fallback strategy
    - Comprehensive error handling
    - Audit logging
    """
    
    APPROVAL_DIR = Path('audit/approvals')
    
    def __init__(self):
        """Initialize hybrid approval manager"""
        self.approval_dir = self.APPROVAL_DIR
        self.approval_dir.mkdir(parents=True, exist_ok=True)
        logger.info("✅ Hybrid Approval Manager initialized")
    
    # ========================================================================
    # OFFLINE (FILE-BASED) METHODS
    # ========================================================================
    
    def _get_approval_file(self, case_id: str) -> Path:
        """Get approval file path"""
        return self.approval_dir / f"{case_id}_approval.json"
    
    def _read_approval_file(self, case_id: str) -> Optional[Dict[str, Any]]:
        """Read approval from file with error handling"""
        try:
            approval_file = self._get_approval_file(case_id)
            
            if not approval_file.exists():
                logger.debug(f"Approval file not found for {case_id}")
                return None
            
            logger.debug(f"Reading approval file for {case_id}")
            approval_data = json.loads(approval_file.read_text(encoding='utf-8'))
            
            logger.info(f"✅ Approval loaded from file for {case_id}: {approval_data.get('decision')}")
            return approval_data
        
        except json.JSONDecodeError as e:
            logger.error(f"Corrupted approval file for {case_id}: {e}", exc_info=True)
            return None
        except PermissionError as e:
            logger.error(f"Permission denied reading approval file for {case_id}: {e}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"Unexpected error reading approval file for {case_id}: {e}", exc_info=True)
            return None
    
    def _write_approval_file(self, case_id: str, approval_data: Dict[str, Any]) -> bool:
        """Write approval to file with error handling"""
        try:
            approval_file = self._get_approval_file(case_id)
            
            logger.debug(f"Writing approval file for {case_id}")
            approval_file.write_text(
                json.dumps(approval_data, indent=2),
                encoding='utf-8'
            )
            
            logger.info(f"✅ Approval saved to file for {case_id}")
            return True
        
        except PermissionError as e:
            logger.error(f"Permission denied writing approval file for {case_id}: {e}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"Failed to write approval file for {case_id}: {e}", exc_info=True)
            return False
    
    # ========================================================================
    # ONLINE (SUPABASE) METHODS
    # ========================================================================
    
    def _get_supabase_client(self):
        """Get Supabase client"""
        try:
            from modules.approval.supabase_client import get_supabase_client
            return get_supabase_client()
        except ImportError:
            logger.warning("Supabase client not available")
            return None
        except Exception as e:
            logger.warning(f"Failed to get Supabase client: {e}")
            return None
    
    def _read_approval_online(self, case_id: str) -> Optional[Dict[str, Any]]:
        """Read approval from Supabase"""
        try:
            supabase_client = self._get_supabase_client()
            if not supabase_client or not supabase_client.is_available:
                logger.debug("Supabase not available, skipping online read")
                return None
            
            approval = supabase_client.get_approval(case_id)
            if approval:
                logger.info(f"✅ Approval loaded from Supabase for {case_id}")
            return approval
        
        except Exception as e:
            logger.warning(f"Failed to read approval from Supabase: {e}")
            return None
    
    def _write_approval_online(self, case_id: str, approval_data: Dict[str, Any]) -> bool:
        """Write approval to Supabase"""
        try:
            supabase_client = self._get_supabase_client()
            if not supabase_client or not supabase_client.is_available:
                logger.debug("Supabase not available, skipping online write")
                return False
            
            success = supabase_client.save_approval(
                case_id=case_id,
                decision=approval_data.get('decision', 'unknown'),
                nominee_name=approval_data.get('nominee_name', ''),
                metadata={
                    'timestamp': approval_data.get('timestamp'),
                    'source': approval_data.get('source', 'unknown')
                }
            )
            
            if success:
                logger.info(f"✅ Approval saved to Supabase for {case_id}")
            return success
        
        except Exception as e:
            logger.warning(f"Failed to write approval to Supabase: {e}")
            return False
    
    # ========================================================================
    # HYBRID METHODS (Main API)
    # ========================================================================
    
    def check_approval_with_fallback(self, case_id: str) -> Dict[str, Any]:
        """
        Check approval with fallback strategy:
        1. Try Supabase (online)
        2. Fall back to file (offline)
        3. Return status and source
        
        Returns:
            {
                'approved': bool,
                'status': 'approved' | 'denied' | 'pending',
                'source': 'supabase' | 'file' | 'none',
                'decision': str,
                'nominee_name': str,
                'timestamp': str,
                'error': str (if any)
            }
        """
        logger.info(f"🔍 Checking approval for {case_id} with fallback strategy")
        
        # Try online first
        logger.debug("Step 1: Trying Supabase (online)")
        approval = self._read_approval_online(case_id)
        
        if approval:
            decision = approval.get('decision', 'unknown')
            return {
                'approved': decision == 'approved',
                'status': decision,
                'source': 'supabase',
                'decision': decision,
                'nominee_name': approval.get('nominee_name', ''),
                'timestamp': approval.get('timestamp', ''),
                'error': None
            }
        
        logger.debug("Step 2: Supabase unavailable, trying file (offline)")
        
        # Fall back to file
        approval = self._read_approval_file(case_id)
        
        if approval:
            decision = approval.get('decision', 'unknown')
            return {
                'approved': decision == 'approved',
                'status': decision,
                'source': 'file',
                'decision': decision,
                'nominee_name': approval.get('nominee_name', ''),
                'timestamp': approval.get('timestamp', ''),
                'error': None
            }
        
        logger.debug("Step 3: No approval found in either source")
        
        # No approval found
        return {
            'approved': False,
            'status': 'pending',
            'source': 'none',
            'decision': 'pending',
            'nominee_name': '',
            'timestamp': '',
            'error': 'No approval found'
        }
    
    def mark_approved(self, case_id: str, nominee_name: str = "") -> Dict[str, Any]:
        """
        Mark case as approved in both online and offline
        
        Returns:
            {
                'success': bool,
                'online_saved': bool,
                'offline_saved': bool,
                'error': str (if any)
            }
        """
        logger.info(f"✅ Marking case {case_id} as APPROVED")
        
        approval_data = {
            'case_id': case_id,
            'decision': 'approved',
            'nominee_name': nominee_name,
            'timestamp': datetime.now().isoformat(),
            'source': 'hybrid'
        }
        
        # Save to both sources
        online_saved = self._write_approval_online(case_id, approval_data)
        offline_saved = self._write_approval_file(case_id, approval_data)
        
        # Success if at least one saved
        success = online_saved or offline_saved
        
        if success:
            logger.info(f"✅ Approval marked for {case_id} (online: {online_saved}, offline: {offline_saved})")
        else:
            logger.error(f"❌ Failed to mark approval for {case_id}")
        
        return {
            'success': success,
            'online_saved': online_saved,
            'offline_saved': offline_saved,
            'error': None if success else 'Failed to save approval'
        }
    
    def mark_denied(self, case_id: str, nominee_name: str = "") -> Dict[str, Any]:
        """
        Mark case as denied in both online and offline
        
        Returns:
            {
                'success': bool,
                'online_saved': bool,
                'offline_saved': bool,
                'error': str (if any)
            }
        """
        logger.info(f"❌ Marking case {case_id} as DENIED")
        
        approval_data = {
            'case_id': case_id,
            'decision': 'denied',
            'nominee_name': nominee_name,
            'timestamp': datetime.now().isoformat(),
            'source': 'hybrid'
        }
        
        # Save to both sources
        online_saved = self._write_approval_online(case_id, approval_data)
        offline_saved = self._write_approval_file(case_id, approval_data)
        
        # Success if at least one saved
        success = online_saved or offline_saved
        
        if success:
            logger.info(f"❌ Denial marked for {case_id} (online: {online_saved}, offline: {offline_saved})")
        else:
            logger.error(f"❌ Failed to mark denial for {case_id}")
        
        return {
            'success': success,
            'online_saved': online_saved,
            'offline_saved': offline_saved,
            'error': None if success else 'Failed to save denial'
        }
    
    def get_approval_status(self, case_id: str) -> Dict[str, Any]:
        """Get detailed approval status"""
        logger.debug(f"Getting approval status for {case_id}")
        
        result = self.check_approval_with_fallback(case_id)
        
        return {
            'case_id': case_id,
            'is_approved': result['approved'],
            'status': result['status'],
            'source': result['source'],
            'nominee_name': result['nominee_name'],
            'timestamp': result['timestamp'],
            'can_extract': result['approved'],
            'error': result['error']
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Check approval system health"""
        logger.debug("Performing approval system health check")
        
        supabase_client = self._get_supabase_client()
        supabase_health = supabase_client.health_check() if supabase_client else {
            'status': 'offline',
            'available': False,
            'message': 'Supabase client not available'
        }
        
        # Check file system
        file_system_ok = self.approval_dir.exists() and self.approval_dir.is_dir()
        
        return {
            'system': 'hybrid_approval_manager',
            'status': 'healthy' if (supabase_health['available'] or file_system_ok) else 'degraded',
            'online': supabase_health,
            'offline': {
                'status': 'online' if file_system_ok else 'offline',
                'available': file_system_ok,
                'message': 'File system ready' if file_system_ok else 'File system not available'
            },
            'timestamp': datetime.now().isoformat()
        }


# Singleton instance
_hybrid_manager = None


def get_hybrid_approval_manager() -> HybridApprovalManager:
    """Get or create hybrid approval manager singleton"""
    global _hybrid_manager
    if _hybrid_manager is None:
        _hybrid_manager = HybridApprovalManager()
    return _hybrid_manager
