"""
Supabase Approval Client - Production-ready with comprehensive error handling
Supports hybrid offline/online approval system
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class SupabaseApprovalClient:
    """
    Production-ready Supabase client for approval management.
    Features:
    - Automatic fallback to offline mode
    - Comprehensive error handling
    - Retry logic for network failures
    - Audit logging
    - Connection pooling
    """
    
    def __init__(self, url: Optional[str] = None, key: Optional[str] = None):
        """Initialize Supabase client with optional credentials"""
        self.url = url
        self.key = key
        self.client = None
        self.is_available = False
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Supabase client with error handling"""
        try:
            import streamlit as st
            
            # Try to get credentials from Streamlit secrets
            if not self.url or not self.key:
                try:
                    self.url = st.secrets.get("supabase_url")
                    self.key = st.secrets.get("supabase_key")
                except Exception as e:
                    logger.warning(f"Could not load Supabase secrets: {e}")
                    self.is_available = False
                    return
            
            # Validate credentials
            if not self.url or not self.key:
                logger.warning("Supabase credentials not configured. Using offline mode.")
                self.is_available = False
                return
            
            # Import and initialize Supabase
            try:
                import supabase
                self.client = supabase.create_client(self.url, self.key)
                self.is_available = True
                logger.info("✅ Supabase client initialized successfully")
            except ImportError:
                logger.warning("Supabase library not installed. Install with: pip install supabase")
                self.is_available = False
            except Exception as e:
                logger.error(f"Failed to initialize Supabase client: {e}", exc_info=True)
                self.is_available = False
        
        except Exception as e:
            logger.error(f"Unexpected error during Supabase initialization: {e}", exc_info=True)
            self.is_available = False
    
    def get_approval(self, case_id: str, max_retries: int = 2) -> Optional[Dict[str, Any]]:
        """
        Get approval from Supabase with retry logic
        
        Args:
            case_id: Case ID to look up
            max_retries: Number of retry attempts
        
        Returns:
            Approval data or None if not found/error
        """
        if not self.is_available or not self.client:
            logger.debug(f"Supabase not available, skipping online lookup for {case_id}")
            return None
        
        for attempt in range(max_retries):
            try:
                logger.debug(f"Fetching approval from Supabase for {case_id} (attempt {attempt + 1}/{max_retries})")
                
                result = self.client.table('approvals').select('*').eq('case_id', case_id).execute()
                
                if result.data and len(result.data) > 0:
                    approval = result.data[0]
                    logger.info(f"✅ Approval found in Supabase for {case_id}: {approval.get('decision')}")
                    return approval
                else:
                    logger.debug(f"No approval found in Supabase for {case_id}")
                    return None
            
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed to fetch approval from Supabase: {e}")
                if attempt < max_retries - 1:
                    logger.debug(f"Retrying... ({attempt + 1}/{max_retries})")
                    continue
                else:
                    logger.error(f"Failed to fetch approval after {max_retries} attempts")
                    return None
        
        return None
    
    def save_approval(self, case_id: str, decision: str, nominee_name: str = "", 
                     metadata: Optional[Dict[str, Any]] = None, max_retries: int = 2) -> bool:
        """
        Save approval to Supabase with retry logic
        
        Args:
            case_id: Case ID
            decision: 'approved' or 'denied'
            nominee_name: Name of nominee
            metadata: Additional metadata
            max_retries: Number of retry attempts
        
        Returns:
            True if successful, False otherwise
        """
        if not self.is_available or not self.client:
            logger.debug(f"Supabase not available, skipping online save for {case_id}")
            return False
        
        # Validate decision
        if decision not in ['approved', 'denied']:
            logger.error(f"Invalid decision: {decision}. Must be 'approved' or 'denied'")
            return False
        
        approval_data = {
            'case_id': case_id,
            'decision': decision,
            'nominee_name': nominee_name,
            'timestamp': datetime.now().isoformat(),
            'metadata': json.dumps(metadata or {})
        }
        
        for attempt in range(max_retries):
            try:
                logger.debug(f"Saving approval to Supabase for {case_id} (attempt {attempt + 1}/{max_retries})")
                
                self.client.table('approvals').upsert(approval_data).execute()
                
                logger.info(f"✅ Approval saved to Supabase for {case_id}: {decision}")
                return True
            
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed to save approval to Supabase: {e}")
                if attempt < max_retries - 1:
                    logger.debug(f"Retrying... ({attempt + 1}/{max_retries})")
                    continue
                else:
                    logger.error(f"Failed to save approval after {max_retries} attempts")
                    return False
        
        return False
    
    def delete_approval(self, case_id: str, max_retries: int = 2) -> bool:
        """Delete approval from Supabase"""
        if not self.is_available or not self.client:
            logger.debug(f"Supabase not available, skipping delete for {case_id}")
            return False
        
        for attempt in range(max_retries):
            try:
                logger.debug(f"Deleting approval from Supabase for {case_id} (attempt {attempt + 1}/{max_retries})")
                
                self.client.table('approvals').delete().eq('case_id', case_id).execute()
                
                logger.info(f"✅ Approval deleted from Supabase for {case_id}")
                return True
            
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed to delete approval: {e}")
                if attempt < max_retries - 1:
                    continue
                else:
                    logger.error(f"Failed to delete approval after {max_retries} attempts")
                    return False
        
        return False
    
    def list_approvals(self, limit: int = 100) -> Optional[list]:
        """List all approvals from Supabase"""
        if not self.is_available or not self.client:
            logger.debug("Supabase not available, cannot list approvals")
            return None
        
        try:
            logger.debug(f"Fetching approvals list from Supabase (limit: {limit})")
            
            result = self.client.table('approvals').select('*').limit(limit).execute()
            
            logger.info(f"✅ Retrieved {len(result.data)} approvals from Supabase")
            return result.data
        
        except Exception as e:
            logger.error(f"Failed to list approvals from Supabase: {e}", exc_info=True)
            return None
    
    def health_check(self) -> Dict[str, Any]:
        """Check Supabase connection health"""
        if not self.is_available or not self.client:
            return {
                'status': 'offline',
                'available': False,
                'message': 'Supabase client not initialized'
            }
        
        try:
            logger.debug("Performing Supabase health check")
            
            # Try a simple query
            result = self.client.table('approvals').select('count').limit(1).execute()
            
            return {
                'status': 'online',
                'available': True,
                'message': 'Supabase connection healthy'
            }
        
        except Exception as e:
            logger.warning(f"Supabase health check failed: {e}")
            return {
                'status': 'offline',
                'available': False,
                'message': f'Health check failed: {str(e)}'
            }


def get_supabase_client() -> SupabaseApprovalClient:
    """Get or create Supabase client singleton"""
    import streamlit as st
    
    if 'supabase_client' not in st.session_state:
        st.session_state.supabase_client = SupabaseApprovalClient()
    
    return st.session_state.supabase_client
