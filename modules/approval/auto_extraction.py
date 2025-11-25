"""Auto-extraction trigger when approval is received."""
from __future__ import annotations

import logging
from typing import Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class ApprovalAutoExtraction:
    """Handle automatic extraction when approval is received."""
    
    @staticmethod
    def check_and_trigger_extraction(
        case_id: str,
        device_id: str,
        extraction_type: str = "android"
    ) -> Dict[str, Any]:
        """Check if approval exists and trigger extraction if approved."""
        result = {
            "triggered": False,
            "case_id": case_id,
            "device_id": device_id,
            "extraction_type": extraction_type,
            "message": "",
            "approval_status": None
        }
        
        try:
            from modules.approval.sync import ApprovalSync
            from modules.approval.redirect import ApprovalNotifier
            
            # Check approval status
            approval = ApprovalSync.get_approval_status(case_id, use_cache=False)
            
            if not approval:
                result["message"] = "No approval found"
                result["approval_status"] = "pending"
                logger.info(f"No approval found for {case_id}")
                return result
            
            decision = approval.get("decision")
            
            if decision == "denied":
                result["message"] = "Approval was denied"
                result["approval_status"] = "denied"
                logger.warning(f"Approval denied for {case_id}")
                return result
            
            if decision != "approved":
                result["message"] = "Approval is still pending"
                result["approval_status"] = "pending"
                logger.info(f"Approval pending for {case_id}")
                return result
            
            # Approval is granted - mark as ready for extraction
            result["triggered"] = True
            result["approval_status"] = "approved"
            result["message"] = f"Approval confirmed - ready to start {extraction_type} extraction"
            
            # Acknowledge the notification if it exists
            try:
                pending = ApprovalNotifier.get_pending_notifications()
                for notification in pending:
                    if notification['case_id'] == case_id:
                        ApprovalNotifier.acknowledge_notification(notification['id'])
                        logger.info(f"Acknowledged notification for {case_id}")
                        break
            except Exception as e:
                logger.warning(f"Failed to acknowledge notification: {e}")
            
            logger.info(f"Extraction auto-trigger ready for {case_id}")
            return result
            
        except Exception as e:
            result["message"] = f"Error checking approval: {str(e)}"
            logger.error(f"Failed to check approval: {e}")
            return result
    
    @staticmethod
    def get_auto_extraction_params() -> Optional[Dict[str, str]]:
        """Get auto-extraction parameters from URL query params."""
        try:
            import streamlit as st
            
            params = st.query_params if hasattr(st, 'query_params') else st.experimental_get_query_params()
            
            # Check for auto-extraction trigger
            if 'auto_extract' in params or 'case_id' in params:
                case_id = params.get('case_id')
                if isinstance(case_id, list):
                    case_id = case_id[-1]
                
                extraction_type = params.get('extraction_type', 'android')
                if isinstance(extraction_type, list):
                    extraction_type = extraction_type[-1]
                
                device_id = params.get('device_id')
                if isinstance(device_id, list):
                    device_id = device_id[-1]
                
                if case_id:
                    return {
                        'case_id': case_id,
                        'device_id': device_id or 'auto',
                        'extraction_type': extraction_type,
                        'auto_extract': True
                    }
        except Exception as e:
            logger.warning(f"Failed to get auto-extraction params: {e}")
        
        return None
    
    @staticmethod
    def render_auto_extraction_ui(
        case_id: str,
        device_id: str,
        extraction_type: str = "android"
    ) -> bool:
        """Render UI for auto-extraction and return if extraction should start."""
        try:
            import streamlit as st
            
            # Check approval status
            result = ApprovalAutoExtraction.check_and_trigger_extraction(
                case_id, device_id, extraction_type
            )
            
            if result["triggered"]:
                st.success(f"✅ {result['message']}")
                st.info("Starting automatic extraction in 3 seconds...")
                
                # Show countdown
                import time
                for i in range(3, 0, -1):
                    st.write(f"⏳ {i}...")
                    time.sleep(1)
                
                return True
            else:
                if result["approval_status"] == "denied":
                    st.error(f"❌ {result['message']}")
                elif result["approval_status"] == "pending":
                    st.warning(f"⏳ {result['message']}")
                    st.info("Please wait for the nominee to approve the extraction request.")
                else:
                    st.info(f"ℹ️ {result['message']}")
                
                return False
                
        except Exception as e:
            logger.error(f"Failed to render auto-extraction UI: {e}")
            return False
    
    @staticmethod
    def should_auto_extract() -> bool:
        """Check if auto-extraction should be triggered."""
        try:
            params = ApprovalAutoExtraction.get_auto_extraction_params()
            if params and params.get('auto_extract'):
                case_id = params['case_id']
                
                from modules.approval.sync import ApprovalSync
                return ApprovalSync.is_approved(case_id)
            
            return False
        except Exception as e:
            logger.warning(f"Failed to check auto-extract: {e}")
            return False


__all__ = ["ApprovalAutoExtraction"]
