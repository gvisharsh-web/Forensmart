"""
CONSENT APPROVAL WORKFLOW - Online API & Database Integration

Handles complete consent approval workflow:
- Send approval links via API
- Track approval status in database
- Verify approval responses
- Update consent records
- Handle approval expiration
- Generate approval reports
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json

logger = logging.getLogger(__name__)

# ============================================================================
# CONSENT APPROVAL WORKFLOW CLASS
# ============================================================================

class ConsentApprovalWorkflow:
    """Manages complete consent approval workflow with API & Database"""
    
    def __init__(self, api_client=None, database_manager=None):
        """Initialize workflow with API and Database"""
        self.api_client = api_client
        self.database_manager = database_manager
        self.approval_requests = []
        self.approval_responses = []
        
        # Register API endpoints for consent workflow
        if self.api_client:
            self._register_consent_endpoints()
    
    # ========================================================================
    # API ENDPOINT REGISTRATION
    # ========================================================================
    
    def _register_consent_endpoints(self):
        """Register consent-related API endpoints"""
        endpoints = [
            ('send_approval_link', 'POST', '/consent/send-link', 'Send approval link to nominee'),
            ('check_approval_status', 'GET', '/consent/status/{case_id}', 'Check approval status'),
            ('verify_approval', 'POST', '/consent/verify', 'Verify approval response'),
            ('get_approval_history', 'GET', '/consent/history/{case_id}', 'Get approval history'),
            ('revoke_approval', 'POST', '/consent/revoke', 'Revoke approval'),
            ('extend_approval', 'POST', '/consent/extend', 'Extend approval expiration'),
        ]
        
        for name, method, path, description in endpoints:
            self.api_client.register_endpoint(name, method, path, description)
            logger.info(f"Registered endpoint: {name}")
    
    # ========================================================================
    # SEND APPROVAL LINK
    # ========================================================================
    
    def send_approval_link(self, case_id: str, nominee_email: str, 
                          nominee_name: str = None) -> Dict[str, Any]:
        """Send approval link to nominee via API"""
        try:
            logger.info(f"Sending approval link for case {case_id} to {nominee_email}")
            
            # Generate approval token
            approval_token = self._generate_approval_token(case_id)
            
            # Create approval link
            approval_link = f"https://forensmart.app/approve/{approval_token}"
            
            # Prepare request data
            request_data = {
                'case_id': case_id,
                'nominee_email': nominee_email,
                'nominee_name': nominee_name or 'Nominee',
                'approval_link': approval_link,
                'approval_token': approval_token,
                'sent_at': datetime.now().isoformat(),
                'expires_at': (datetime.now() + timedelta(days=7)).isoformat()
            }
            
            # Send via API
            if self.api_client:
                api_response = self.api_client.post('/consent/send-link', request_data)
                logger.info(f"API response: {api_response}")
            
            # Store in database
            if self.database_manager:
                db_record = self.database_manager.create('approval_requests', {
                    'case_id': case_id,
                    'nominee_email': nominee_email,
                    'nominee_name': nominee_name,
                    'approval_token': approval_token,
                    'approval_link': approval_link,
                    'status': 'sent',
                    'sent_at': datetime.now().isoformat(),
                    'expires_at': (datetime.now() + timedelta(days=7)).isoformat()
                })
                logger.info(f"Database record created: {db_record['id']}")
            
            # Track locally
            self.approval_requests.append(request_data)
            
            return {
                'success': True,
                'case_id': case_id,
                'nominee_email': nominee_email,
                'approval_link': approval_link,
                'approval_token': approval_token,
                'sent_at': datetime.now().isoformat(),
                'expires_at': (datetime.now() + timedelta(days=7)).isoformat(),
                'message': f'Approval link sent to {nominee_email}',
                'next_action': 'Wait for nominee approval'
            }
        except Exception as e:
            logger.error(f"Failed to send approval link: {e}")
            return {'success': False, 'error': str(e)}
    
    # ========================================================================
    # CHECK APPROVAL STATUS
    # ========================================================================
    
    def check_approval_status(self, case_id: str) -> Dict[str, Any]:
        """Check approval status from database"""
        try:
            logger.info(f"Checking approval status for case {case_id}")
            
            # Query database
            if self.database_manager:
                records = self.database_manager.read('approval_requests', 
                                                     {'case_id': case_id})
                
                if records:
                    latest_record = records[-1]
                    status = latest_record.get('status', 'unknown')
                    
                    # Check if expired
                    expires_at = datetime.fromisoformat(latest_record.get('expires_at', ''))
                    is_expired = datetime.now() > expires_at
                    
                    if is_expired and status == 'sent':
                        status = 'expired'
                        # Update database
                        self.database_manager.update('approval_requests', 
                                                    latest_record['id'], 
                                                    {'status': 'expired'})
                    
                    return {
                        'success': True,
                        'case_id': case_id,
                        'status': status,
                        'nominee_email': latest_record.get('nominee_email'),
                        'sent_at': latest_record.get('sent_at'),
                        'expires_at': latest_record.get('expires_at'),
                        'approved_at': latest_record.get('approved_at'),
                        'is_expired': is_expired
                    }
            
            return {'success': False, 'error': 'No approval request found'}
        except Exception as e:
            logger.error(f"Failed to check approval status: {e}")
            return {'success': False, 'error': str(e)}
    
    # ========================================================================
    # VERIFY APPROVAL RESPONSE
    # ========================================================================
    
    def verify_approval(self, approval_token: str, response: str) -> Dict[str, Any]:
        """Verify approval response from nominee"""
        try:
            logger.info(f"Verifying approval response for token {approval_token}")
            
            # Validate token
            if not self._validate_approval_token(approval_token):
                return {'success': False, 'error': 'Invalid or expired approval token'}
            
            # Verify response (approved/rejected)
            if response.lower() not in ['approved', 'rejected']:
                return {'success': False, 'error': 'Invalid response'}
            
            # Find approval request
            approval_request = None
            if self.database_manager:
                records = self.database_manager.read('approval_requests',
                                                    {'approval_token': approval_token})
                if records:
                    approval_request = records[-1]
            
            if not approval_request:
                return {'success': False, 'error': 'Approval request not found'}
            
            # Update database
            if self.database_manager:
                update_data = {
                    'status': response.lower(),
                    'responded_at': datetime.now().isoformat()
                }
                
                if response.lower() == 'approved':
                    update_data['approved_at'] = datetime.now().isoformat()
                
                self.database_manager.update('approval_requests',
                                            approval_request['id'],
                                            update_data)
            
            # Track response
            response_record = {
                'approval_token': approval_token,
                'case_id': approval_request.get('case_id'),
                'response': response.lower(),
                'responded_at': datetime.now().isoformat()
            }
            self.approval_responses.append(response_record)
            
            return {
                'success': True,
                'case_id': approval_request.get('case_id'),
                'approval_token': approval_token,
                'response': response.lower(),
                'responded_at': datetime.now().isoformat(),
                'message': f'Approval {response.lower()} recorded'
            }
        except Exception as e:
            logger.error(f"Failed to verify approval: {e}")
            return {'success': False, 'error': str(e)}
    
    # ========================================================================
    # GET APPROVAL HISTORY
    # ========================================================================
    
    def get_approval_history(self, case_id: str) -> Dict[str, Any]:
        """Get approval history from database"""
        try:
            logger.info(f"Getting approval history for case {case_id}")
            
            if self.database_manager:
                records = self.database_manager.read('approval_requests',
                                                    {'case_id': case_id})
                
                return {
                    'success': True,
                    'case_id': case_id,
                    'total_requests': len(records),
                    'history': records,
                    'latest_status': records[-1].get('status') if records else 'none'
                }
            
            return {'success': False, 'error': 'Database not available'}
        except Exception as e:
            logger.error(f"Failed to get approval history: {e}")
            return {'success': False, 'error': str(e)}
    
    # ========================================================================
    # REVOKE APPROVAL
    # ========================================================================
    
    def revoke_approval(self, case_id: str, reason: str = None) -> Dict[str, Any]:
        """Revoke approval for a case"""
        try:
            logger.info(f"Revoking approval for case {case_id}")
            
            if self.database_manager:
                records = self.database_manager.read('approval_requests',
                                                    {'case_id': case_id})
                
                if records:
                    latest_record = records[-1]
                    
                    # Update status to revoked
                    self.database_manager.update('approval_requests',
                                                latest_record['id'],
                                                {
                                                    'status': 'revoked',
                                                    'revoked_at': datetime.now().isoformat(),
                                                    'revoke_reason': reason
                                                })
                    
                    return {
                        'success': True,
                        'case_id': case_id,
                        'status': 'revoked',
                        'revoked_at': datetime.now().isoformat(),
                        'reason': reason
                    }
            
            return {'success': False, 'error': 'Approval not found'}
        except Exception as e:
            logger.error(f"Failed to revoke approval: {e}")
            return {'success': False, 'error': str(e)}
    
    # ========================================================================
    # EXTEND APPROVAL
    # ========================================================================
    
    def extend_approval(self, case_id: str, days: int = 7) -> Dict[str, Any]:
        """Extend approval expiration"""
        try:
            logger.info(f"Extending approval for case {case_id} by {days} days")
            
            if self.database_manager:
                records = self.database_manager.read('approval_requests',
                                                    {'case_id': case_id})
                
                if records:
                    latest_record = records[-1]
                    
                    # Calculate new expiration
                    new_expiration = (datetime.now() + timedelta(days=days)).isoformat()
                    
                    # Update database
                    self.database_manager.update('approval_requests',
                                                latest_record['id'],
                                                {'expires_at': new_expiration})
                    
                    return {
                        'success': True,
                        'case_id': case_id,
                        'new_expiration': new_expiration,
                        'extended_by_days': days
                    }
            
            return {'success': False, 'error': 'Approval not found'}
        except Exception as e:
            logger.error(f"Failed to extend approval: {e}")
            return {'success': False, 'error': str(e)}
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def _generate_approval_token(self, case_id: str) -> str:
        """Generate unique approval token"""
        import hashlib
        import uuid
        
        unique_string = f"{case_id}-{uuid.uuid4()}-{datetime.now().isoformat()}"
        token = hashlib.sha256(unique_string.encode()).hexdigest()[:32]
        return token
    
    def _validate_approval_token(self, token: str) -> bool:
        """Validate approval token"""
        # Check if token exists in database
        if self.database_manager:
            records = self.database_manager.read('approval_requests',
                                                {'approval_token': token})
            
            if records:
                record = records[-1]
                expires_at = datetime.fromisoformat(record.get('expires_at', ''))
                return datetime.now() <= expires_at
        
        return False
    
    # ========================================================================
    # STATISTICS & REPORTING
    # ========================================================================
    
    def get_approval_statistics(self) -> Dict[str, Any]:
        """Get approval statistics from database"""
        try:
            if self.database_manager:
                all_records = self.database_manager.read('approval_requests', {})
                
                stats = {
                    'total_requests': len(all_records),
                    'approved': len([r for r in all_records if r.get('status') == 'approved']),
                    'rejected': len([r for r in all_records if r.get('status') == 'rejected']),
                    'pending': len([r for r in all_records if r.get('status') == 'sent']),
                    'expired': len([r for r in all_records if r.get('status') == 'expired']),
                    'revoked': len([r for r in all_records if r.get('status') == 'revoked'])
                }
                
                # Calculate approval rate
                if stats['total_requests'] > 0:
                    stats['approval_rate'] = f"{(stats['approved'] / stats['total_requests'] * 100):.1f}%"
                
                return {'success': True, 'statistics': stats}
            
            return {'success': False, 'error': 'Database not available'}
        except Exception as e:
            logger.error(f"Failed to get approval statistics: {e}")
            return {'success': False, 'error': str(e)}

# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_consent_approval_workflow(api_client=None, database_manager=None):
    """Factory function to create consent approval workflow"""
    return ConsentApprovalWorkflow(api_client, database_manager)
