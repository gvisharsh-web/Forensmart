"""
CONSENT ERROR HANDLER - Error handling for consent operations

Integrates error handling with:
- Consent validation
- Approval verification
- Consent level checking
- Module access control
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

# ============================================================================
# CONSENT ERROR HANDLER CLASS
# ============================================================================

class ConsentErrorHandler:
    """Handles errors in consent operations"""
    
    def __init__(self):
        try:
            from modules.error_handling import ErrorHandlingSystem
            self.error_system = ErrorHandlingSystem()
            self.available = True
        except ImportError:
            logger.warning("Error handling system not available")
            self.available = False
        
        self.consent_errors = []
    
    # ========================================================================
    # CONSENT VALIDATION ERRORS
    # ========================================================================
    
    def handle_consent_not_given_error(self, case_id: str) -> Dict[str, Any]:
        """Handle consent not given error"""
        try:
            logger.warning(f"Consent not given for case {case_id}")
            
            error_info = {
                'type': 'ConsentNotGiven',
                'case_id': case_id,
                'message': 'Extraction requires consent'
            }
            
            if self.available:
                error_result = self.error_system.handle_error(
                    error=Exception("Consent not given")
                )
                
                self.consent_errors.append({
                    'case_id': case_id,
                    'error_type': 'ConsentNotGiven',
                    'error_class': error_result['error_info']['type'],
                    'message': 'Consent not given',
                    'status': 'handled',
                    'timestamp': datetime.now().isoformat()
                })
                
                return {
                    'success': False,
                    'case_id': case_id,
                    'error': 'Consent not given',
                    'error_type': error_result['error_info']['type'],
                    'recovery': error_result['rectification'],
                    'recommendations': self._get_consent_not_given_recommendations(),
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'success': False,
                    'case_id': case_id,
                    'error': 'Consent not given'
                }
        except Exception as e:
            logger.error(f"Consent not given error handling failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _get_consent_not_given_recommendations(self) -> List[str]:
        """Get consent not given recommendations"""
        return [
            'Request consent from nominee',
            'Send approval link to nominee',
            'Provide clear instructions for approval',
            'Follow up with nominee if needed',
            'Check approval email for confirmation',
            'Verify nominee contact information'
        ]
    
    # ========================================================================
    # APPROVAL PENDING ERRORS
    # ========================================================================
    
    def handle_approval_pending_error(self, case_id: str, 
                                     nominee_id: str) -> Dict[str, Any]:
        """Handle approval pending error"""
        try:
            logger.info(f"Approval pending for case {case_id}, nominee {nominee_id}")
            
            if self.available:
                error_result = self.error_system.handle_error(
                    error=Exception("Approval pending")
                )
                
                self.consent_errors.append({
                    'case_id': case_id,
                    'nominee_id': nominee_id,
                    'error_type': 'ApprovalPending',
                    'error_class': error_result['error_info']['type'],
                    'message': 'Approval pending',
                    'status': 'handled',
                    'timestamp': datetime.now().isoformat()
                })
                
                return {
                    'success': False,
                    'case_id': case_id,
                    'nominee_id': nominee_id,
                    'error': 'Approval pending',
                    'error_type': error_result['error_info']['type'],
                    'recovery': error_result['rectification'],
                    'recommendations': self._get_approval_pending_recommendations(),
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'success': False,
                    'case_id': case_id,
                    'error': 'Approval pending'
                }
        except Exception as e:
            logger.error(f"Approval pending error handling failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _get_approval_pending_recommendations(self) -> List[str]:
        """Get approval pending recommendations"""
        return [
            'Wait for nominee approval',
            'Send reminder to nominee',
            'Check approval status regularly',
            'Follow up with nominee',
            'Provide additional context if needed',
            'Escalate if approval is delayed'
        ]
    
    # ========================================================================
    # INSUFFICIENT CONSENT LEVEL ERRORS
    # ========================================================================
    
    def handle_insufficient_consent_level_error(self, case_id: str,
                                               current_level: str,
                                               required_level: str,
                                               module: str) -> Dict[str, Any]:
        """Handle insufficient consent level error"""
        try:
            logger.warning(f"Insufficient consent level for {module}: {current_level} < {required_level}")
            
            if self.available:
                error_result = self.error_system.handle_error(
                    error=Exception(f"Insufficient consent level: {current_level} < {required_level}")
                )
                
                self.consent_errors.append({
                    'case_id': case_id,
                    'module': module,
                    'current_level': current_level,
                    'required_level': required_level,
                    'error_type': 'InsufficientConsentLevel',
                    'error_class': error_result['error_info']['type'],
                    'message': f'Insufficient consent level for {module}',
                    'status': 'handled',
                    'timestamp': datetime.now().isoformat()
                })
                
                return {
                    'success': False,
                    'case_id': case_id,
                    'module': module,
                    'current_level': current_level,
                    'required_level': required_level,
                    'error': f'Insufficient consent level for {module}',
                    'error_type': error_result['error_info']['type'],
                    'recovery': error_result['rectification'],
                    'recommendations': self._get_insufficient_consent_recommendations(module),
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'success': False,
                    'case_id': case_id,
                    'error': f'Insufficient consent level for {module}'
                }
        except Exception as e:
            logger.error(f"Insufficient consent level error handling failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _get_insufficient_consent_recommendations(self, module: str) -> List[str]:
        """Get insufficient consent recommendations"""
        return [
            f'Request higher consent level for {module}',
            'Explain why higher consent is needed',
            'Provide additional context to nominee',
            'Get supervisor approval if needed',
            'Document consent escalation',
            'Follow up with nominee'
        ]
    
    # ========================================================================
    # CONSENT VERIFICATION ERRORS
    # ========================================================================
    
    def handle_consent_verification_error(self, case_id: str,
                                         error: Exception) -> Dict[str, Any]:
        """Handle consent verification error"""
        try:
            logger.error(f"Consent verification error for case {case_id}: {str(error)}")
            
            if self.available:
                error_result = self.error_system.handle_error(error=error)
                
                self.consent_errors.append({
                    'case_id': case_id,
                    'error_type': 'ConsentVerificationError',
                    'error_class': error_result['error_info']['type'],
                    'message': error_result['error_info']['message'],
                    'status': 'handled',
                    'timestamp': datetime.now().isoformat()
                })
                
                return {
                    'success': False,
                    'case_id': case_id,
                    'error': 'Consent verification failed',
                    'error_type': error_result['error_info']['type'],
                    'recovery': error_result['rectification'],
                    'recommendations': self._get_verification_recommendations(),
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'success': False,
                    'case_id': case_id,
                    'error': 'Consent verification failed'
                }
        except Exception as e:
            logger.error(f"Consent verification error handling failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _get_verification_recommendations(self) -> List[str]:
        """Get verification recommendations"""
        return [
            'Verify consent details',
            'Check approval link validity',
            'Validate nominee identity',
            'Request new consent if needed',
            'Check consent expiration',
            'Review consent history'
        ]
    
    # ========================================================================
    # CONSENT EXPIRATION ERRORS
    # ========================================================================
    
    def handle_consent_expired_error(self, case_id: str,
                                    expiry_date: str) -> Dict[str, Any]:
        """Handle consent expired error"""
        try:
            logger.warning(f"Consent expired for case {case_id} on {expiry_date}")
            
            if self.available:
                error_result = self.error_system.handle_error(
                    error=Exception(f"Consent expired on {expiry_date}")
                )
                
                self.consent_errors.append({
                    'case_id': case_id,
                    'expiry_date': expiry_date,
                    'error_type': 'ConsentExpired',
                    'error_class': error_result['error_info']['type'],
                    'message': 'Consent expired',
                    'status': 'handled',
                    'timestamp': datetime.now().isoformat()
                })
                
                return {
                    'success': False,
                    'case_id': case_id,
                    'expiry_date': expiry_date,
                    'error': 'Consent expired',
                    'error_type': error_result['error_info']['type'],
                    'recovery': error_result['rectification'],
                    'recommendations': self._get_expiration_recommendations(),
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'success': False,
                    'case_id': case_id,
                    'error': 'Consent expired'
                }
        except Exception as e:
            logger.error(f"Consent expiration error handling failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _get_expiration_recommendations(self) -> List[str]:
        """Get expiration recommendations"""
        return [
            'Request new consent from nominee',
            'Send new approval link',
            'Explain consent expiration',
            'Get fresh approval',
            'Update consent records',
            'Document consent renewal'
        ]
    
    # ========================================================================
    # CONSENT VALIDATION
    # ========================================================================
    
    def validate_consent(self, case_id: str, consent_level: str,
                        required_level: str) -> Dict[str, Any]:
        """Validate consent"""
        try:
            if not case_id:
                raise ValueError("Case ID is required")
            if not consent_level:
                raise ValueError("Consent level is required")
            if not required_level:
                raise ValueError("Required level is required")
            
            # Simple level comparison (STANDARD < LEGAL < FULL)
            level_order = {'STANDARD': 1, 'LEGAL': 2, 'FULL': 3}
            current = level_order.get(consent_level.upper(), 0)
            required = level_order.get(required_level.upper(), 0)
            
            if current >= required:
                return {
                    'valid': True,
                    'case_id': case_id,
                    'consent_level': consent_level,
                    'required_level': required_level,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'valid': False,
                    'case_id': case_id,
                    'error': f'Insufficient consent: {consent_level} < {required_level}',
                    'timestamp': datetime.now().isoformat()
                }
        except Exception as e:
            logger.error(f"Consent validation failed: {e}")
            return {
                'valid': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    # ========================================================================
    # LOGGING & STATISTICS
    # ========================================================================
    
    def get_consent_error_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get consent error log"""
        return self.consent_errors[-limit:]
    
    def get_consent_error_statistics(self) -> Dict[str, Any]:
        """Get consent error statistics"""
        if not self.consent_errors:
            return {}
        
        stats = {
            'total_errors': len(self.consent_errors),
            'by_type': {},
            'by_case': {},
            'timestamp': datetime.now().isoformat()
        }
        
        for error in self.consent_errors:
            error_type = error.get('error_type', 'unknown')
            case_id = error.get('case_id', 'unknown')
            
            stats['by_type'][error_type] = stats['by_type'].get(error_type, 0) + 1
            stats['by_case'][case_id] = stats['by_case'].get(case_id, 0) + 1
        
        return stats

# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

consent_error_handler = ConsentErrorHandler()

# ============================================================================
# HELPER FUNCTION
# ============================================================================

def get_consent_error_handler() -> ConsentErrorHandler:
    """Get consent error handler instance"""
    return consent_error_handler
