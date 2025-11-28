"""
CONSENT APPROVAL SIGNATURE INTEGRATION - Integrates signatures into consent workflow

Provides:
- Signature-based approval workflow
- Legal binding creation
- Verification and audit trail
- Database integration
- API endpoints

This module integrates digital signatures as legal barriers in consent approval.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import json

logger = logging.getLogger(__name__)

# ============================================================================
# SIGNATURE-BASED CONSENT APPROVAL
# ============================================================================

class SignatureBasedConsentApproval:
    """Manages signature-based consent approvals with legal binding"""
    
    def __init__(self, signature_service=None, database_manager=None, api_client=None):
        """Initialize signature-based consent approval"""
        self.signature_service = signature_service
        self.database_manager = database_manager
        self.api_client = api_client
        self.approval_records: Dict[str, Dict[str, Any]] = {}
        self.legal_bindings: Dict[str, Dict[str, Any]] = {}
        logger.info("✅ SignatureBasedConsentApproval initialized")
    
    # ========================================================================
    # SIGNATURE APPROVAL WORKFLOW
    # ========================================================================
    
    def create_signature_approval(self, case_id: str, 
                                 nominee_email: str,
                                 consent_level: str,
                                 signature_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create signature-based approval
        
        Args:
            case_id: Case ID
            nominee_email: Nominee email
            consent_level: Consent level (STANDARD, LEGAL, FULL)
            signature_data: Digital signature data
            
        Returns:
            Approval result
        """
        try:
            # Validate signature legally
            if not self.signature_service:
                return {'status': 'error', 'error': 'Signature service not available'}
            
            is_compliant, issues = self.signature_service.validate_legal_requirements(signature_data)
            
            if not is_compliant:
                logger.warning(f"❌ Signature not legally compliant: {issues}")
                return {
                    'status': 'rejected',
                    'reason': 'Signature does not meet legal requirements',
                    'issues': issues,
                    'timestamp': datetime.now().isoformat()
                }
            
            # Get trusted timestamp
            timestamp_result = self.signature_service.get_trusted_timestamp()
            if timestamp_result.get('status') != 'success':
                return {'status': 'error', 'error': 'Failed to obtain timestamp'}
            
            timestamp_data = timestamp_result.get('timestamp', {})
            
            # Create approval record
            approval_id = f"APPROVAL-{case_id}-{datetime.now().timestamp()}"
            
            approval_record = {
                'approval_id': approval_id,
                'case_id': case_id,
                'nominee_email': nominee_email,
                'consent_level': consent_level,
                'approval_type': 'signature',
                'signature': signature_data,
                'timestamp': timestamp_data,
                'legal_status': 'binding',
                'status': 'approved',
                'created_at': datetime.now().isoformat(),
                'expires_at': (datetime.now() + timedelta(days=365)).isoformat()
            }
            
            # Store in memory
            self.approval_records[approval_id] = approval_record
            
            # Store in database
            if self.database_manager:
                try:
                    self.database_manager.create('signature_approvals', approval_record)
                except Exception as e:
                    logger.warning(f"⚠️ Database storage failed: {str(e)}")
            
            logger.info(f"✅ Signature approval created: {approval_id}")
            
            return {
                'status': 'approved',
                'approval_id': approval_id,
                'legal_status': 'binding',
                'timestamp': timestamp_data,
                'message': 'Approval legally binding with signature'
            }
            
        except Exception as e:
            logger.error(f"❌ Signature approval creation failed: {str(e)}")
            return {'status': 'error', 'error': str(e)}
    
    # ========================================================================
    # LEGAL BINDING CREATION
    # ========================================================================
    
    def create_legal_binding(self, approval_id: str) -> Dict[str, Any]:
        """
        Create legally binding record from approval
        
        Args:
            approval_id: Approval ID
            
        Returns:
            Legal binding result
        """
        try:
            # Retrieve approval
            if approval_id not in self.approval_records:
                return {'status': 'error', 'error': 'Approval not found'}
            
            approval = self.approval_records[approval_id]
            
            # Create legal binding
            binding_id = f"BINDING-{approval_id}"
            
            legal_binding = {
                'binding_id': binding_id,
                'approval_id': approval_id,
                'case_id': approval.get('case_id'),
                'nominee_email': approval.get('nominee_email'),
                'consent_level': approval.get('consent_level'),
                'signature_hash': approval.get('signature', {}).get('data_hash'),
                'legal_status': 'binding',
                'enforceability': 'legally_enforceable',
                'jurisdiction': 'India',
                'applicable_laws': [
                    'Information Technology Act, 2000',
                    'Indian Evidence Act, 1872',
                    'Indian Penal Code'
                ],
                'created_at': datetime.now().isoformat(),
                'expires_at': (datetime.now() + timedelta(days=365)).isoformat(),
                'status': 'active'
            }
            
            # Store in memory
            self.legal_bindings[binding_id] = legal_binding
            
            # Store in database
            if self.database_manager:
                try:
                    self.database_manager.create('legal_bindings', legal_binding)
                except Exception as e:
                    logger.warning(f"⚠️ Database storage failed: {str(e)}")
            
            logger.info(f"✅ Legal binding created: {binding_id}")
            
            return {
                'status': 'success',
                'binding_id': binding_id,
                'enforceability': 'legally_enforceable',
                'message': 'Legally binding record created'
            }
            
        except Exception as e:
            logger.error(f"❌ Legal binding creation failed: {str(e)}")
            return {'status': 'error', 'error': str(e)}
    
    # ========================================================================
    # SIGNATURE VERIFICATION
    # ========================================================================
    
    def verify_approval_signature(self, approval_id: str, 
                                 secret_key: str) -> Tuple[bool, str]:
        """
        Verify signature of approval
        
        Args:
            approval_id: Approval ID
            secret_key: Secret key for verification
            
        Returns:
            (is_valid, message)
        """
        try:
            # Retrieve approval
            if approval_id not in self.approval_records:
                return False, "Approval not found"
            
            approval = self.approval_records[approval_id]
            signature_data = approval.get('signature', {})
            
            # Verify signature
            if not self.signature_service:
                return False, "Signature service not available"
            
            # Create verification data
            verification_data = {
                'case_id': approval.get('case_id'),
                'nominee_email': approval.get('nominee_email'),
                'consent_level': approval.get('consent_level')
            }
            
            is_valid, message = self.signature_service.verify_signature(
                verification_data,
                signature_data.get('signature_id'),
                secret_key
            )
            
            if is_valid:
                logger.info(f"✅ Approval signature verified: {approval_id}")
            else:
                logger.warning(f"❌ Approval signature verification failed: {approval_id}")
            
            return is_valid, message
            
        except Exception as e:
            logger.error(f"❌ Verification failed: {str(e)}")
            return False, str(e)
    
    # ========================================================================
    # AUDIT TRAIL
    # ========================================================================
    
    def get_approval_audit_trail(self, approval_id: str) -> Dict[str, Any]:
        """
        Get audit trail for approval
        
        Args:
            approval_id: Approval ID
            
        Returns:
            Audit trail information
        """
        try:
            # Retrieve approval
            if approval_id not in self.approval_records:
                return {'status': 'error', 'error': 'Approval not found'}
            
            approval = self.approval_records[approval_id]
            signature_data = approval.get('signature', {})
            
            # Get signature audit trail
            sig_audit = {}
            if self.signature_service and signature_data.get('signature_id'):
                sig_audit = self.signature_service.get_audit_trail(
                    signature_data.get('signature_id')
                ).get('audit_trail', {})
            
            audit_trail = {
                'approval_id': approval_id,
                'case_id': approval.get('case_id'),
                'created_at': approval.get('created_at'),
                'status': approval.get('status'),
                'legal_status': approval.get('legal_status'),
                'signature_audit': sig_audit,
                'verification_count': len(sig_audit.get('verifications', [])),
                'last_verified': sig_audit.get('verifications', [{}])[-1].get('verified_at') if sig_audit.get('verifications') else None
            }
            
            logger.info(f"✅ Audit trail retrieved: {approval_id}")
            return {'status': 'success', 'audit_trail': audit_trail}
            
        except Exception as e:
            logger.error(f"❌ Audit trail retrieval failed: {str(e)}")
            return {'status': 'error', 'error': str(e)}
    
    # ========================================================================
    # COMPLIANCE CHECKING
    # ========================================================================
    
    def check_legal_compliance(self, approval_id: str) -> Dict[str, Any]:
        """
        Check legal compliance of approval
        
        Args:
            approval_id: Approval ID
            
        Returns:
            Compliance check result
        """
        try:
            # Retrieve approval
            if approval_id not in self.approval_records:
                return {'status': 'error', 'error': 'Approval not found'}
            
            approval = self.approval_records[approval_id]
            signature_data = approval.get('signature', {})
            
            # Check compliance
            if not self.signature_service:
                return {'status': 'error', 'error': 'Signature service not available'}
            
            is_compliant, issues = self.signature_service.validate_legal_requirements(signature_data)
            
            compliance_result = {
                'approval_id': approval_id,
                'is_compliant': is_compliant,
                'issues': issues,
                'legal_status': approval.get('legal_status'),
                'enforceability': 'legally_enforceable' if is_compliant else 'not_enforceable',
                'checked_at': datetime.now().isoformat()
            }
            
            if is_compliant:
                logger.info(f"✅ Approval is legally compliant: {approval_id}")
            else:
                logger.warning(f"⚠️ Approval has compliance issues: {approval_id}")
            
            return {'status': 'success', 'compliance': compliance_result}
            
        except Exception as e:
            logger.error(f"❌ Compliance check failed: {str(e)}")
            return {'status': 'error', 'error': str(e)}
    
    # ========================================================================
    # STATISTICS & REPORTING
    # ========================================================================
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get signature approval statistics"""
        
        total_approvals = len(self.approval_records)
        approved_count = sum(1 for a in self.approval_records.values() if a.get('status') == 'approved')
        rejected_count = sum(1 for a in self.approval_records.values() if a.get('status') == 'rejected')
        
        return {
            'total_approvals': total_approvals,
            'approved': approved_count,
            'rejected': rejected_count,
            'legal_bindings': len(self.legal_bindings),
            'approval_rate': (approved_count / total_approvals * 100) if total_approvals > 0 else 0
        }

# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_signature_based_consent_approval(signature_service=None, 
                                          database_manager=None,
                                          api_client=None) -> SignatureBasedConsentApproval:
    """Factory function to create signature-based consent approval"""
    return SignatureBasedConsentApproval(signature_service, database_manager, api_client)
