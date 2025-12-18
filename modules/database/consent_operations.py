"""
Consent Approval Database Operations
Implements CRUD operations for approval links, approvals, and history
"""

from sqlalchemy.orm import Session
from sqlalchemy import and_, desc
from datetime import datetime, timedelta
from .consent_approval_schema import ApprovalLink, ConsentApproval, ApprovalHistory
import hashlib
import logging

logger = logging.getLogger(__name__)


class ConsentApprovalOperations:
    """Database operations for consent approvals"""
    
    def __init__(self, db_session: Session):
        """Initialize with database session"""
        self.db = db_session
    
    # ==================== APPROVAL LINK OPERATIONS ====================
    
    def create_approval_link(self, case_id: str, nominee_email: str, 
                            consent_level: str, approval_method: str = None,
                            expires_in_hours: int = 24) -> ApprovalLink:
        """
        Create a new approval link
        
        Args:
            case_id: Case ID
            nominee_email: Nominee email
            consent_level: STANDARD, LEGAL, or FULL
            approval_method: PIN, PATTERN, or BIOMETRIC
            expires_in_hours: Hours until link expires (default 24)
        
        Returns:
            ApprovalLink object
        """
        try:
            # Generate unique token
            token = ApprovalLink.generate_token()
            
            # Calculate expiration time
            expires_at = datetime.utcnow() + timedelta(hours=expires_in_hours)
            
            # Create approval link
            approval_link = ApprovalLink(
                case_id=case_id,
                token=token,
                nominee_email=nominee_email,
                consent_level=consent_level,
                approval_method=approval_method,
                expires_at=expires_at,
                status='pending'
            )
            
            # Save to database
            self.db.add(approval_link)
            self.db.commit()
            
            logger.info(f"Created approval link: {token} for case {case_id}")
            
            # Log event
            self.log_approval_event(
                case_id=case_id,
                approval_link_id=approval_link.id,
                action='link_generated',
                details=f'Approval link generated for {nominee_email}'
            )
            
            return approval_link
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error creating approval link: {str(e)}")
            raise
    
    def get_approval_link(self, token: str) -> ApprovalLink:
        """
        Get approval link by token
        
        Args:
            token: Approval token
        
        Returns:
            ApprovalLink object or None
        """
        try:
            approval_link = self.db.query(ApprovalLink).filter(
                ApprovalLink.token == token
            ).first()
            
            if approval_link:
                logger.info(f"Retrieved approval link: {token}")
            
            return approval_link
            
        except Exception as e:
            logger.error(f"Error getting approval link: {str(e)}")
            raise
    
    def get_approval_links_by_case(self, case_id: str) -> list:
        """
        Get all approval links for a case
        
        Args:
            case_id: Case ID
        
        Returns:
            List of ApprovalLink objects
        """
        try:
            links = self.db.query(ApprovalLink).filter(
                ApprovalLink.case_id == case_id
            ).all()
            
            logger.info(f"Retrieved {len(links)} approval links for case {case_id}")
            return links
            
        except Exception as e:
            logger.error(f"Error getting approval links: {str(e)}")
            raise
    
    def revoke_approval_link(self, token: str) -> bool:
        """
        Revoke an approval link
        
        Args:
            token: Approval token
        
        Returns:
            True if successful, False otherwise
        """
        try:
            approval_link = self.get_approval_link(token)
            
            if not approval_link:
                logger.warning(f"Approval link not found: {token}")
                return False
            
            approval_link.status = 'revoked'
            self.db.commit()
            
            logger.info(f"Revoked approval link: {token}")
            
            # Log event
            self.log_approval_event(
                case_id=approval_link.case_id,
                approval_link_id=approval_link.id,
                action='link_revoked',
                details='Approval link revoked'
            )
            
            return True
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error revoking approval link: {str(e)}")
            raise
    
    # ==================== CONSENT APPROVAL OPERATIONS ====================
    
    def approve_consent(self, token: str, approval_method: str, 
                       nominee_email: str, pin_code: str = None, 
                       pattern: str = None) -> ConsentApproval:
        """
        Record consent approval
        
        Args:
            token: Approval token
            approval_method: PIN, PATTERN, or BIOMETRIC
            nominee_email: Nominee email
            pin_code: PIN code (if PIN method)
            pattern: Pattern (if PATTERN method)
        
        Returns:
            ConsentApproval object
        """
        try:
            # Get approval link
            approval_link = self.get_approval_link(token)
            
            if not approval_link:
                raise ValueError(f"Approval link not found: {token}")
            
            # Check if link is valid
            if not approval_link.is_valid():
                raise ValueError(f"Approval link is not valid: {token}")
            
            # Create approval record
            approval = ConsentApproval(
                case_id=approval_link.case_id,
                nominee_email=nominee_email,
                approval_link_id=approval_link.id,
                consent_level=approval_link.consent_level,
                approval_method=approval_method,
                status='approved',
                approved_at=datetime.utcnow(),
                approved_by=nominee_email
            )
            
            # Hash PIN if provided
            if pin_code:
                approval.pin_hash = self._hash_value(pin_code)
            
            # Hash pattern if provided
            if pattern:
                approval.pattern_hash = self._hash_value(pattern)
            
            # Save to database
            self.db.add(approval)
            
            # Update approval link status
            approval_link.status = 'approved'
            
            self.db.commit()
            
            logger.info(f"Approved consent for case {approval_link.case_id}")
            
            # Log event
            self.log_approval_event(
                case_id=approval_link.case_id,
                approval_link_id=approval_link.id,
                approval_id=approval.id,
                action='approved',
                details=f'Consent approved via {approval_method}',
                user_email=nominee_email
            )
            
            return approval
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error approving consent: {str(e)}")
            raise
    
    def get_approval_status(self, case_id: str) -> dict:
        """
        Get current approval status for a case
        
        Args:
            case_id: Case ID
        
        Returns:
            Dictionary with approval status
        """
        try:
            approval = self.db.query(ConsentApproval).filter(
                ConsentApproval.case_id == case_id
            ).order_by(desc(ConsentApproval.created_at)).first()
            
            if not approval:
                return {
                    'case_id': case_id,
                    'status': 'pending',
                    'approved_at': None,
                    'consent_level': None
                }
            
            return {
                'case_id': case_id,
                'status': approval.status,
                'approved_at': approval.approved_at,
                'consent_level': approval.consent_level,
                'approval_method': approval.approval_method,
                'nominee_email': approval.nominee_email
            }
            
        except Exception as e:
            logger.error(f"Error getting approval status: {str(e)}")
            raise
    
    def revoke_approval(self, case_id: str) -> bool:
        """
        Revoke an approval
        
        Args:
            case_id: Case ID
        
        Returns:
            True if successful, False otherwise
        """
        try:
            approval = self.db.query(ConsentApproval).filter(
                ConsentApproval.case_id == case_id
            ).order_by(desc(ConsentApproval.created_at)).first()
            
            if not approval:
                logger.warning(f"Approval not found for case {case_id}")
                return False
            
            approval.status = 'revoked'
            self.db.commit()
            
            logger.info(f"Revoked approval for case {case_id}")
            
            # Log event
            self.log_approval_event(
                case_id=case_id,
                approval_id=approval.id,
                action='revoked',
                details='Approval revoked'
            )
            
            return True
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error revoking approval: {str(e)}")
            raise
    
    # ==================== APPROVAL HISTORY OPERATIONS ====================
    
    def log_approval_event(self, case_id: str, action: str, details: str = None,
                          approval_link_id: int = None, approval_id: int = None,
                          user_email: str = None, ip_address: str = None) -> ApprovalHistory:
        """
        Log approval event for audit trail
        
        Args:
            case_id: Case ID
            action: Event action
            details: Event details
            approval_link_id: Approval link ID
            approval_id: Approval ID
            user_email: User email
            ip_address: IP address
        
        Returns:
            ApprovalHistory object
        """
        try:
            history = ApprovalHistory(
                case_id=case_id,
                approval_link_id=approval_link_id,
                approval_id=approval_id,
                action=action,
                details=details,
                user_email=user_email,
                ip_address=ip_address,
                timestamp=datetime.utcnow()
            )
            
            self.db.add(history)
            self.db.commit()
            
            logger.info(f"Logged event: {action} for case {case_id}")
            
            return history
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error logging approval event: {str(e)}")
            raise
    
    def get_approval_history(self, case_id: str) -> list:
        """
        Get approval history for a case
        
        Args:
            case_id: Case ID
        
        Returns:
            List of ApprovalHistory objects
        """
        try:
            history = self.db.query(ApprovalHistory).filter(
                ApprovalHistory.case_id == case_id
            ).order_by(desc(ApprovalHistory.timestamp)).all()
            
            logger.info(f"Retrieved {len(history)} history records for case {case_id}")
            return history
            
        except Exception as e:
            logger.error(f"Error getting approval history: {str(e)}")
            raise
    
    # ==================== UTILITY METHODS ====================
    
    @staticmethod
    def _hash_value(value: str) -> str:
        """
        Hash a value for storage with fallback protocol
        
        Uses SHA256 as primary method, with fallback to MD5 if needed
        """
        try:
            # Primary: SHA256 (recommended)
            return hashlib.sha256(str(value).encode()).hexdigest()
        except Exception as e:
            logger.warning(f"SHA256 hashing failed: {str(e)}, falling back to MD5")
            try:
                # Fallback: MD5
                return hashlib.md5(str(value).encode()).hexdigest()
            except Exception as e2:
                logger.error(f"MD5 hashing also failed: {str(e2)}, using plain text")
                # Last resort: Return as-is (not recommended for production)
                return str(value)
    
    @staticmethod
    def verify_hash(value: str, hash_value: str) -> bool:
        """
        Verify a value against its hash
        
        Tries multiple hash methods to verify
        """
        if not hash_value or not value:
            return False
        
        try:
            # Try SHA256
            if hashlib.sha256(str(value).encode()).hexdigest() == hash_value:
                return True
        except Exception as e:
            logger.warning(f"SHA256 verification failed: {str(e)}")
        
        try:
            # Try MD5
            if hashlib.md5(str(value).encode()).hexdigest() == hash_value:
                return True
        except Exception as e:
            logger.warning(f"MD5 verification failed: {str(e)}")
        
        try:
            # Try plain text comparison (fallback)
            if str(value) == hash_value:
                return True
        except Exception as e:
            logger.warning(f"Plain text verification failed: {str(e)}")
        
        return False
    
    def close(self):
        """Close database session"""
        if self.db:
            self.db.close()


# Convenience functions for use in other modules

def get_db_session(database_url: str):
    """Get database session"""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    
    engine = create_engine(database_url)
    Session = sessionmaker(bind=engine)
    return Session()


def create_approval_link(database_url: str, case_id: str, nominee_email: str,
                        consent_level: str, approval_method: str = None,
                        expires_in_hours: int = 24) -> ApprovalLink:
    """Create approval link (convenience function)"""
    db = get_db_session(database_url)
    ops = ConsentApprovalOperations(db)
    try:
        return ops.create_approval_link(case_id, nominee_email, consent_level, 
                                       approval_method, expires_in_hours)
    finally:
        ops.close()


def approve_consent(database_url: str, token: str, approval_method: str,
                   nominee_email: str, pin_code: str = None,
                   pattern: str = None) -> ConsentApproval:
    """Approve consent (convenience function)"""
    db = get_db_session(database_url)
    ops = ConsentApprovalOperations(db)
    try:
        return ops.approve_consent(token, approval_method, nominee_email, 
                                  pin_code, pattern)
    finally:
        ops.close()


def get_approval_status(database_url: str, case_id: str) -> dict:
    """Get approval status (convenience function)"""
    db = get_db_session(database_url)
    ops = ConsentApprovalOperations(db)
    try:
        return ops.get_approval_status(case_id)
    finally:
        ops.close()


def get_approval_history(database_url: str, case_id: str) -> list:
    """Get approval history (convenience function)"""
    db = get_db_session(database_url)
    ops = ConsentApprovalOperations(db)
    try:
        return ops.get_approval_history(case_id)
    finally:
        ops.close()
