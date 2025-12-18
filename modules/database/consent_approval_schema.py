"""
Consent Approval Database Schema
Defines SQLAlchemy models for approval links, approvals, and history
"""

from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime, timedelta
import uuid

Base = declarative_base()


class ApprovalLink(Base):
    """Stores approval links for nominees"""
    __tablename__ = 'approval_links'
    
    id = Column(Integer, primary_key=True)
    case_id = Column(String(255), nullable=False, index=True)
    token = Column(String(255), unique=True, nullable=False, index=True)
    nominee_email = Column(String(255), nullable=False)
    consent_level = Column(String(50), nullable=False)  # STANDARD, LEGAL, FULL
    approval_method = Column(String(50))  # PIN, PATTERN, BIOMETRIC
    expires_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String(50), default='pending')  # pending, approved, expired, revoked
    
    # Relationships
    approvals = relationship('ConsentApproval', back_populates='approval_link')
    history = relationship('ApprovalHistory', back_populates='approval_link')
    
    def is_expired(self):
        """Check if link is expired"""
        return datetime.utcnow() > self.expires_at
    
    def is_valid(self):
        """Check if link is valid"""
        return self.status == 'pending' and not self.is_expired()
    
    @staticmethod
    def generate_token():
        """Generate unique token"""
        return str(uuid.uuid4())


class ConsentApproval(Base):
    """Stores consent approvals"""
    __tablename__ = 'consent_approvals'
    
    id = Column(Integer, primary_key=True)
    case_id = Column(String(255), nullable=False, index=True)
    nominee_email = Column(String(255), nullable=False)
    approval_link_id = Column(Integer, ForeignKey('approval_links.id'))
    consent_level = Column(String(50), nullable=False)  # STANDARD, LEGAL, FULL
    approval_method = Column(String(50))  # PIN, PATTERN, BIOMETRIC
    approved_at = Column(DateTime)
    approved_by = Column(String(255))
    status = Column(String(50), default='pending')  # pending, approved, rejected, revoked
    pin_hash = Column(String(255))  # Hashed PIN for verification
    pattern_hash = Column(String(255))  # Hashed pattern for verification
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    approval_link = relationship('ApprovalLink', back_populates='approvals')
    history = relationship('ApprovalHistory', back_populates='approval')
    
    def approve(self, approval_method, pin_code=None, pattern=None):
        """Mark as approved"""
        self.status = 'approved'
        self.approved_at = datetime.utcnow()
        self.approval_method = approval_method
        if pin_code:
            self.pin_hash = self._hash_value(pin_code)
        if pattern:
            self.pattern_hash = self._hash_value(pattern)
    
    @staticmethod
    def _hash_value(value):
        """Hash a value for storage"""
        import hashlib
        return hashlib.sha256(str(value).encode()).hexdigest()


class ApprovalHistory(Base):
    """Stores approval events for audit trail"""
    __tablename__ = 'approval_history'
    
    id = Column(Integer, primary_key=True)
    case_id = Column(String(255), nullable=False, index=True)
    approval_link_id = Column(Integer, ForeignKey('approval_links.id'))
    approval_id = Column(Integer, ForeignKey('consent_approvals.id'))
    action = Column(String(100), nullable=False)  # link_generated, link_accessed, approved, rejected, revoked
    details = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    user_email = Column(String(255))
    ip_address = Column(String(50))
    
    # Relationships
    approval_link = relationship('ApprovalLink', back_populates='history')
    approval = relationship('ConsentApproval', back_populates='history')


# Export for use in other modules
__all__ = ['Base', 'ApprovalLink', 'ConsentApproval', 'ApprovalHistory']
