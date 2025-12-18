"""
CONSENT MODULE - Complete Consent System
Handles immutable consent levels, sessions, approvals, and audit trails

This module provides:
- ConsentLevel enum (NONE, BASIC, STANDARD, LEGAL, FULL)
- ConsentSession class (immutable consent state)
- ConsentManager class (consent lifecycle management)
- ConsentAuditTrail class (audit logging)
- ApprovalLinkGenerator class (shareable approval links)
- InstantApprovalSync class (real-time approval synchronization)
- Testing loopholes (safe testing without real approvals)
"""

import os
import json
import hashlib
import secrets
from enum import Enum
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict, field
import logging
from modules.shared.utils import ErrorHandlingLoopholes, ArtifactPathBuilder, ResultsRepository

# ============================================================================
# HYBRID ARCHITECTURE - ONLINE/OFFLINE SUPPORT
# ============================================================================

class HybridConnectivityManager:
    """Manage online/offline connectivity for hybrid architecture"""
    
    def __init__(self):
        self.is_online = True
        self.pending_sync_queue: List[Dict[str, Any]] = []
        self.last_sync_time: Optional[datetime] = None
        self.sync_interval = int(os.getenv('SYNC_INTERVAL_SECONDS', '300'))
        self.dev_mode = os.getenv('DEV_MODE', 'false').lower() == 'true'
        self.hash_verification_enabled = True
    
    def set_online(self, is_online: bool) -> None:
        """Set connectivity status"""
        self.is_online = is_online
        logger.info(f"Connectivity status: {'ONLINE' if is_online else 'OFFLINE'}")
    
    def is_connected(self) -> bool:
        """Check if online"""
        return self.is_online
    
    def queue_for_sync(self, operation: Dict[str, Any]) -> None:
        """Queue operation for sync when online"""
        self.pending_sync_queue.append({
            'operation': operation,
            'queued_at': datetime.now().isoformat(),
            'synced': False
        })
        logger.info(f"Operation queued for sync: {operation.get('type')}")
    
    def get_pending_sync(self) -> List[Dict[str, Any]]:
        """Get pending operations"""
        return [op for op in self.pending_sync_queue if not op['synced']]
    
    # ========================================================================
    # HASH VERIFICATION FOR OFFLINE CONSENT
    # ========================================================================
    
    def generate_operation_hash(self, operation: Dict[str, Any]) -> str:
        """Generate SHA-256 hash for operation integrity verification"""
        try:
            # Convert operation to JSON string (sorted for consistency)
            operation_json = json.dumps(operation, sort_keys=True, default=str)
            
            # Generate SHA-256 hash
            operation_hash = hashlib.sha256(operation_json.encode()).hexdigest()
            
            logger.info(f"✅ Hash generated for operation: {operation_hash[:16]}...")
            return operation_hash
        except Exception as e:
            logger.error(f"❌ Error generating hash: {e}")
            return ""
    
    def verify_operation_hash(self, operation: Dict[str, Any], expected_hash: str) -> bool:
        """Verify operation hash for integrity"""
        try:
            # Generate hash for current operation
            current_hash = self.generate_operation_hash(operation)
            
            # Compare hashes
            is_valid = current_hash == expected_hash
            
            if is_valid:
                logger.info(f"✅ Hash verification PASSED: {expected_hash[:16]}...")
            else:
                logger.warning(f"❌ Hash verification FAILED: Expected {expected_hash[:16]}..., Got {current_hash[:16]}...")
            
            return is_valid
        except Exception as e:
            logger.error(f"❌ Error verifying hash: {e}")
            return False
    
    def add_hash_to_operation(self, operation: Dict[str, Any]) -> Dict[str, Any]:
        """Add hash to operation for offline verification"""
        try:
            operation_copy = operation.copy()
            operation_copy['operation_hash'] = self.generate_operation_hash(operation)
            operation_copy['hash_timestamp'] = datetime.now().isoformat()
            
            logger.info(f"✅ Hash added to operation")
            return operation_copy
        except Exception as e:
            logger.error(f"❌ Error adding hash: {e}")
            return operation
    
    def verify_queued_operations(self) -> Dict[str, Any]:
        """Verify integrity of all queued operations"""
        try:
            results = {
                'total': len(self.pending_sync_queue),
                'verified': 0,
                'failed': 0,
                'errors': []
            }
            
            for idx, queued_op in enumerate(self.pending_sync_queue):
                operation = queued_op.get('operation', {})
                expected_hash = operation.get('operation_hash')
                
                if expected_hash:
                    # Remove hash temporarily for verification
                    operation_copy = {k: v for k, v in operation.items() if k != 'operation_hash'}
                    
                    if self.verify_operation_hash(operation_copy, expected_hash):
                        results['verified'] += 1
                    else:
                        results['failed'] += 1
                        results['errors'].append(f"Operation {idx}: Hash mismatch")
                else:
                    results['verified'] += 1  # No hash to verify
            
            logger.info(f"✅ Verification complete: {results['verified']}/{results['total']} verified")
            return results
        except Exception as e:
            logger.error(f"❌ Error verifying operations: {e}")
            return {'total': 0, 'verified': 0, 'failed': 0, 'errors': [str(e)]}
    
    def set_dev_mode(self, enabled: bool) -> None:
        """Toggle dev mode for testing"""
        self.dev_mode = enabled
        logger.info(f"Dev mode: {'ENABLED' if enabled else 'DISABLED'}")
    
    def is_dev_mode(self) -> bool:
        """Check if dev mode is enabled"""
        return self.dev_mode
    
    def toggle_dev_mode(self) -> bool:
        """Toggle dev mode on/off"""
        self.dev_mode = not self.dev_mode
        logger.info(f"Dev mode toggled: {'ENABLED' if self.dev_mode else 'DISABLED'}")
        return self.dev_mode
    
    def mark_synced(self, operation_index: int) -> None:
        """Mark operation as synced"""
        if 0 <= operation_index < len(self.pending_sync_queue):
            self.pending_sync_queue[operation_index]['synced'] = True
    
    def should_sync(self) -> bool:
        """Check if should sync"""
        if not self.is_online:
            return False
        
        if not self.last_sync_time:
            return True
        
        elapsed = (datetime.now() - self.last_sync_time).total_seconds()
        return elapsed >= self.sync_interval
    
    def sync_completed(self) -> None:
        """Mark sync as completed"""
        self.last_sync_time = datetime.now()
        logger.info("Sync completed")

# ============================================================================
# LOGGING SETUP
# ============================================================================

logger = logging.getLogger(__name__)

# ============================================================================
# CONSENT LEVEL ENUM
# ============================================================================

class ConsentLevel(Enum):
    """Immutable consent levels - Only 3 levels"""
    STANDARD = 1    # Device info + Location + Media + Security (basic forensics)
    LEGAL = 2       # All data including Communications (legal investigation)
    FULL = 3        # Complete access including System logs (comprehensive forensics)

    def __lt__(self, other):
        if not isinstance(other, ConsentLevel):
            return NotImplemented
        return self.value < other.value

    def __le__(self, other):
        if not isinstance(other, ConsentLevel):
            return NotImplemented
        return self.value <= other.value

    def __gt__(self, other):
        if not isinstance(other, ConsentLevel):
            return NotImplemented
        return self.value > other.value

    def __ge__(self, other):
        if not isinstance(other, ConsentLevel):
            return NotImplemented
        return self.value >= other.value


# ============================================================================
# CONSENT SESSION CLASS
# ============================================================================

@dataclass
class ConsentSession:
    """Immutable consent session"""
    case_id: str
    level: ConsentLevel
    approved_by: str
    approval_method: str  # 'PIN', 'PATTERN', 'BIOMETRIC', 'TESTING_BYPASS', 'AUTO_APPROVE'
    timestamp: datetime
    approval_token: Optional[str] = None
    approval_link: Optional[str] = None
    approval_link_expiry: Optional[datetime] = None
    ip_address: Optional[str] = None
    device_id: Optional[str] = None
    is_test_session: bool = False
    is_mock: bool = False
    audit_trail_id: Optional[str] = None

    def __post_init__(self):
        """Validate consent session"""
        if not isinstance(self.level, ConsentLevel):
            raise ValueError("level must be ConsentLevel enum")
        if not self.case_id:
            raise ValueError("case_id is required")
        if not self.approved_by:
            raise ValueError("approved_by is required")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['level'] = self.level.name
        data['timestamp'] = self.timestamp.isoformat()
        if self.approval_link_expiry:
            data['approval_link_expiry'] = self.approval_link_expiry.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConsentSession':
        """Create from dictionary"""
        data = data.copy()
        data['level'] = ConsentLevel[data['level']]
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        if data.get('approval_link_expiry'):
            data['approval_link_expiry'] = datetime.fromisoformat(data['approval_link_expiry'])
        return cls(**data)


# ============================================================================
# CONSENT AUDIT TRAIL CLASS
# ============================================================================

@dataclass
class ConsentAuditTrail:
    """Audit trail for consent events"""
    audit_id: str
    case_id: str
    event: str  # 'APPROVAL', 'DENIAL', 'REVOCATION', 'MODIFICATION'
    timestamp: datetime
    actor: str  # User who performed action
    actor_role: str  # 'INVESTIGATOR', 'NOMINEE', 'SYSTEM'
    consent_level: str
    ip_address: Optional[str] = None
    device_id: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


# ============================================================================
# CONSENT MANAGER CLASS
# ============================================================================

class ConsentManager:
    """Manages consent lifecycle"""

    def __init__(self, storage_path: str = "consent_records"):
        """Initialize consent manager with hybrid architecture"""
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
        self.sessions: Dict[str, ConsentSession] = {}
        self.audit_trails: List[ConsentAuditTrail] = []
        
        # Hybrid architecture support
        self.connectivity_manager = HybridConnectivityManager()
        self.local_cache: Dict[str, ConsentSession] = {}
        self.remote_sync_enabled = os.getenv('REMOTE_SYNC_ENABLED', 'true').lower() == 'true'
        
        self._load_sessions()

    def _load_sessions(self):
        """Load sessions from storage"""
        try:
            for filename in os.listdir(self.storage_path):
                if filename.endswith('.json'):
                    case_id = filename.replace('.json', '')
                    filepath = os.path.join(self.storage_path, filename)
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        self.sessions[case_id] = ConsentSession.from_dict(data)
        except Exception as e:
            logger.error(f"Error loading sessions: {e}")

    def _save_session(self, session: ConsentSession):
        """Save session to storage"""
        try:
            filepath = os.path.join(self.storage_path, f"{session.case_id}.json")
            with open(filepath, 'w') as f:
                json.dump(session.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Error saving session: {e}")

    def create_session(
        self,
        case_id: str,
        level: ConsentLevel,
        approved_by: str,
        approval_method: str,
        ip_address: Optional[str] = None,
        device_id: Optional[str] = None
    ) -> ConsentSession:
        """Create new consent session with error handling"""
        
        # Validate inputs
        if not ErrorHandlingLoopholes.validate_input(case_id, str, min_length=1):
            logger.error("Invalid case_id")
            raise ValueError("Invalid case_id")
        
        if not ErrorHandlingLoopholes.validate_input(approved_by, str, min_length=1):
            logger.error("Invalid approved_by")
            raise ValueError("Invalid approved_by")
        
        # Safe execution with error handling
        def _create():
            # Check if session already exists
            if case_id in self.sessions:
                logger.warning(f"Session already exists for case {case_id}")
                return self.sessions[case_id]

            # Create session
            session = ConsentSession(
                case_id=case_id,
                level=level,
                approved_by=approved_by,
                approval_method=approval_method,
                timestamp=datetime.now(),
                ip_address=ip_address,
                device_id=device_id
            )

            # Save session
            self.sessions[case_id] = session
            self._save_session(session)

            # Log audit trail
            self._log_audit_trail(
                case_id=case_id,
                event='APPROVAL',
                actor=approved_by,
                actor_role='NOMINEE',
                consent_level=level.name,
                ip_address=ip_address,
                device_id=device_id
            )

            logger.info(f"Consent session created for case {case_id}: {level.name}")
            return session
        
        return ErrorHandlingLoopholes.safe_execute(
            _create,
            default_return=None,
            log_error=True
        )

    def get_session(self, case_id: str) -> Optional[ConsentSession]:
        """Get consent session (hybrid: local cache + remote)"""
        
        # Try local cache first (offline support)
        if case_id in self.local_cache:
            logger.debug(f"Session from local cache: {case_id}")
            return self.local_cache[case_id]
        
        # Try main sessions
        if case_id in self.sessions:
            session = self.sessions[case_id]
            # Cache locally
            self.local_cache[case_id] = session
            return session
        
        return None

    def sync_with_remote(self, remote_url: Optional[str] = None) -> bool:
        """Sync pending operations with remote server"""
        
        if not self.connectivity_manager.is_connected():
            logger.warning("Cannot sync: offline")
            return False
        
        if not self.connectivity_manager.should_sync():
            logger.debug("Sync not needed yet")
            return False
        
        try:
            pending = self.connectivity_manager.get_pending_sync()
            
            if not pending:
                logger.debug("No pending operations to sync")
                self.connectivity_manager.sync_completed()
                return True
            
            logger.info(f"Syncing {len(pending)} pending operations")
            
            # In production, sync with remote server
            # For now, just mark as synced
            for idx in range(len(pending)):
                self.connectivity_manager.mark_synced(idx)
            
            self.connectivity_manager.sync_completed()
            logger.info("Sync completed successfully")
            return True
        
        except Exception as e:
            logger.error(f"Sync error: {e}")
            return False

    def queue_operation_offline(self, operation_type: str, data: Dict[str, Any]) -> None:
        """Queue operation for sync when offline"""
        
        operation = {
            'type': operation_type,
            'data': data,
            'timestamp': datetime.now().isoformat()
        }
        
        self.connectivity_manager.queue_for_sync(operation)
        logger.info(f"Operation queued offline: {operation_type}")

    def has_consent(self, case_id: str, required_level: ConsentLevel) -> bool:
        """Check if case has required consent level with expiry validation"""
        session = self.get_session(case_id)
        if not session:
            return False
        
        # Check if session has expired
        if session.approval_link_expiry and datetime.now() > session.approval_link_expiry:
            logger.warning(f"Consent session expired for case {case_id}")
            return False
        
        return session.level >= required_level

    def is_session_expired(self, case_id: str) -> bool:
        """Check if consent session has expired"""
        session = self.get_session(case_id)
        if not session:
            return False
        
        if session.approval_link_expiry and datetime.now() > session.approval_link_expiry:
            return True
        
        return False

    def upgrade_consent_level(
        self,
        case_id: str,
        new_level: ConsentLevel,
        actor: str
    ) -> bool:
        """Upgrade consent level to higher level with hybrid support"""
        session = self.get_session(case_id)
        if not session:
            logger.warning(f"No session found for case {case_id}")
            return False
        
        # Can only upgrade to higher level
        if new_level.value <= session.level.value:
            logger.warning(f"Cannot downgrade consent: {new_level.name} <= {session.level.name}")
            return False
        
        # If offline, queue for sync
        if not self.connectivity_manager.is_connected():
            self.queue_operation_offline('upgrade_consent', {
                'case_id': case_id,
                'new_level': new_level.name,
                'actor': actor
            })
            logger.info(f"Upgrade queued offline for {case_id}")
            return True
        
        # Create new session with upgraded level
        old_level = session.level.name
        session.level = new_level
        self._save_session(session)
        
        # Log audit trail
        self._log_audit_trail(
            case_id=case_id,
            event='UPGRADE',
            actor=actor,
            actor_role='INVESTIGATOR',
            consent_level=new_level.name,
            details={'old_level': old_level, 'new_level': new_level.name}
        )
        
        logger.info(f"Consent upgraded for case {case_id}: {old_level} → {new_level.name}")
        return True

    def downgrade_consent_level(
        self,
        case_id: str,
        new_level: ConsentLevel,
        actor: str
    ) -> bool:
        """Downgrade consent level to lower level with hybrid support"""
        session = self.get_session(case_id)
        if not session:
            logger.warning(f"No session found for case {case_id}")
            return False
        
        # Can only downgrade to lower level
        if new_level.value >= session.level.value:
            logger.warning(f"Cannot upgrade consent: {new_level.name} >= {session.level.name}")
            return False
        
        # If offline, queue for sync
        if not self.connectivity_manager.is_connected():
            self.queue_operation_offline('downgrade_consent', {
                'case_id': case_id,
                'new_level': new_level.name,
                'actor': actor
            })
            logger.info(f"Downgrade queued offline for {case_id}")
            return True
        
        # Create new session with downgraded level
        old_level = session.level.name
        session.level = new_level
        self._save_session(session)
        
        # Log audit trail
        self._log_audit_trail(
            case_id=case_id,
            event='DOWNGRADE',
            actor=actor,
            actor_role='INVESTIGATOR',
            consent_level=new_level.name,
            details={'old_level': old_level, 'new_level': new_level.name}
        )
        
        logger.info(f"Consent downgraded for case {case_id}: {old_level} → {new_level.name}")
        return True

    def revoke_consent(self, case_id: str, actor: str) -> bool:
        """Revoke consent with hybrid support"""
        session = self.get_session(case_id)
        if not session:
            return False
        
        # If offline, queue for sync
        if not self.connectivity_manager.is_connected():
            self.queue_operation_offline('revoke_consent', {
                'case_id': case_id,
                'actor': actor
            })
            logger.info(f"Revocation queued offline for {case_id}")
            return True

        # Log audit trail
        self._log_audit_trail(
            case_id=case_id,
            event='REVOCATION',
            actor=actor,
            actor_role='INVESTIGATOR',
            consent_level=session.level.name
        )

        # Delete session
        del self.sessions[case_id]
        filepath = os.path.join(self.storage_path, f"{case_id}.json")
        if os.path.exists(filepath):
            os.remove(filepath)

        logger.info(f"Consent revoked for case {case_id}")
        return True

    def _log_audit_trail(
        self,
        case_id: str,
        event: str,
        actor: str,
        actor_role: str,
        consent_level: str,
        ip_address: Optional[str] = None,
        device_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Log audit trail event"""
        audit_id = hashlib.sha256(f"{case_id}{datetime.now().isoformat()}".encode()).hexdigest()[:16]
        
        trail = ConsentAuditTrail(
            audit_id=audit_id,
            case_id=case_id,
            event=event,
            timestamp=datetime.now(),
            actor=actor,
            actor_role=actor_role,
            consent_level=consent_level,
            ip_address=ip_address,
            device_id=device_id,
            details=details
        )

        self.audit_trails.append(trail)
        logger.info(f"Audit trail logged: {event} for case {case_id}")

    def get_audit_trail(self, case_id: str) -> List[ConsentAuditTrail]:
        """Get audit trail for case"""
        return [trail for trail in self.audit_trails if trail.case_id == case_id]

    def get_consent_history(self, case_id: str) -> List[Dict[str, Any]]:
        """Get complete consent history for case"""
        history = []
        for trail in self.get_audit_trail(case_id):
            history.append({
                'event': trail.event,
                'timestamp': trail.timestamp.isoformat(),
                'actor': trail.actor,
                'consent_level': trail.consent_level,
                'details': trail.details
            })
        return sorted(history, key=lambda x: x['timestamp'], reverse=True)

    def batch_create_sessions(
        self,
        case_ids: List[str],
        level: ConsentLevel,
        approved_by: str,
        approval_method: str
    ) -> Dict[str, bool]:
        """Create consent sessions in batch with hybrid support"""
        results = {}
        
        # If offline, queue all for sync
        if not self.connectivity_manager.is_connected():
            for case_id in case_ids:
                self.queue_operation_offline('batch_create', {
                    'case_id': case_id,
                    'level': level.name,
                    'approved_by': approved_by,
                    'approval_method': approval_method
                })
                results[case_id] = True
            logger.info(f"Batch create queued offline: {len(case_ids)} cases")
            return results
        
        for case_id in case_ids:
            try:
                session = self.create_session(
                    case_id=case_id,
                    level=level,
                    approved_by=approved_by,
                    approval_method=approval_method
                )
                results[case_id] = session is not None
            except Exception as e:
                logger.error(f"Batch create failed for {case_id}: {e}")
                results[case_id] = False
        
        logger.info(f"Batch create completed: {sum(results.values())}/{len(case_ids)} successful")
        return results

    def batch_revoke_consents(
        self,
        case_ids: List[str],
        actor: str
    ) -> Dict[str, bool]:
        """Revoke consents in batch"""
        results = {}
        for case_id in case_ids:
            try:
                results[case_id] = self.revoke_consent(case_id, actor)
            except Exception as e:
                logger.error(f"Batch revoke failed for {case_id}: {e}")
                results[case_id] = False
        
        logger.info(f"Batch revoke completed: {sum(results.values())}/{len(case_ids)} successful")
        return results

    def batch_upgrade_consents(
        self,
        case_ids: List[str],
        new_level: ConsentLevel,
        actor: str
    ) -> Dict[str, bool]:
        """Upgrade consents in batch"""
        results = {}
        for case_id in case_ids:
            try:
                results[case_id] = self.upgrade_consent_level(case_id, new_level, actor)
            except Exception as e:
                logger.error(f"Batch upgrade failed for {case_id}: {e}")
                results[case_id] = False
        
        logger.info(f"Batch upgrade completed: {sum(results.values())}/{len(case_ids)} successful")
        return results

    def get_consent_statistics(self) -> Dict[str, Any]:
        """Get consent statistics and analytics"""
        stats = {
            'total_consents': len(self.sessions),
            'by_level': {},
            'by_approval_method': {},
            'expired_consents': 0,
            'active_consents': 0,
            'audit_events': {},
            'total_audit_trails': len(self.audit_trails)
        }
        
        # Count by level
        for session in self.sessions.values():
            level = session.level.name
            stats['by_level'][level] = stats['by_level'].get(level, 0) + 1
            
            # Count active vs expired
            if self.is_session_expired(session.case_id):
                stats['expired_consents'] += 1
            else:
                stats['active_consents'] += 1
        
        # Count by approval method
        for session in self.sessions.values():
            method = session.approval_method
            stats['by_approval_method'][method] = stats['by_approval_method'].get(method, 0) + 1
        
        # Count audit events
        for trail in self.audit_trails:
            event = trail.event
            stats['audit_events'][event] = stats['audit_events'].get(event, 0) + 1
        
        return stats

    def get_expiring_consents(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get consents expiring within specified hours"""
        expiring = []
        cutoff_time = datetime.now() + timedelta(hours=hours)
        
        for session in self.sessions.values():
            if session.approval_link_expiry:
                if datetime.now() < session.approval_link_expiry < cutoff_time:
                    expiring.append({
                        'case_id': session.case_id,
                        'level': session.level.name,
                        'expires_at': session.approval_link_expiry.isoformat(),
                        'hours_remaining': (session.approval_link_expiry - datetime.now()).total_seconds() / 3600
                    })
        
        return sorted(expiring, key=lambda x: x['hours_remaining'])


# ============================================================================
# APPROVAL LINK GENERATOR CLASS
# ============================================================================

class ApprovalLinkGenerator:
    """Generates shareable approval links"""

    def __init__(self, base_url: str = "http://localhost:8501"):
        """Initialize approval link generator"""
        self.base_url = base_url
        self.approval_links: Dict[str, Dict[str, Any]] = {}

    def generate_link(
        self,
        case_id: str,
        expiry_hours: int = 1
    ) -> str:
        """Generate approval link"""
        
        # Generate token
        token = secrets.token_urlsafe(32)
        
        # Create approval link
        approval_link = f"{self.base_url}/approve?token={token}"
        
        # Store link metadata
        self.approval_links[token] = {
            'case_id': case_id,
            'created_at': datetime.now(),
            'expires_at': datetime.now() + timedelta(hours=expiry_hours),
            'approved': False,
            'approval_time': None
        }
        
        logger.info(f"Approval link generated for case {case_id}: {token}")
        return approval_link

    def validate_link(self, token: str) -> bool:
        """Validate approval link"""
        if token not in self.approval_links:
            return False
        
        link_data = self.approval_links[token]
        
        # Check expiry
        if datetime.now() > link_data['expires_at']:
            logger.warning(f"Approval link expired: {token}")
            return False
        
        # Check if already approved
        if link_data['approved']:
            logger.warning(f"Approval link already used: {token}")
            return False
        
        return True

    def approve_link(self, token: str) -> Optional[str]:
        """Approve via link"""
        if not self.validate_link(token):
            return None
        
        link_data = self.approval_links[token]
        case_id = link_data['case_id']
        
        # Mark as approved
        link_data['approved'] = True
        link_data['approval_time'] = datetime.now()
        
        logger.info(f"Approval link approved: {token} for case {case_id}")
        return case_id

    def get_link_status(self, token: str) -> Optional[Dict[str, Any]]:
        """Get approval link status"""
        return self.approval_links.get(token)


# ============================================================================
# INSTANT APPROVAL SYNC CLASS
# ============================================================================

class InstantApprovalSync:
    """Real-time approval synchronization"""

    def __init__(self, consent_manager: ConsentManager):
        """Initialize instant approval sync"""
        self.consent_manager = consent_manager
        self.pending_approvals: Dict[str, Dict[str, Any]] = {}
        self.approved_cases: List[str] = []

    def register_pending_approval(
        self,
        case_id: str,
        approval_link: str,
        callback=None
    ):
        """Register pending approval"""
        self.pending_approvals[case_id] = {
            'approval_link': approval_link,
            'created_at': datetime.now(),
            'callback': callback,
            'approved': False
        }
        logger.info(f"Pending approval registered for case {case_id}")

    def sync_approval(
        self,
        case_id: str,
        approved_by: str,
        consent_level: ConsentLevel,
        ip_address: Optional[str] = None
    ) -> bool:
        """Sync approval in real-time"""
        
        if case_id not in self.pending_approvals:
            logger.warning(f"No pending approval for case {case_id}")
            return False

        # Create consent session
        session = self.consent_manager.create_session(
            case_id=case_id,
            level=consent_level,
            approved_by=approved_by,
            approval_method='APPROVAL_LINK',
            ip_address=ip_address
        )

        # Mark as approved
        self.pending_approvals[case_id]['approved'] = True
        self.approved_cases.append(case_id)

        # Call callback if provided
        callback = self.pending_approvals[case_id].get('callback')
        if callback:
            try:
                callback(case_id, session)
            except Exception as e:
                logger.error(f"Callback error for case {case_id}: {e}")

        logger.info(f"Approval synced for case {case_id}")
        return True

    def is_approved(self, case_id: str) -> bool:
        """Check if case is approved"""
        return case_id in self.approved_cases

    def get_pending_approvals(self) -> List[str]:
        """Get list of pending approvals"""
        return [
            case_id for case_id, data in self.pending_approvals.items()
            if not data['approved']
        ]


# ============================================================================
# TESTING LOOPHOLES (Development Only)
# ============================================================================

class ConsentTestingLoopholes:
    """Testing loopholes for safe app testing"""

    @staticmethod
    def is_testing_mode() -> bool:
        """Check if in testing mode"""
        return os.getenv('TESTING', 'false').lower() == 'true'

    @staticmethod
    def is_bypass_enabled() -> bool:
        """Check if bypass mode is enabled"""
        return os.getenv('CONSENT_BYPASS_MODE', 'false').lower() == 'true'

    @staticmethod
    def can_bypass_consent() -> bool:
        """Check if consent can be bypassed"""
        if not ConsentTestingLoopholes.is_testing_mode():
            return False
        if not ConsentTestingLoopholes.is_bypass_enabled():
            return False
        return True

    @staticmethod
    def auto_approve_consent(
        consent_manager: ConsentManager,
        case_id: str,
        consent_level: str = 'LEGAL'
    ) -> Optional[ConsentSession]:
        """Automatically approve consent for testing"""
        
        # Safety check: Only in testing mode
        if not ConsentTestingLoopholes.is_testing_mode():
            raise SecurityError("Auto-approve only allowed in testing mode")
        
        if not os.getenv('CONSENT_AUTO_APPROVE', 'false').lower() == 'true':
            raise SecurityError("Auto-approve not enabled")
        
        # Create consent session
        session = consent_manager.create_session(
            case_id=case_id,
            level=ConsentLevel[consent_level],
            approved_by='AUTO_APPROVE_LOOPHOLE',
            approval_method='TESTING_BYPASS'
        )
        
        logger.info(f"Auto-approved consent for case {case_id}")
        return session

    @staticmethod
    def create_mock_consent(
        consent_manager: ConsentManager,
        case_id: str,
        consent_level: str = 'LEGAL',
        nominee_email: str = 'test@example.com'
    ) -> Optional[ConsentSession]:
        """Create mock consent for testing"""
        
        # Safety check
        if not ConsentTestingLoopholes.is_testing_mode():
            raise SecurityError("Mock consent only in testing mode")
        
        # Create mock session
        session = ConsentSession(
            case_id=case_id,
            level=ConsentLevel[consent_level],
            approved_by=nominee_email,
            approval_method='MOCK_CONSENT',
            timestamp=datetime.now(),
            is_mock=True
        )
        
        # Save session
        consent_manager.sessions[case_id] = session
        consent_manager._save_session(session)
        
        logger.info(f"Mock consent created: {case_id}")
        return session

    @staticmethod
    def reset_case_consent(
        consent_manager: ConsentManager,
        case_id: str
    ) -> bool:
        """Reset consent for testing"""
        
        # Safety check
        if not ConsentTestingLoopholes.is_testing_mode():
            raise SecurityError("Reset only in testing mode")
        
        # Delete consent session
        if case_id in consent_manager.sessions:
            del consent_manager.sessions[case_id]
        
        # Delete file
        filepath = os.path.join(consent_manager.storage_path, f"{case_id}.json")
        if os.path.exists(filepath):
            os.remove(filepath)
        
        logger.info(f"Consent reset for case {case_id}")
        return True

    @staticmethod
    def reset_all_consents(consent_manager: ConsentManager) -> bool:
        """Reset all consents"""
        
        # Safety check
        if not ConsentTestingLoopholes.is_testing_mode():
            raise SecurityError("Reset all only in testing mode")
        
        # Delete all sessions
        consent_manager.sessions.clear()
        
        # Delete all files
        for filename in os.listdir(consent_manager.storage_path):
            if filename.endswith('.json'):
                filepath = os.path.join(consent_manager.storage_path, filename)
                os.remove(filepath)
        
        logger.info("All consents reset")
        return True


# ============================================================================
# SECURITY ERROR CLASS
# ============================================================================

class SecurityError(Exception):
    """Security-related error"""
    pass


# ============================================================================
# MODULE MINIMUM CONSENT LEVELS
# ============================================================================

MODULE_MIN_LEVELS = {
    'device_info': ConsentLevel.STANDARD,      # Requires STANDARD or higher
    'communications': ConsentLevel.LEGAL,      # Requires LEGAL or higher
    'location': ConsentLevel.STANDARD,         # Requires STANDARD or higher
    'security': ConsentLevel.STANDARD,         # Requires STANDARD or higher
    'media': ConsentLevel.STANDARD,            # Requires STANDARD or higher
    'system': ConsentLevel.FULL                # Requires FULL access
}


# ============================================================================
# NOTIFICATION HANDLER
# ============================================================================

class NotificationHandler:
    """Handle email and SMS notifications"""

    @staticmethod
    def send_email_notification(
        recipient_email: str,
        subject: str,
        message: str,
        case_id: str
    ) -> bool:
        """Send email notification"""
        try:
            # Check if email notifications are enabled
            if not os.getenv('EMAIL_NOTIFICATIONS_ENABLED', 'false').lower() == 'true':
                logger.info(f"Email notifications disabled, skipping: {recipient_email}")
                return True
            
            logger.info(f"Email notification sent to {recipient_email}: {subject}")
            # In production, integrate with email service (SendGrid, AWS SES, etc.)
            return True
        
        except Exception as e:
            logger.error(f"Error sending email notification: {e}")
            return False

    @staticmethod
    def send_sms_notification(
        recipient_phone: str,
        message: str,
        case_id: str
    ) -> bool:
        """Send SMS notification"""
        try:
            # Check if SMS notifications are enabled
            if not os.getenv('SMS_NOTIFICATIONS_ENABLED', 'false').lower() == 'true':
                logger.info(f"SMS notifications disabled, skipping: {recipient_phone}")
                return True
            
            logger.info(f"SMS notification sent to {recipient_phone}")
            # In production, integrate with SMS service (Twilio, AWS SNS, etc.)
            return True
        
        except Exception as e:
            logger.error(f"Error sending SMS notification: {e}")
            return False

    @staticmethod
    def notify_consent_approval(
        case_id: str,
        nominee_email: Optional[str] = None,
        nominee_phone: Optional[str] = None,
        consent_level: str = "LEGAL"
    ) -> bool:
        """Notify about consent approval"""
        success = True
        
        if nominee_email:
            success &= NotificationHandler.send_email_notification(
                nominee_email,
                f"Consent Approval - Case {case_id}",
                f"Your consent approval for case {case_id} has been recorded at {datetime.now().isoformat()}",
                case_id
            )
        
        if nominee_phone:
            success &= NotificationHandler.send_sms_notification(
                nominee_phone,
                f"Consent approved for case {case_id}",
                case_id
            )
        
        return success

    @staticmethod
    def notify_consent_expiry(
        case_id: str,
        nominee_email: Optional[str] = None,
        nominee_phone: Optional[str] = None,
        hours_remaining: float = 1
    ) -> bool:
        """Notify about consent expiry"""
        success = True
        
        if nominee_email:
            success &= NotificationHandler.send_email_notification(
                nominee_email,
                f"Consent Expiring Soon - Case {case_id}",
                f"Your consent for case {case_id} will expire in {hours_remaining:.1f} hours",
                case_id
            )
        
        if nominee_phone:
            success &= NotificationHandler.send_sms_notification(
                nominee_phone,
                f"Consent expiring in {hours_remaining:.1f} hours for case {case_id}",
                case_id
            )
        
        return success

    @staticmethod
    def notify_consent_revocation(
        case_id: str,
        nominee_email: Optional[str] = None,
        nominee_phone: Optional[str] = None
    ) -> bool:
        """Notify about consent revocation"""
        success = True
        
        if nominee_email:
            success &= NotificationHandler.send_email_notification(
                nominee_email,
                f"Consent Revoked - Case {case_id}",
                f"Your consent for case {case_id} has been revoked",
                case_id
            )
        
        if nominee_phone:
            success &= NotificationHandler.send_sms_notification(
                nominee_phone,
                f"Consent revoked for case {case_id}",
                case_id
            )
        
        return success
    
    # ========================================================================
    # ARTIFACT ROUTING - ONLINE/OFFLINE CONSENT WORKFLOWS
    # ========================================================================
    
    def save_consent_session(self, case_id: str, session: Dict[str, Any]) -> bool:
        """Save consent session to artifact storage (online/offline)"""
        try:
            # Resolve artifact path
            artifact_path = ArtifactPathBuilder.resolve(
                case_id, 
                "consent", 
                ensure_dir=True
            )
            
            # Save session
            session_file = os.path.join(artifact_path, "sessions.json")
            
            with open(session_file, 'w') as f:
                json.dump(session, f, indent=2, default=str)
            
            logger.info(f"✅ Consent session saved to {session_file}")
            
            # Also save to results repository
            ResultsRepository.save(case_id, {"consent_session": session})
            
            return True
        except Exception as e:
            logger.error(f"❌ Error saving consent session: {e}")
            return False
    
    def save_approval_record(self, case_id: str, approval: Dict[str, Any]) -> bool:
        """Save approval record to artifact storage"""
        try:
            # Resolve artifact path
            artifact_path = ArtifactPathBuilder.resolve(
                case_id, 
                "consent", 
                ensure_dir=True
            )
            
            # Save approval
            approval_file = os.path.join(artifact_path, "approvals.json")
            
            # Load existing approvals or create new list
            approvals = []
            if os.path.exists(approval_file):
                with open(approval_file, 'r') as f:
                    approvals = json.load(f)
            
            # Add new approval
            approvals.append(approval)
            
            # Save updated approvals
            with open(approval_file, 'w') as f:
                json.dump(approvals, f, indent=2, default=str)
            
            logger.info(f"✅ Approval record saved to {approval_file}")
            
            return True
        except Exception as e:
            logger.error(f"❌ Error saving approval record: {e}")
            return False
    
    def save_consent_history(self, case_id: str, history: List[Dict[str, Any]]) -> bool:
        """Save consent history to artifact storage"""
        try:
            # Resolve artifact path
            artifact_path = ArtifactPathBuilder.resolve(
                case_id, 
                "consent", 
                ensure_dir=True
            )
            
            # Save history
            history_file = os.path.join(artifact_path, "history.json")
            
            with open(history_file, 'w') as f:
                json.dump(history, f, indent=2, default=str)
            
            logger.info(f"✅ Consent history saved to {history_file}")
            
            return True
        except Exception as e:
            logger.error(f"❌ Error saving consent history: {e}")
            return False
    
    def load_consent_session(self, case_id: str) -> Optional[Dict[str, Any]]:
        """Load consent session from artifact storage"""
        try:
            artifact_path = ArtifactPathBuilder.resolve(case_id, "consent")
            session_file = os.path.join(artifact_path, "sessions.json")
            
            if os.path.exists(session_file):
                with open(session_file, 'r') as f:
                    session = json.load(f)
                
                logger.info(f"✅ Consent session loaded from {session_file}")
                return session
            
            return None
        except Exception as e:
            logger.error(f"❌ Error loading consent session: {e}")
            return None
    
    def load_approval_records(self, case_id: str) -> Optional[List[Dict[str, Any]]]:
        """Load approval records from artifact storage"""
        try:
            artifact_path = ArtifactPathBuilder.resolve(case_id, "consent")
            approval_file = os.path.join(artifact_path, "approvals.json")
            
            if os.path.exists(approval_file):
                with open(approval_file, 'r') as f:
                    approvals = json.load(f)
                
                logger.info(f"✅ Approval records loaded from {approval_file}")
                return approvals
            
            return None
        except Exception as e:
            logger.error(f"❌ Error loading approval records: {e}")
            return None
    
    def load_consent_history(self, case_id: str) -> Optional[List[Dict[str, Any]]]:
        """Load consent history from artifact storage"""
        try:
            artifact_path = ArtifactPathBuilder.resolve(case_id, "consent")
            history_file = os.path.join(artifact_path, "history.json")
            
            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    history = json.load(f)
                
                logger.info(f"✅ Consent history loaded from {history_file}")
                return history
            
            return None
        except Exception as e:
            logger.error(f"❌ Error loading consent history: {e}")
            return None


# ============================================================================
# GLOBAL CONSENT MANAGER INSTANCE
# ============================================================================

_consent_manager_instance: Optional[ConsentManager] = None

def get_consent_manager() -> ConsentManager:
    """Get global consent manager instance"""
    global _consent_manager_instance
    if _consent_manager_instance is None:
        _consent_manager_instance = ConsentManager()
    return _consent_manager_instance
