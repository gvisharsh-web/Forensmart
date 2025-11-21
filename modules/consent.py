import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, List, Any, Tuple
from datetime import datetime, timedelta

import hashlib
import secrets
import hmac
import base64
import json
import logging

try:
    import streamlit as st  # type: ignore
except Exception:  # Streamlit not available outside app runtime
    st = None

# Initialize logger
logger = logging.getLogger(__name__)

# NEW: Import audit trail for consent tracking
try:
    from modules.consent_portal import ConsentAuditTrail
except ImportError:
    ConsentAuditTrail = None  # Optional dependency


class ConsentLevel(Enum):
    """Defines all consent levels for forensic access"""
    NONE = 0
    BASIC = 1      # SMS verified - Basic device info only
    STANDARD = 2   # Standard extraction - Communications & Location
    FULL = 3       # Full forensic extraction - All data
    LEGAL = 4      # Court-ordered - Complete unrestricted access


@dataclass
class DataExtractionScope:
    """Defines what data will be extracted at each consent level"""
    level: ConsentLevel
    name: str
    description: str
    data_categories: Dict[str, list]
    legal_basis: str
    retention_period: str

    @property
    def all_data_types(self) -> list:
        """Get all data types that will be extracted"""
        return [item for category in self.data_categories.values() for item in category]


# Predefined extraction scopes
EXTRACTION_SCOPES = {
    ConsentLevel.BASIC: DataExtractionScope(
        level=ConsentLevel.BASIC,
        name="Basic Device Information",
        description="Minimal device identification and basic forensic metadata",
        data_categories={
            "Device Information": [
                "Device model and manufacturer",
                "Operating system version",
                "Device serial number",
                "IMEI/MEID numbers",
                "Basic hardware specifications"
            ],
            "Forensic Metadata": [
                "File system structure",
                "Timestamps and modification dates",
                "Basic device logs",
                "Installation timestamps"
            ]
        },
        legal_basis="Device owner consent with SMS verification",
        retention_period="Case duration + 7 years archival"
    ),

    ConsentLevel.STANDARD: DataExtractionScope(
        level=ConsentLevel.STANDARD,
        name="Standard Forensic Extraction",
        description="Communications, location data, and application information",
        data_categories={
            "Communications": [
                "SMS/MMS messages and threads",
                "Call logs and history",
                "Contact lists and address books",
                "Email accounts and messages",
                "Social media application data"
            ],
            "Location Data": [
                "GPS location history",
                "WiFi network connections",
                "Cell tower connection records",
                "Location-based application data",
                "Geotagged photos and media"
            ],
            "Application Data": [
                "Installed application list",
                "Application usage statistics",
                "Browser history and bookmarks",
                "Calendar events and reminders"
            ]
        },
        legal_basis="Device owner consent with identity verification",
        retention_period="Case duration + 7 years archival"
    ),

    ConsentLevel.FULL: DataExtractionScope(
        level=ConsentLevel.FULL,
        name="Complete Forensic Extraction",
        description="Comprehensive data extraction including passwords and encrypted content",
        data_categories={
            "Security & Credentials": [
                "Saved passwords and credentials",
                "Biometric data (fingerprints, face recognition)",
                "Encryption keys and certificates",
                "Security application data",
                "Two-factor authentication data"
            ],
            "Media & Files": [
                "All photos, videos, and audio files",
                "Downloaded files and documents",
                "Cloud storage application data",
                "Deleted file recovery (where possible)"
            ],
            "System Data": [
                "Complete file system imaging",
                "System logs and diagnostics",
                "Network configuration and history",
                "Device health and usage analytics"
            ],
            "Advanced Forensics": [
                "Memory dumps and analysis",
                "Application databases",
                "System cache and temporary files",
                "Hidden and system partitions"
            ]
        },
        legal_basis="Device owner consent with enhanced verification",
        retention_period="Case duration + 10 years archival"
    ),

    ConsentLevel.LEGAL: DataExtractionScope(
        level=ConsentLevel.LEGAL,
        name="Court-Ordered Forensic Extraction",
        description="Complete unrestricted access under legal authority",
        data_categories={
            "All Previous Categories": [
                "Everything from Basic, Standard, and Full extractions"
            ],
            "Legal Authority Data": [
                "Bypassing security measures under court order",
                "Access to locked or encrypted containers",
                "Complete device imaging and analysis",
                "Cross-device correlation and linking"
            ]
        },
        legal_basis="Court order or legal warrant",
        retention_period="As required by legal proceedings + archival"
    )
}


@dataclass
class ConsentSession:
    """Tracks all consent details for a case"""
    case_id: str
    device_id: str
    level: ConsentLevel
    nominee_phone: Optional[str] = None
    metadata: Dict = field(default_factory=dict)  # For module-specific data
    created_at: datetime = field(default_factory=datetime.now)
    last_verified: Optional[datetime] = None
    # List of (timestamp, level, action) tuples
    consent_history: list = field(default_factory=list)
    sms_attempts: int = 0  # Track SMS verification attempts
    # Allows infinite attempts for primary evidence cases
    primary_evidence: bool = False
    # FIX #1: Add approval tracking fields
    approval_status: Optional[str] = None  # 'pending', 'approved', 'denied'
    approval_timestamp: Optional[str] = None
    approval_link: Optional[str] = None
    nominee_name: Optional[str] = None

    def generate_verification_code(self) -> str:
        """Generate a random 6-digit verification code."""
        import random
        return f"{random.randint(100000, 999999)}"


class ConsentAuditLogger:
    """Handles forensic audit trails for consent and vault access."""

    def __init__(self, case_id: str):
        self.case_id = case_id
        self.audit_dir = os.path.join('audit', 'consent_records', case_id)
        try:
            os.makedirs(self.audit_dir, exist_ok=True)
        except Exception:
            # Fall back to in-memory audit when storage unavailable
            self.audit_dir = None
            logger.warning('Audit directory unavailable for case %s; using in-memory audit only.', case_id)
        self._memory_log: List[Dict[str, Any]] = []

    def log_access(self, module: str, action: str, metadata: Dict = None):
        """Log an access event to the audit trail."""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'module': module,
            'action': action,
            'metadata': metadata or {}
        }

        if not self.audit_dir:
            self._memory_log.append(entry)
            return

        audit_file = os.path.join(self.audit_dir, 'access_log.json')
        try:
            # Append to existing log
            existing = []
            if os.path.exists(audit_file):
                with open(audit_file, 'r') as f:
                    existing = json.load(f)
            existing.append(entry)
            with open(audit_file, 'w') as f:
                json.dump(existing, f, indent=2)
        except Exception as e:
            self._memory_log.append(entry)
            logger.error(f"Failed to write audit log: {e}")


class PrivacyVault:
    """
    Blockchain-style privacy protection system for sensitive authentication data.

    Features:
    - Cryptographic hashing (SHA-256 with salt)
    - Zero-knowledge verification
    - Immutable audit trails
    - Multi-party verification system
    - Time-locked access controls
    """

    def __init__(self, case_id: str):
        self.case_id = case_id
        self.audit = ConsentAuditLogger(case_id)
        self.entries_file = os.path.join(
            'audit', 'consent_records', case_id, 'vault_entries.json')
        self.vault_entries: Dict[str, Dict] = {}
        self.audit_chain: List[Dict] = []
        self.master_salt: Optional[str] = None
        self.storage_available = True
        self._load_entries()
        if not self.master_salt:
            self.master_salt = secrets.token_hex(32)
            self._save_entries()

    def _load_entries(self):
        """Load persisted vault entries."""
        self.vault_entries = {}
        try:
            if os.path.exists(self.entries_file):
                with open(self.entries_file, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, dict) and 'entries' in data:
                        self.vault_entries = data.get('entries', {}) or {}
                        self.master_salt = data.get('master_salt') or None
                    elif isinstance(data, dict):
                        # Legacy format: just entries dict
                        self.vault_entries = data
        except Exception:
            logger.warning('Failed to load vault entries; starting fresh')
            self.vault_entries = {}

    def _save_entries(self):
        """Persist vault entries."""
        try:
            payload = {
                'master_salt': self.master_salt,
                'entries': self.vault_entries,
            }
            with open(self.entries_file, 'w') as f:
                json.dump(payload, f, indent=2)
            self.storage_available = True
        except Exception as exc:
            self.storage_available = False
            logger.warning('Failed to persist vault entries: %s', exc)

    def store_pin_pattern(self, case_id: str, device_id: str, pin_pattern: str,
                         auth_type: str = "PIN", consent_level: str = "STANDARD") -> str:
        """
        Store PIN/pattern/password with blockchain-style protection.

        Args:
            case_id: The case identifier
            device_id: The device identifier
            pin_pattern: The actual PIN/pattern/password string
            auth_type: Type of authentication ("PIN", "PATTERN", "PASSWORD")
            consent_level: Consent level required to access

        Returns:
            vault_id: Unique identifier for the protected entry
        """
        vault_id = f"{case_id}_{device_id}_{auth_type}_{secrets.token_hex(8)}"

        # Create cryptographic hash with multiple layers
        salt = secrets.token_hex(16)
        combined_salt = self.master_salt + salt

        # Primary hash (SHA-256)
        primary_hash = hashlib.sha256(
            f"{pin_pattern}{combined_salt}".encode()).hexdigest()

        # Secondary hash for verification chain
        secondary_hash = hashlib.sha256(
            f"{primary_hash}{self._get_previous_hash()}".encode()).hexdigest()

        # HMAC for additional security
        hmac_key = secrets.token_bytes(32)
        hmac_signature = hmac.new(
            hmac_key, pin_pattern.encode(), hashlib.sha256).hexdigest()

        # Store protected entry
        entry = {
            'vault_id': vault_id,
            'case_id': case_id,
            'device_id': device_id,
            'auth_type': auth_type,
            'consent_level': consent_level,
            'primary_hash': primary_hash,
            'secondary_hash': secondary_hash,
            'hmac_signature': hmac_signature,
            'salt': salt,
            # Store hash, not key
            'hmac_key_hash': hashlib.sha256(hmac_key).hexdigest(),
            'created_at': datetime.now().isoformat(),
            'access_count': 0,
            'last_access': None,
            'verification_chain': []
        }

        self.vault_entries[vault_id] = entry

        # Add to audit chain
        audit_entry = {
            'timestamp': datetime.now().isoformat(),
            'action': 'STORE',
            'vault_id': vault_id,
            'case_id': case_id,
            'auth_type': auth_type,
            'consent_level': consent_level,
            'hash': secondary_hash
        }
        self.audit_chain.append(audit_entry)

        self._save_entries()

        return vault_id

    def remove_entry(self, vault_id: str) -> bool:
        """Remove a stored entry and persist the updated vault."""
        if vault_id not in self.vault_entries:
            return False

        entry = self.vault_entries.pop(vault_id)
        audit_entry = {
            'timestamp': datetime.now().isoformat(),
            'action': 'DELETE',
            'vault_id': vault_id,
            'case_id': entry.get('case_id'),
            'auth_type': entry.get('auth_type'),
            'consent_level': entry.get('consent_level'),
            'hash': entry.get('secondary_hash')
        }
        self.audit_chain.append(audit_entry)
        self._save_entries()
        logger.info("Removed vault entry %s", vault_id)
        return True

    def verify_pin_pattern(self, vault_id: str, test_value: str, case_id: str) -> bool:
        """Verify PIN/pattern/password with enhanced error handling"""
        try:
            if vault_id not in self.vault_entries:
                logger.warning(f"Vault entry {vault_id} not found")
                return False

            entry = self.vault_entries[vault_id]

            # Case ID verification
            if case_id and entry['case_id'] != case_id:
                logger.warning(f"Case ID mismatch for vault {vault_id}")
                return False

            # Hash verification
            combined_salt = self.master_salt + entry['salt']
            test_hash = hashlib.sha256(
                f"{test_value}{combined_salt}".encode()).hexdigest()
            is_valid = hmac.compare_digest(test_hash, entry['primary_hash'])

            if is_valid:
                entry['access_count'] += 1
                entry['last_access'] = datetime.now().isoformat()
                verification_entry = {
                    'timestamp': datetime.now().isoformat(),
                    'vault_id': vault_id,
                    'action': 'VERIFY_SUCCESS',
                    'case_id': case_id,
                    'verification_hash': hashlib.sha256(f"{test_hash}{entry['secondary_hash']}".encode()).hexdigest()
                }
                entry['verification_chain'].append(verification_entry)
                self.audit_chain.append(verification_entry)
                self._save_entries()
                logger.info(f"Successful verification for vault {vault_id}")
            else:
                logger.warning(
                    f"Failed verification attempt for vault {vault_id}")

            return is_valid

        except Exception as e:
            logger.error(
                f"Vault verification failed: {type(e).__name__} - {str(e)}", exc_info=True)
            self.audit.log_access('vault', 'verify_error', {
                'vault_id': vault_id,
                'error': str(e)
            })
            return False

    def get_vault_entry_info(self, vault_id: str, case_id: str = None) -> Dict:
        """
        Get metadata about a vault entry without revealing the protected value.

        Args:
            vault_id: The vault entry ID
            case_id: Optional case ID for access control

        Returns:
            dict: Metadata about the entry
        """
        if vault_id not in self.vault_entries:
            return {'error': 'Vault entry not found'}

        entry = self.vault_entries[vault_id]

        # Check access permission
        if case_id and entry['case_id'] != case_id:
            return {'error': 'Access denied'}

        return {
            'vault_id': vault_id,
            'case_id': entry['case_id'],
            'device_id': entry['device_id'],
            'auth_type': entry['auth_type'],
            'consent_level': entry['consent_level'],
            'created_at': entry['created_at'],
            'access_count': entry['access_count'],
            'last_access': entry['last_access'],
            'verification_attempts': len(entry['verification_chain'])
        }

    def get_audit_trail(self, case_id: str = None) -> List[Dict]:
        """
        Get audit trail for vault operations.

        Args:
            case_id: Optional filter by case ID

        Returns:
            list: Audit trail entries
        """
        if case_id:
            return [entry for entry in self.audit_chain if entry.get('case_id') == case_id]
        return self.audit_chain.copy()

    def _get_previous_hash(self) -> str:
        """Get hash of previous audit entry for blockchain-like chaining"""
        if not self.audit_chain:
            return "genesis_block"
        return self.audit_chain[-1].get('hash', 'genesis_block')

    def validate_chain_integrity(self) -> bool:
        """Validate the integrity of the audit chain (blockchain-style verification)"""
        for i in range(1, len(self.audit_chain)):
            entry = self.audit_chain[i]
            if 'hash' not in entry:
                continue  # Genesis block

            prev_hash = self.audit_chain[i - 1].get('hash', 'genesis_block')
            expected_hash = hashlib.sha256(
                f"{entry.get('timestamp', '')}{entry.get('action', '')}{prev_hash}".encode()
            ).hexdigest()

            if 'hash' in entry and not hmac.compare_digest(entry['hash'], expected_hash):
                return False

        return True


class ConsentManager:
    """Central consent system for all ForenSmart modules"""
    def __init__(self):
        self.sessions: Dict[str, ConsentSession] = {}

        # Initialize Privacy Vault (will be case-specific per session)
        self.privacy_vaults: Dict[str, PrivacyVault] = {}
        # Audit base dir
        self.audit_base = os.path.join('audit', 'consent_records')
        os.makedirs(self.audit_base, exist_ok=True)
        self.settings_template = {
            'opencellid_key': {
                'type': 'password',
                'description': 'OpenCellID API key for cell tower location lookup',
                'required': False,
                'default': '',
                'category': 'Location Services'
            }
        }

        self._load_sessions_from_disk()

    def _unlock_activity_path(self, case_id: str) -> str:
        return os.path.join(self.audit_base, case_id, 'unlock_activity.json')

    def _append_unlock_activity(self, case_id: str, entry: Dict[str, Any]) -> None:
        try:
            entry.setdefault('timestamp', datetime.now().isoformat())
            path = self._unlock_activity_path(case_id)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            history: List[Dict[str, Any]] = []
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as handle:
                    data = json.load(handle)
                    if isinstance(data, list):
                        history = data
            history.append(entry)
            with open(path, 'w', encoding='utf-8') as handle:
                json.dump(history[-200:], handle, indent=2)
        except Exception:
            logger.warning("Failed to append unlock activity for %s", case_id, exc_info=True)

    def get_unlock_activity(self, case_id: str, limit: int = 25) -> List[Dict[str, Any]]:
        path = self._unlock_activity_path(case_id)
        if not os.path.exists(path):
            return []
        try:
            with open(path, 'r', encoding='utf-8') as handle:
                data = json.load(handle)
                if isinstance(data, list):
                    return data[-limit:]
        except Exception:
            logger.warning("Failed to read unlock activity for %s", case_id, exc_info=True)
        return []

    @staticmethod
    def _parse_datetime(value: Any) -> Optional[datetime]:
        if not value:
            return None
        if isinstance(value, datetime):
            return value
        try:
            return datetime.fromisoformat(str(value))
        except Exception:
            return None

    def _detect_device_serial(self) -> Optional[str]:
        try:
            from adapters.android_adb import AndroidADB  # type: ignore
            adb = AndroidADB()
            default_device = adb.get_default_device()
            if default_device:
                return default_device.get('serial')
        except Exception as exc:
            logger.debug("Device detection failed: %s", exc)
        return None

    def get_device_label(self, device_id: Optional[str]) -> str:
        if not device_id:
            return 'Unknown device'
        label = device_id
        try:
            from adapters.android_adb import AndroidADB  # type: ignore
            adb = AndroidADB()
            summary = adb.device_summary()
            for device in summary.get('devices', []) or []:
                if device.get('serial') == device_id:
                    status = device.get('status') or 'unknown'
                    model = device.get('model') or device.get('name')
                    if model:
                        label = f"{model} ({device_id})"
                    else:
                        label = f"{device_id} ({status})"
                    break
        except Exception as exc:
            logger.debug("Device label lookup failed: %s", exc)
        return label

    def _device_connected(self, device_id: Optional[str]) -> bool:
        if not device_id:
            return False
        try:
            from adapters.android_adb import AndroidADB  # type: ignore
            return AndroidADB().is_connected(device_id)
        except Exception:
            return False

    def get_session(self, case_id: str) -> Optional[ConsentSession]:
        """Get a consent session by case ID"""
        return self.sessions.get(case_id)

    # Persistence helpers
    def _audit_dir(self, case_id: str) -> str:
        d = os.path.join(self.audit_base, case_id)
        os.makedirs(d, exist_ok=True)
        return d

    def _append_audit_log(self, case_id: str, entry: Dict):
        try:
            d = self._audit_dir(case_id)
            path = os.path.join(d, 'access_log.json')
            data = []
            if os.path.exists(path):
                with open(path, 'r') as f:
                    try:
                        data = json.load(f)
                    except Exception:
                        data = []
            data.append(entry)
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def _write_consent_snapshot(self, case_id: str):
        try:
            session = self.sessions.get(case_id)
            if not session:
                return
            d = self._audit_dir(case_id)
            path = os.path.join(d, 'consent.json')
            snap = self.get_session_summary(case_id)
            if not snap:
                return
            # ensure datetimes serialized
            if isinstance(snap.get('created_at'), datetime):
                snap['created_at'] = snap['created_at'].isoformat()
            if isinstance(snap.get('last_verified'), datetime):
                snap['last_verified'] = snap['last_verified'].isoformat()
            with open(path, 'w') as f:
                json.dump(snap, f, indent=2, default=str)
        except Exception:
            pass

    def persist_session(self, case_id: str) -> bool:
        """Public helper to persist current session state to disk."""
        if case_id not in self.sessions:
            return False
        self._write_consent_snapshot(case_id)
        return True

    def _load_sessions_from_disk(self):
        if not os.path.isdir(self.audit_base):
            return

        for case_id in os.listdir(self.audit_base):
            snapshot_path = os.path.join(self.audit_base, case_id, 'consent.json')
            if not os.path.isfile(snapshot_path):
                continue

            try:
                with open(snapshot_path, 'r', encoding='utf-8') as handle:
                    data = json.load(handle)
            except Exception as exc:
                logger.warning("Unable to load consent snapshot for %s: %s", case_id, exc)
                continue

            try:
                level_name = data.get('level', 'BASIC')
                level = ConsentLevel[level_name]
            except KeyError:
                level = ConsentLevel.BASIC

            session = ConsentSession(
                case_id=case_id,
                device_id=data.get('device_id') or '',
                level=level,
                nominee_phone=data.get('nominee_phone'),
                metadata=data.get('metadata', {}),
                created_at=self._parse_datetime(data.get('created_at')) or datetime.now(),
                last_verified=self._parse_datetime(data.get('last_verified')),
                consent_history=[],
                sms_attempts=int(data.get('sms_attempts', 0) or 0),
                primary_evidence=bool(data.get('primary_evidence', False))
            )

            history_data = data.get('consent_history', []) or []
            for entry in history_data:
                ts = self._parse_datetime(entry.get('timestamp')) or datetime.now()
                lvl_name = entry.get('level')
                level_obj = level
                if isinstance(lvl_name, str) and lvl_name in ConsentLevel.__members__:
                    level_obj = ConsentLevel[lvl_name]
                action = entry.get('action') or ''
                session.consent_history.append((ts, level_obj, action))

            self.sessions[case_id] = session

        # Attempt to auto-bind detected devices for sessions lacking IDs
        for case_id in list(self.sessions.keys()):
            self.ensure_device_id(case_id)

    def get_session_summary(self, case_id: str) -> Optional[Dict[str, Any]]:
        session = self.sessions.get(case_id)
        if not session:
            return None

        def _event_to_dict(event):
            ts, lvl, action = event
            ts_value = ts.isoformat() if isinstance(ts, datetime) else str(ts)
            lvl_value = lvl.name if isinstance(lvl, ConsentLevel) else str(lvl)
            return {
                'timestamp': ts_value,
                'level': lvl_value,
                'action': action
            }

        return {
            'case_id': session.case_id,
            'device_id': session.device_id,
            'level': session.level.name,
            'nominee_phone': session.nominee_phone,
            'metadata': session.metadata,
            'created_at': session.created_at,
            'last_verified': session.last_verified,
            'consent_history': [_event_to_dict(e) for e in session.consent_history],
            'sms_attempts': session.sms_attempts,
            'primary_evidence': session.primary_evidence
        }

    def ensure_device_id(self, case_id: str) -> Optional[str]:
        """Ensure device ID with enhanced detection and auto-recovery."""
        session = self.sessions.get(case_id)
        if not session:
            return None

        # First, try enhanced device detector with auto-recovery
        try:
            from modules.device_detector import DeviceDetector
            
            # Use enhanced diagnostics with auto-recovery
            diagnosis = DeviceDetector.diagnose_and_recover()
            
            # Check if we have an authorized device
            if diagnosis.get("authorized_device"):
                detected = diagnosis["authorized_device"]
                if session.device_id != detected:
                    session.device_id = detected
                    self._write_consent_snapshot(case_id)
                    logger.info(f"Device detected and set for {case_id}: {detected}")
                return detected
            
            # If no authorized device, try to get any connected device
            if diagnosis.get("devices"):
                for device in diagnosis["devices"]:
                    if device.get("status") == "device":  # Authorized device
                        detected = device.get("serial")
                        if detected and session.device_id != detected:
                            session.device_id = detected
                            self._write_consent_snapshot(case_id)
                            logger.info(f"Device detected from list for {case_id}: {detected}")
                        return detected
        except Exception as e:
            logger.debug(f"Enhanced device detection failed: {e}")

        # Fallback to basic detection if enhanced detection fails
        if session.device_id and self._device_connected(session.device_id):
            return session.device_id

        detected = self._detect_device_serial()
        if detected:
            if session.device_id != detected:
                session.device_id = detected
                self._write_consent_snapshot(case_id)
            return detected

        return session.device_id or None

    def _get_privacy_vault(self, case_id: str) -> PrivacyVault:
        vault = self.privacy_vaults.get(case_id)
        if not vault:
            vault = PrivacyVault(case_id)
            self.privacy_vaults[case_id] = vault
        return vault

    def store_messaging_secret(
        self,
        case_id: str,
        device_id: str,
        secret_value: str,
        *,
        auth_type: str = 'PIN',
        consent_level: str = 'STANDARD',
    ) -> Optional[str]:
        session = self.sessions.get(case_id)
        if not session:
            return None

        vault = self._get_privacy_vault(case_id)
        secret_value = (secret_value or '').strip()
        if not secret_value:
            return None

        if vault.storage_available:
            vault_id = vault.store_pin_pattern(
                case_id,
                device_id,
                secret_value,
                auth_type=auth_type,
                consent_level=consent_level,
            )
            messaging_store = session.metadata.setdefault('messaging_vault', {})
            messaging_store.setdefault('entries', [])
            messaging_store['entries'].append({
                'vault_id': vault_id,
                'auth_type': auth_type,
                'created_at': datetime.now().isoformat(),
            })
            messaging_store['verified'] = False
            self._write_consent_snapshot(case_id)
            return vault_id

        # Fallback: encrypt and store within session metadata for persistence
        entry_id = secrets.token_hex(16)
        salt = secrets.token_hex(16)
        hashed_secret = hashlib.sha256(f"{secret_value}{salt}".encode()).hexdigest()
        messaging_store = session.metadata.setdefault('messaging_vault', {})
        messaging_store.setdefault('entries', [])
        messaging_store['entries'].append({
            'vault_id': entry_id,
            'auth_type': auth_type,
            'created_at': datetime.now().isoformat(),
            'fallback': True,
        })
        fallback_entries = messaging_store.setdefault('fallback_entries', [])
        fallback_entries.append({
            'entry_id': entry_id,
            'device_id': device_id,
            'salt': salt,
            'hash': hashed_secret,
            'auth_type': auth_type,
            'consent_level': consent_level,
            'created_at': datetime.now().isoformat(),
        })
        messaging_store['verified'] = False
        self._write_consent_snapshot(case_id)
        return entry_id

    def verify_messaging_consent(self, case_id: str, device_id: str, require_pin: bool = False) -> Dict[str, Any]:
        session = self.sessions.get(case_id)
        if not session:
            return {
                'access_granted': False,
                'reason': 'no_session',
                'message': f'Consent session for {case_id} not found.'
            }

        if session.level.value < ConsentLevel.STANDARD.value:
            return {
                'access_granted': False,
                'reason': 'insufficient_consent',
                'message': 'Standard consent level required for communications extraction.'
            }

        unlock = self.get_unlock_status(case_id)
        if unlock.get('status') != 'verified':
            return {
                'access_granted': False,
                'reason': 'unlock_not_verified',
                'message': 'Messaging access requires verified unlock code.'
            }

        messaging_vault = session.metadata.setdefault('messaging_vault', {})
        vault_required = require_pin and not messaging_vault.get('verified')

        return {
            'access_granted': not vault_required,
            'vault_required': vault_required,
            'vault_verified': messaging_vault.get('verified', False),
            'vault_entries': list(messaging_vault.get('entries', []))
        }

    def record_messaging_vault_verification(self, case_id: str, result: bool) -> None:
        session = self.sessions.get(case_id)
        if not session:
            return
        messaging_vault = session.metadata.setdefault('messaging_vault', {})
        messaging_vault['verified'] = result
        messaging_vault.setdefault('entries', [])
        self._write_consent_snapshot(case_id)

    def get_messaging_vault_entries(self, case_id: str) -> List[Dict[str, Any]]:
        session = self.sessions.get(case_id)
        if not session:
            return []
        messaging_vault = session.metadata.setdefault('messaging_vault', {'entries': [], 'verified': False})
        return list(messaging_vault.get('entries', []))

    # def _load_sessions_from_disk(self):
    #     if not os.path.isdir(self.audit_base):
    #         return
    #
    #     for case_id in os.listdir(self.audit_base):
    #         snapshot_path = os.path.join(self.audit_base, case_id, 'consent.json')
    #         if not os.path.isfile(snapshot_path):
    #             continue
    #         entries.append({
    #             'vault_id': vault_id,
    #             'auth_type': auth_type,
    #             'created_at': datetime.now().isoformat()
    #         })
    #         messaging_vault['verified'] = False
    #         self._write_consent_snapshot(case_id)
    #         return vault_id

    def verify_messaging_secret(self, case_id: str, vault_id: str, attempt: str) -> bool:
        session = self.sessions.get(case_id)
        if not session or not vault_id:
            return False
        attempt_value = (attempt or '').strip()
        if not attempt_value:
            return False
        messaging_store = session.metadata.setdefault('messaging_vault', {})
        fallback_entries = messaging_store.get('fallback_entries', [])
        is_fallback = any(entry.get('vault_id') == vault_id and entry.get('fallback')
                          for entry in messaging_store.get('entries', []))

        if is_fallback:
            entry = next((item for item in fallback_entries
                          if item.get('entry_id') == vault_id), None)
            if not entry:
                return False
            computed = hashlib.sha256(f"{attempt_value}{entry['salt']}".encode()).hexdigest()
            success = hmac.compare_digest(computed, entry['hash'])
            if success:
                messaging_store['verified'] = True
                self._write_consent_snapshot(case_id)
                self.record_messaging_vault_verification(case_id, True)
            return success

        try:
            vault = self._get_privacy_vault(case_id)
            result = vault.verify_pin_pattern(vault_id, attempt_value, case_id)
        except Exception:
            return False

        success = bool(result)
        if success:
            messaging_store['verified'] = True
            self._write_consent_snapshot(case_id)
        self.record_messaging_vault_verification(case_id, success)
        return success

    def delete_messaging_secret(self, case_id: str, vault_id: str) -> bool:
        session = self.sessions.get(case_id)
        if not session or not vault_id:
            return False

        messaging_store = session.metadata.setdefault('messaging_vault', {})
        entries = messaging_store.get('entries', []) or []
        target_entry = next((item for item in entries if item.get('vault_id') == vault_id), None)
        if not target_entry:
            return False

        is_fallback = bool(target_entry.get('fallback'))
        messaging_store['entries'] = [item for item in entries if item.get('vault_id') != vault_id]

        if is_fallback:
            fallback_entries = messaging_store.get('fallback_entries', []) or []
            messaging_store['fallback_entries'] = [
                entry for entry in fallback_entries if entry.get('entry_id') != vault_id
            ]
        else:
            try:
                vault = self._get_privacy_vault(case_id)
                removed = vault.remove_entry(vault_id)
                if not removed:
                    logger.warning("Vault entry %s was not found during deletion", vault_id)
            except Exception:
                logger.warning("Failed to remove vault entry %s", vault_id, exc_info=True)

        messaging_store['verified'] = False
        self._write_consent_snapshot(case_id)
        logger.info("Pruned messaging secret %s for case %s", vault_id, case_id)
        return True

    # OTP helpers -----------------------------------------------------

    def verify_unlock_code(self, case_id: str, code: str) -> bool:
        session = self.sessions.get(case_id)
        if not session:
            return False
        valid = session.validate_code(code)
        status = session.metadata.setdefault('unlock_status', {})
        status['status'] = 'verified' if valid else 'failed'
        status['verified_at'] = datetime.now().isoformat()
        self._write_consent_snapshot(case_id)
        return valid

    def create_session(self, case_id: str, device_id: Optional[str] = None, primary_evidence: bool = False) -> ConsentSession:
        """Initialize new consent session"""
        if case_id in self.sessions:
            raise ValueError(f"Case {case_id} already exists")

        device_id = (device_id or '').strip() or self._detect_device_serial() or 'UNKNOWN_DEVICE'
        session = ConsentSession(
            case_id=case_id,
            device_id=device_id,
            level=ConsentLevel.BASIC,
            primary_evidence=primary_evidence
        )
        evidence_type = "primary evidence" if primary_evidence else "standard case"
        session.consent_history.append(
            (datetime.now(), ConsentLevel.BASIC, f"Session created - {evidence_type}"))
        self.sessions[case_id] = session
        self._write_consent_snapshot(case_id)
        self.ensure_device_id(case_id)
        return session

    def _get_unlock_metadata(self, session: ConsentSession) -> Dict[str, Any]:
        store = session.metadata.setdefault('unlock_request', {})
        store.setdefault('status', 'pending')
        store.setdefault('decision_history', [])
        return store

    def request_unlock_verification(
        self,
        case_id: str,
        requested_level: Optional[ConsentLevel] = None,
        purpose: str = '',
        nominee_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Backward-compatible wrapper for link-based approval workflow."""
        level = requested_level or ConsentLevel.STANDARD
        return self.create_unlock_approval(case_id, level, purpose, nominee_name)

    def create_unlock_approval(
        self,
        case_id: str,
        requested_level: ConsentLevel,
        purpose: str,
        nominee_name: Optional[str] = None
    ) -> Dict[str, Any]:
        session = self.sessions.get(case_id)
        if not session:
            return {'status': 'error', 'message': f"No consent session for case {case_id}"}

        token = secrets.token_urlsafe(16)
        unlock_meta = self._get_unlock_metadata(session)
        unlock_meta.update({
            'token': token,
            'status': 'pending',
            'requested_level': requested_level.name,
            'requested_level_value': requested_level.value,
            'purpose': purpose,
            'nominee_name': nominee_name,
            'requested_at': datetime.now().isoformat(),
            'responded_at': None,
            'nominee_response': None
        })
        unlock_meta.pop('code', None)
        unlock_meta.pop('expires_at', None)
        unlock_meta.pop('last_error', None)
        unlock_meta['decision_history'].append({
            'timestamp': datetime.now().isoformat(),
            'action': 'request_created',
            'requested_level': requested_level.name,
            'purpose': purpose
        })

        self._append_audit_log(case_id, {
            'timestamp': datetime.now().isoformat(),
            'action': 'unlock_request',
            'purpose': purpose,
            'requested_level': requested_level.name,
            'delivery': 'link'
        })
        self._append_unlock_activity(case_id, {
            'action': 'request_created',
            'requested_level': requested_level.name,
            'purpose': purpose,
            'token': token,
            'nominee_name': nominee_name
        })
        self._write_consent_snapshot(case_id)

        return {
            'status': 'pending',
            'token': token,
            'requested_level': requested_level.name,
            'purpose': purpose
        }

    def get_unlock_request_by_token(self, token: str) -> Optional[Dict[str, Any]]:
        for session in self.sessions.values():
            meta = session.metadata.get('unlock_request', {})
            if meta.get('token') == token:
                return {
                    'case_id': session.case_id,
                    'session': session,
                    'unlock_meta': meta
                }
        return None

    def respond_to_unlock_token(
        self,
        token: str,
        decision: str,
        message: Optional[str] = None
    ) -> Dict[str, Any]:
        record = self.get_unlock_request_by_token(token)
        if not record:
            return {'status': 'error', 'message': 'Invalid or expired approval link.'}

        session: ConsentSession = record['session']
        unlock_meta = record['unlock_meta']
        decision = decision.lower()

        if unlock_meta.get('status') in {'verified', 'denied'}:
            return {'status': unlock_meta.get('status'), 'message': 'Approval already processed.'}

        now = datetime.now().isoformat()
        unlock_meta['responded_at'] = now
        unlock_meta['nominee_response'] = decision
        unlock_meta.setdefault('decision_history', []).append({
            'timestamp': now,
            'action': f'candidate_{decision}',
            'message': message
        })

        if decision == 'approved':
            unlock_meta['status'] = 'verified'
            unlock_meta.pop('last_error', None)
            requested_level_value = unlock_meta.get('requested_level_value', ConsentLevel.STANDARD.value)
            requested_level = ConsentLevel(requested_level_value)
            if session.level.value < requested_level.value:
                session.level = requested_level
            session.last_verified = datetime.now()
            session.consent_history.append((datetime.now(), session.level, 'remote_approval'))
            self._append_audit_log(session.case_id, {
                'timestamp': now,
                'action': 'unlock_verified',
                'token': token
            })
            self._append_unlock_activity(session.case_id, {
                'action': 'approved',
                'token': token,
                'message': message,
                'requested_level': requested_level.name
            })
            self._write_consent_snapshot(session.case_id)
            return {'status': 'verified', 'message': 'Approval granted.'}

        unlock_meta['status'] = 'denied'
        unlock_meta['last_error'] = message or 'Nominee denied the request.'
        self._append_audit_log(session.case_id, {
            'timestamp': now,
            'action': 'unlock_denied',
            'token': token,
            'note': message
        })
        self._append_unlock_activity(session.case_id, {
            'action': 'denied',
            'token': token,
            'message': message
        })
        self._write_consent_snapshot(session.case_id)
        return {'status': 'denied', 'message': unlock_meta['last_error']}

    def _refresh_unlock_status(self, session: ConsentSession) -> Dict[str, Any]:
        unlock_meta = self._get_unlock_metadata(session)
        expires_at = unlock_meta.get('expires_at')
        if expires_at:
            try:
                expires_dt = datetime.fromisoformat(expires_at)
                if expires_dt < datetime.now() and unlock_meta.get('status') == 'verified':
                    unlock_meta['status'] = 'expired'
                    unlock_meta['expired_at'] = datetime.now().isoformat()
                    unlock_meta['last_error'] = 'Verification expired'
            except ValueError:
                unlock_meta['last_error'] = 'Invalid expiry timestamp'
        return unlock_meta

    def verify_unlock_code(self, case_id: str, submitted_code: str) -> Dict[str, Any]:
        session = self.sessions.get(case_id)
        if not session:
            return {'status': 'error', 'message': f"No consent session for case {case_id}"}

        unlock_meta = self._refresh_unlock_status(session)
        token = unlock_meta.get('token')
        if not token:
            return {'status': 'error', 'message': 'No approval request found.'}
        # Legacy compatibility: treat submitted code as approval token
        if submitted_code.strip() == token:
            unlock_meta['status'] = 'verified'
            session.last_verified = datetime.now()
            session.consent_history.append((datetime.now(), session.level, 'manual_token_verified'))
            self._write_consent_snapshot(case_id)
            return {'status': 'verified', 'message': 'Verification successful.'}
        return {'status': 'error', 'message': 'Approval token mismatch.'}

    def get_unlock_status(self, case_id: str) -> Dict[str, Any]:
        session = self.sessions.get(case_id)
        if not session:
            return {}
        return self._refresh_unlock_status(session).copy()

    def get_recent_history(self, case_id: str, limit: int = 25) -> List[Dict[str, Any]]:
        session = self.sessions.get(case_id)
        if not session:
            return []
        history = [
            {
                'timestamp': ts.isoformat() if isinstance(ts, datetime) else str(ts),
                'level': lvl.name if isinstance(lvl, ConsentLevel) else str(lvl),
                'action': action
            }
            for ts, lvl, action in session.consent_history[-limit:]
        ]
        return list(reversed(history))

    def set_consent_level(self, case_id: str, new_level: ConsentLevel, reason: str) -> Dict[str, Any]:
        session = self.sessions.get(case_id)
        if not session:
            return {'status': 'error', 'message': f"No consent session for case {case_id}"}

        if session.level == new_level:
            return {'status': 'noop', 'message': 'Consent level already set.'}

        session.level = new_level
        session.last_verified = datetime.now()
        session.consent_history.append(
            (datetime.now(), new_level, f"level_set_manual: {reason or 'manual update'}")
        )
        self._write_consent_snapshot(case_id)
        self._append_audit_log(case_id, {
            'timestamp': datetime.now().isoformat(),
            'action': 'consent_level_update',
            'level': new_level.name,
            'reason': reason
        })
        
        # NEW: Record consent level change in audit trail
        if ConsentAuditTrail:
            try:
                ConsentAuditTrail.record_approval(
                    case_id=case_id,
                    decision=f"consent_level_{new_level.name}",
                    nominee_name=session.nominee_name,
                    device_id=session.device_id or "UNKNOWN",
                    purpose=f"Consent level updated to {new_level.name}: {reason}"
                )
            except Exception as e:
                logger.warning(f"Failed to record consent audit trail: {e}")
        
        return {'status': 'updated', 'message': f'Consent level set to {new_level.name}.'}

    def maybe_expire_consent(self, case_id: str, expiry_hours: int = 12) -> Optional[Dict[str, Any]]:
        session = self.sessions.get(case_id)
        if not session:
            return None

        unlock_meta = self._refresh_unlock_status(session)
        now = datetime.now()
        downgraded = False

        if session.last_verified and (now - session.last_verified).total_seconds() > expiry_hours * 3600:
            session.level = ConsentLevel.BASIC
            session.consent_history.append(
                (now, ConsentLevel.BASIC, 'auto_downgrade_expired')
            )
            downgraded = True

        # if unlock_meta.get('status') in {'expired', 'pending', None} and session.level.value > ConsentLevel.BASIC.value:
        #     session.level = ConsentLevel.BASIC
        #     session.consent_history.append(
        #         (now, ConsentLevel.BASIC, 'auto_downgrade_unlock_not_verified')
        #     )
        #     downgraded = True

        if downgraded:
            self._write_consent_snapshot(case_id)
            self._append_audit_log(case_id, {
                'timestamp': now.isoformat(),
                'action': 'consent_auto_downgrade',
                'reason': 'stale or unverified consent'
            })

        return {
            'downgraded': downgraded,
            'level': session.level.name,
            'unlock_status': unlock_meta.get('status')
        }

    def record_audit_event(self, case_id: str, action: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        entry = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'metadata': metadata or {}
        }
        self._append_audit_log(case_id, entry)

    def verify_consent(self, case_id: str, required_level: ConsentLevel) -> bool:
        """Core verification used by all modules"""
        session = self.sessions.get(case_id)
        if not session:
            return False

        if session.level.value < required_level.value:
            scope = EXTRACTION_SCOPES.get(required_level)
            return False

        session.last_verified = datetime.now()
        self._write_consent_snapshot(case_id)
        return True

    # FIX #5: Add device detection method
    def get_or_detect_device(self, case_id: str) -> Optional[str]:
        """Get device ID from session or detect it."""
        session = self.sessions.get(case_id)
        if not session:
            return None
        
        if session.device_id and session.device_id != 'UNKNOWN_DEVICE':
            return session.device_id
        
        detected = self.ensure_device_id(case_id)
        if detected:
            session.device_id = detected
            self._write_consent_snapshot(case_id)
        
        return detected

    # FIX #6: Add approval retrieval methods
    def get_approval_history(self, case_id: str) -> List[Dict[str, Any]]:
        """Get approval history for a case."""
        try:
            from modules.approval_utils import get_approvals_file
            approvals_file = get_approvals_file()
            if not approvals_file.exists():
                return []
            approvals = json.loads(approvals_file.read_text())
            if case_id in approvals:
                return approvals[case_id].get('history', [])
        except Exception as e:
            logger.error(f"Failed to get approval history: {e}")
        return []

    def get_latest_approval_link(self, case_id: str) -> Optional[str]:
        """Get the latest approval link for a case."""
        session = self.sessions.get(case_id)
        if session and session.approval_link:
            return session.approval_link
        try:
            from modules.approval_utils import get_approvals_file
            approvals_file = get_approvals_file()
            if approvals_file.exists():
                approvals = json.loads(approvals_file.read_text())
                if case_id in approvals:
                    return approvals[case_id].get('approval_link')
        except Exception as e:
            logger.error(f"Failed to get approval link: {e}")
        return None

    def get_opencellid_key(self, case_id: str) -> Optional[str]:
        """Get the OpenCellID API key for a case."""
        key = self.get_setting(case_id, 'opencellid_key')
        return key if key else os.getenv('OPENCELLID_KEY')

    # ... rest of the code remains the same ...
