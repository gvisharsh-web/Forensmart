"""Standalone Streamlit entrypoint for nominee consent approval links."""
from __future__ import annotations

import sys
import json
import base64
from pathlib import Path
from typing import Optional, Dict, Any
from urllib.parse import unquote
from datetime import datetime

import streamlit as st

# Ensure project root is importable when deployed separately (e.g., Streamlit Cloud)
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
parent = PROJECT_ROOT.parent
if str(parent) not in sys.path:
    sys.path.append(str(parent))

from modules.consent import ConsentManager, ConsentLevel  # noqa: E402  (loaded after sys.path tweak)
from modules.approval_utils import get_approvals_file, save_approval_decision  # noqa: E402
from modules.approval_redirect import ApprovalRedirect, ApprovalNotifier  # noqa: E402
import logging  # noqa: E402
import logging.handlers  # noqa: E402
from urllib.parse import quote  # noqa: E402

try:
    from modules.dashboard import render_nominee_approval  # noqa: E402
except Exception as exc:  # pragma: no cover - safety net for partial deployments
    render_nominee_approval = None  # type: ignore
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


# ============================================================================
# INTEGRATED LOGGING & AUDIT TRAIL CLASSES
# ============================================================================

class ConsentPortalLogger:
    """Persistent logging for consent portal."""
    
    _instance = None
    _logger = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize logger with file handlers."""
        self._logger = logging.getLogger('consent_portal')
        self._logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        self._logger.handlers = []
        
        # Create audit directory
        audit_dir = Path('audit/consent_portal')
        audit_dir.mkdir(parents=True, exist_ok=True)
        
        # File handler (text log)
        log_file = audit_dir / f'portal_{datetime.now().strftime("%Y%m%d")}.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        self._logger.addHandler(file_handler)
        
        # Rotating file handler
        rotating_handler = logging.handlers.RotatingFileHandler(
            audit_dir / 'portal_current.log',
            maxBytes=10*1024*1024,  # 10 MB
            backupCount=5
        )
        rotating_handler.setLevel(logging.INFO)
        rotating_handler.setFormatter(file_formatter)
        self._logger.addHandler(rotating_handler)
    
    def get_logger(self):
        """Get the configured logger."""
        return self._logger


class ConsentAuditTrail:
    """Structured audit trail for consent portal approvals."""
    
    AUDIT_FILE = Path('audit/consent_portal/audit_trail.json')
    
    @classmethod
    def initialize(cls):
        """Create audit file if needed."""
        cls.AUDIT_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not cls.AUDIT_FILE.exists():
            cls.AUDIT_FILE.write_text(json.dumps([], indent=2))
    
    @classmethod
    def record_approval(cls,
                       case_id: str,
                       decision: str,
                       nominee_name: str,
                       device_id: str,
                       purpose: str = "Not specified") -> bool:
        """Record approval decision to audit trail."""
        try:
            cls.initialize()
            
            # Read existing trail
            trail = json.loads(cls.AUDIT_FILE.read_text())
            
            # Create new entry
            entry = {
                'id': len(trail) + 1,
                'timestamp': datetime.now().isoformat(),
                'case_id': case_id,
                'decision': decision,
                'nominee_name': nominee_name,
                'device_id': device_id,
                'purpose': purpose,
                'status': 'recorded'
            }
            
            trail.append(entry)
            
            # Write back
            cls.AUDIT_FILE.write_text(json.dumps(trail, indent=2))
            return True
        except Exception as e:
            print(f"Failed to record audit trail: {e}")
            return False
    
    @classmethod
    def get_audit_trail(cls, case_id: Optional[str] = None) -> list:
        """Retrieve audit trail, optionally filtered by case_id."""
        try:
            cls.initialize()
            trail = json.loads(cls.AUDIT_FILE.read_text())
            
            if case_id:
                return [entry for entry in trail if entry['case_id'] == case_id]
            return trail
        except Exception:
            return []
    
    @classmethod
    def get_statistics(cls) -> Dict[str, Any]:
        """Get audit trail statistics."""
        trail = cls.get_audit_trail()
        
        return {
            'total_records': len(trail),
            'approvals': len([e for e in trail if e['decision'] == 'approved']),
            'denials': len([e for e in trail if e['decision'] == 'denied']),
            'cases': len(set(e['case_id'] for e in trail)),
            'first_record': trail[0]['timestamp'] if trail else None,
            'last_record': trail[-1]['timestamp'] if trail else None
        }
    
    @classmethod
    def export_audit_trail(cls, case_id: Optional[str] = None) -> str:
        """Export audit trail as JSON string."""
        trail = cls.get_audit_trail(case_id)
        return json.dumps(trail, indent=2)


# ============================================================================
# ENHANCED CONSENT PORTAL WITH QR CODES AND DELIVERY OPTIONS
# ============================================================================

class ConsentPortalEnhancer:
    """Enhance consent portal with QR codes and delivery options."""

    @staticmethod
    def generate_qr_code_url(approval_link: str) -> str:
        """Generate QR code URL for approval link."""
        try:
            # Use QR code API (free service)
            qr_api = "https://api.qrserver.com/v1/create-qr-code/"
            params = f"?size=300x300&data={quote(approval_link)}"
            qr_url = qr_api + params
            logger = logging.getLogger(__name__)
            logger.info(f"Generated QR code URL for approval link")
            return qr_url
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to generate QR code: {e}")
            return ""

    @staticmethod
    def create_whatsapp_link(phone: str, approval_link: str, nominee_name: str = "") -> str:
        """Create WhatsApp share link."""
        try:
            message = (
                f"Hi {nominee_name or 'there'},\n\n"
                f"Please review and approve this ForenSmart extraction request:\n\n"
                f"{approval_link}\n\n"
                f"Thank you!"
            )
            encoded_message = quote(message)
            whatsapp_link = f"https://wa.me/{phone}?text={encoded_message}"
            logger = logging.getLogger(__name__)
            logger.info(f"Created WhatsApp link for {phone}")
            return whatsapp_link
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to create WhatsApp link: {e}")
            return ""

    @staticmethod
    def create_sms_link(phone: str, approval_link: str) -> str:
        """Create SMS share link."""
        try:
            message = f"ForenSmart approval link: {approval_link}"
            encoded_message = quote(message)
            sms_link = f"sms:{phone}?body={encoded_message}"
            logger = logging.getLogger(__name__)
            logger.info(f"Created SMS link for {phone}")
            return sms_link
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to create SMS link: {e}")
            return ""

    @staticmethod
    def create_email_link(email: str, approval_link: str, case_id: str = "") -> str:
        """Create email share link."""
        try:
            subject = f"ForenSmart Extraction Approval Request - {case_id}"
            body = (
                f"Please review and approve this ForenSmart extraction request:\n\n"
                f"{approval_link}\n\n"
                f"Thank you!"
            )
            encoded_subject = quote(subject)
            encoded_body = quote(body)
            email_link = f"mailto:{email}?subject={encoded_subject}&body={encoded_body}"
            logger = logging.getLogger(__name__)
            logger.info(f"Created email link for {email}")
            return email_link
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to create email link: {e}")
            return ""

    @staticmethod
    def add_link_expiration(approval_link: str, hours: int = 24) -> str:
        """Add expiration info to approval link."""
        try:
            from datetime import timedelta
            
            expiry_time = datetime.now() + timedelta(hours=hours)
            expiry_str = expiry_time.strftime("%Y-%m-%d %H:%M:%S")
            
            # Append expiration as fragment (not sent to server)
            link_with_expiry = f"{approval_link}#expires={quote(expiry_str)}"
            logger = logging.getLogger(__name__)
            logger.info(f"Added {hours}h expiration to approval link")
            return link_with_expiry
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to add expiration: {e}")
            return approval_link

    @staticmethod
    def create_approval_details_json(
        case_id: str,
        device_id: str,
        purpose: str,
        requested_level: str,
        nominee_name: str = ""
    ) -> str:
        """Create JSON with approval details for display."""
        try:
            details = {
                "case_id": case_id,
                "device_id": device_id,
                "purpose": purpose,
                "requested_level": requested_level,
                "nominee_name": nominee_name,
                "created_at": datetime.now().isoformat(),
            }
            
            json_str = json.dumps(details)
            encoded = base64.b64encode(json_str.encode()).decode()
            logger = logging.getLogger(__name__)
            logger.info("Created approval details JSON")
            return encoded
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to create approval details: {e}")
            return ""

    @staticmethod
    def get_delivery_options(
        approval_link: str,
        nominee_phone: str = "",
        nominee_email: str = "",
        nominee_name: str = "",
        case_id: str = ""
    ) -> Dict[str, Dict[str, str]]:
        """Get all available delivery options."""
        options = {
            "direct_link": {
                "label": "Direct Link",
                "url": approval_link,
                "description": "Copy and share directly"
            },
            "qr_code": {
                "label": "QR Code",
                "url": ConsentPortalEnhancer.generate_qr_code_url(approval_link),
                "description": "Scan with phone camera"
            }
        }

        if nominee_phone:
            options["whatsapp"] = {
                "label": "WhatsApp",
                "url": ConsentPortalEnhancer.create_whatsapp_link(
                    nominee_phone, approval_link, nominee_name
                ),
                "description": "Send via WhatsApp"
            }
            options["sms"] = {
                "label": "SMS",
                "url": ConsentPortalEnhancer.create_sms_link(nominee_phone, approval_link),
                "description": "Send via SMS"
            }

        if nominee_email:
            options["email"] = {
                "label": "Email",
                "url": ConsentPortalEnhancer.create_email_link(
                    nominee_email, approval_link, case_id
                ),
                "description": "Send via Email"
            }

        return options

    @staticmethod
    def render_delivery_ui(
        approval_link: str,
        nominee_phone: str = "",
        nominee_email: str = "",
        nominee_name: str = "",
        case_id: str = ""
    ) -> None:
        """Render delivery options UI in Streamlit."""
        try:
            st.markdown("### 📤 Share Approval Link")
            
            options = ConsentPortalEnhancer.get_delivery_options(
                approval_link, nominee_phone, nominee_email, nominee_name, case_id
            )

            cols = st.columns(len(options))
            
            for col, (key, option) in zip(cols, options.items()):
                with col:
                    if st.button(f"📱 {option['label']}", key=f"delivery_{key}"):
                        if key == "direct_link":
                            st.code(option['url'])
                            st.info("Copy the link above and share it")
                        elif key == "qr_code":
                            st.image(option['url'], caption="Scan with phone camera")
                        else:
                            st.markdown(f"[{option['label']}]({option['url']})")
                            st.caption(option['description'])
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to render delivery UI: {e}")
            st.error(f"Failed to render delivery options: {e}")


@st.cache_resource(show_spinner=False)
def get_consent_manager() -> ConsentManager:
    """Reuse a single ConsentManager per process for consistent state."""
    return ConsentManager()


def _extract_query_params() -> Dict[str, Any]:
    """Read query parameters from Streamlit's query params API."""
    try:
        params = st.query_params  # Streamlit >=1.30 API
    except AttributeError:  # pragma: no cover - legacy fallback
        params = st.experimental_get_query_params()
    return dict(params) if params else {}


def _decode_approval_data(encoded: str) -> Optional[Dict[str, Any]]:
    """Decode base64-encoded approval data from URL."""
    try:
        decoded = base64.b64decode(unquote(encoded)).decode('utf-8')
        return json.loads(decoded)
    except Exception:
        return None


def _save_approval(case_id: str, decision: str, nominee_name: Optional[str] = None, message: Optional[str] = None, approval_link: Optional[str] = None) -> bool:
    """Save approval decision to shared file using unified approval_utils."""
    try:
        # Use the unified approval_utils to save decision
        success = save_approval_decision(case_id, decision, nominee_name, message)
        
        if success:
            # Also save the approval link separately for tracking
            approvals_file = get_approvals_file()
            approvals = {}
            
            if approvals_file.exists():
                try:
                    approvals = json.loads(approvals_file.read_text())
                except Exception:
                    approvals = {}
            
            # Update with link info
            if case_id in approvals:
                approvals[case_id]['approval_link'] = approval_link
                approvals_file.write_text(json.dumps(approvals, indent=2))
            
            # FIX #2: Sync approval to ConsentSession
            cm = get_consent_manager()
            session = cm.get_session(case_id)
            if session:
                session.approval_status = decision
                session.approval_timestamp = datetime.now().isoformat()
                session.nominee_name = nominee_name
                session.approval_link = approval_link
                cm.persist_session(case_id)
            
            # Record to audit trail
            device_id = approvals.get(case_id, {}).get('device_id', 'UNKNOWN')
            purpose = approvals.get(case_id, {}).get('purpose', 'Not specified')
            ConsentAuditTrail.record_approval(
                case_id=case_id,
                decision=decision,
                nominee_name=nominee_name or 'Unknown',
                device_id=device_id,
                purpose=purpose
            )
            
            # NEW: Notify dashboard of approval and trigger extraction
            ApprovalNotifier.notify_approval(
                case_id=case_id,
                device_id=device_id,
                decision=decision,
                nominee_name=nominee_name,
                extraction_type="android"
            )
            
            # Log and display success with file details
            logger = logging.getLogger(__name__)
            logger.info(f"✅ Approval saved for case {case_id} to {approvals_file}")
            
            st.success(f"✅ Approval saved successfully for case {case_id}")
            st.info(f"📁 Saved to: `{approvals_file}`")
            st.caption(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            return True
        else:
            st.error(f"Failed to save approval for case {case_id}")
            return False
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to save approval: {e}")
        st.error(f"Failed to save approval: {e}")
        return False


def _save_approval_link(case_id: str, approval_link: str, nominee_name: Optional[str] = None) -> bool:
    """Save approval link for future reference and tracking."""
    try:
        # Get the approvals file path
        approvals_file = get_approvals_file()
        approvals = {}
        
        if approvals_file.exists():
            try:
                approvals = json.loads(approvals_file.read_text())
            except Exception:
                approvals = {}
        
        # Create or update the approval link record
        if case_id not in approvals:
            approvals[case_id] = {}
        
        approvals[case_id].update({
            'approval_link': approval_link,
            'link_created_at': datetime.now().isoformat(),
            'nominee_name': nominee_name,
            'status': 'pending'  # pending, approved, denied
        })
        
        approvals_file.write_text(json.dumps(approvals, indent=2))
        return True
    except Exception as e:
        st.error(f"Failed to save approval link: {e}")
        return False


def _get_approval_links() -> Dict[str, Any]:
    """Retrieve all saved approval links."""
    try:
        approvals_file = get_approvals_file()
        if approvals_file.exists():
            return json.loads(approvals_file.read_text())
        return {}
    except Exception as e:
        st.error(f"Failed to retrieve approval links: {e}")
        return {}


def _display_approval_link_info(case_id: str, approval_data: Dict[str, Any]) -> None:
    """Display approval link information in the UI."""
    st.markdown("### 📋 Approval Link Information")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Case ID", case_id)
        st.metric("Status", approval_data.get('status', 'N/A').upper())
    with col2:
        st.metric("Nominee", approval_data.get('nominee_name', 'Not specified'))
        st.metric("Created", approval_data.get('link_created_at', 'N/A')[:10])
    
    if approval_data.get('approval_link'):
        st.markdown("#### Approval Link")
        st.code(approval_data.get('approval_link'), language='text')
        
        # Copy button
        if st.button("📋 Copy Link", key=f"copy_{case_id}"):
            st.success("Link copied to clipboard!")
    
    if approval_data.get('decision'):
        st.markdown(f"#### Decision: **{approval_data.get('decision').upper()}**")
        st.metric("Decision Time", approval_data.get('timestamp', 'N/A')[:19])


def main() -> None:
    st.set_page_config(page_title="ForenSmart Consent Portal", layout="wide")
    st.markdown("## 🔐 ForenSmart Consent Portal")
    
    # Sidebar for viewing saved approval links and audit trail
    with st.sidebar:
        st.markdown("### 📊 Audit Trail & History")
        
        # Audit trail statistics
        stats = ConsentAuditTrail.get_statistics()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", stats['total_records'])
        with col2:
            st.metric("Approvals", stats['approvals'])
        with col3:
            st.metric("Denials", stats['denials'])
        
        st.divider()
        st.markdown("### 📋 Approval History")
        if st.button("🔄 Refresh Links"):
            st.rerun()
        
        approval_links = _get_approval_links()
        if approval_links:
            st.markdown(f"**Total Cases: {len(approval_links)}**")
            
            # Filter by status
            status_filter = st.selectbox("Filter by Status", ["All", "pending", "approved", "denied"])
            
            for case_id, data in approval_links.items():
                status = data.get('status', 'unknown')
                
                # Apply filter
                if status_filter != "All" and status != status_filter:
                    continue
                
                # Display case in sidebar
                status_emoji = "⏳" if status == "pending" else "✅" if status == "approved" else "❌"
                with st.expander(f"{status_emoji} {case_id}"):
                    st.write(f"**Nominee:** {data.get('nominee_name', 'Not specified')}")
                    st.write(f"**Status:** {status.upper()}")
                    st.write(f"**Created:** {data.get('link_created_at', 'N/A')[:10]}")
                    if data.get('decision'):
                        st.write(f"**Decision:** {data.get('decision').upper()}")
                    if data.get('timestamp'):
                        st.write(f"**Decided:** {data.get('timestamp', 'N/A')[:10]}")
        else:
            st.info("No approval links saved yet.")
        
        # Audit trail viewer
        st.divider()
        if st.checkbox("📊 View Audit Trail"):
            trail = ConsentAuditTrail.get_audit_trail()
            if trail:
                st.markdown("#### Recent Entries")
                for entry in trail[-10:]:  # Last 10
                    with st.expander(f"{entry['timestamp'][:10]} - {entry['case_id']} ({entry['decision'].upper()})"):
                        st.json(entry)
            else:
                st.info("No audit trail records yet")
        
        # Export audit trail
        if st.button("📥 Export Audit Trail"):
            trail = ConsentAuditTrail.get_audit_trail()
            st.download_button(
                label="Download as JSON",
                data=ConsentAuditTrail.export_audit_trail(),
                file_name=f"audit_trail_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

    params = _extract_query_params()
    
    # Support both old token-based and new data-based approaches
    approval_data = None
    token = None
    
    # Try to get approval data from URL (new approach)
    if 'data' in params:
        data_param = params.get('data')
        if isinstance(data_param, list):
            data_param = data_param[-1]
        approval_data = _decode_approval_data(data_param)
    
    # Fallback to token-based lookup (old approach)
    if not approval_data:
        token = params.get("unlock_token")
        if isinstance(token, list):
            token = token[-1]

    if not approval_data and not token:
        st.warning(
            "No approval data supplied. This page must be opened via the secure link "
            "shared by the investigator."
        )
        st.info(
            "Example: https://your-consent-app.streamlit.app/?unlock_token=TOKEN_HERE"
        )
        return

    # If we have embedded approval data, show approval form
    if approval_data:
        cm = get_consent_manager()
        case_id = approval_data.get('case_id')
        device_id = approval_data.get('device_id', 'UNKNOWN_DEVICE')
        purpose = approval_data.get('purpose', 'Investigator did not provide details.')
        requested_level_name = approval_data.get('requested_level', 'STANDARD')
        nominee_name = approval_data.get('nominee_name')
        
        # Attempt to detect device if not provided or unknown (FIX #5: Use shared method)
        if device_id == 'UNKNOWN_DEVICE' or not device_id:
            try:
                detected = cm.get_or_detect_device(case_id)
                if detected:
                    device_id = detected
                    st.info(f"✅ Device auto-detected: {device_id}")
                else:
                    st.warning("⚠️ Could not auto-detect device. Please verify manually.")
            except Exception as e:
                st.warning(f"⚠️ Device detection failed: {e}")
        
        # Create a temporary session for display
        try:
            requested_level = ConsentLevel[requested_level_name]
        except (KeyError, TypeError):
            requested_level = ConsentLevel.STANDARD
        
        st.markdown("# 🔐 ForenSmart Consent Approval")
        st.info("Review the request details below and choose whether to unlock data extraction.")
        
        st.markdown("### Case Information")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Case ID", case_id or "N/A")
            st.metric("Device ID", device_id)
        with col_b:
            st.metric("Requested Level", requested_level.name)
            st.metric("Current Level", "NONE")
        
        st.markdown("### Purpose")
        st.write(purpose)
        
        st.markdown("### Your Decision")
        st.caption("Please confirm whether you approve or deny this extraction request.")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button('✅ Yes, Approve', key='approve_btn', use_container_width=True):
                # Get current URL as the approval link
                current_url = st.query_params.get('_url', 'N/A')
                if _save_approval(case_id, 'approved', nominee_name, approval_link=str(st.query_params)):
                    # Also save the link separately for tracking
                    _save_approval_link(case_id, str(st.query_params), nominee_name)
                    
                    # Clear cache to ensure dashboard sees the approval immediately
                    try:
                        from modules.approval_sync import ApprovalSync
                        ApprovalSync.clear_cache(case_id)
                    except Exception:
                        pass
                    
                    st.success("✅ **Approval Granted** - Thank you for your consent. The investigator has been notified.")
                    st.caption(f"Nominee: {nominee_name or 'Not specified'}")
                    
                    # NEW: Show redirect message
                    st.info("🔄 **Redirecting to dashboard for automatic extraction...**")
                    st.markdown("""
                    The dashboard will automatically:
                    1. Recognize your approval
                    2. Start the extraction process
                    3. Display results in real-time
                    
                    If you're not redirected automatically, you can close this page.
                    """)
                    
                    # Use Streamlit's redirect mechanism
                    import time
                    time.sleep(2)
                    st.markdown(
                        f'<meta http-equiv="refresh" content="0; url=/?case_id={case_id}&auto_extract=true" />',
                        unsafe_allow_html=True
                    )
                    st.balloons()
                else:
                    st.error("Failed to save approval. Please try again.")
        with col2:
            if st.button('❌ No, Deny', key='deny_btn', use_container_width=True):
                # Get current URL as the approval link
                current_url = st.query_params.get('_url', 'N/A')
                if _save_approval(case_id, 'denied', nominee_name, approval_link=str(st.query_params)):
                    # Also save the link separately for tracking
                    _save_approval_link(case_id, str(st.query_params), nominee_name)
                    
                    # Clear cache to ensure dashboard sees the denial immediately
                    try:
                        from modules.approval_sync import ApprovalSync
                        ApprovalSync.clear_cache(case_id)
                    except Exception:
                        pass
                    
                    st.error("❌ **Request Denied** - Your decision has been recorded and the investigator has been notified.")
                    st.caption(f"Nominee: {nominee_name or 'Not specified'}")
                    st.info("You can close this page now.")
                else:
                    st.error("Failed to save denial. Please try again.")
        return

    # Fallback to token-based lookup
    if IMPORT_ERROR or not callable(render_nominee_approval):
        st.error(
            "Consent portal is missing the main dashboard UI components. "
            "Ensure `modules.dashboard.render_nominee_approval` is available."
        )
        if IMPORT_ERROR:
            st.code(str(IMPORT_ERROR))
        return

    cm = get_consent_manager()
    render_nominee_approval(cm, token)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "ConsentPortalLogger",
    "ConsentAuditTrail",
    "ConsentPortalEnhancer",
    "get_consent_manager",
    "main",
]


if __name__ == "__main__":
    main()
